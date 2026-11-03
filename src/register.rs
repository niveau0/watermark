//! Registration of a suspect image (photo, print scan, crop, ...) onto the
//! original: coarse homography from feature matches, then sub-pixel
//! refinement by block matching.

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::features::{extract, match_features};
use crate::homography::{Hom, Pt, apply, ransac};
use crate::plane::{Plane, scale_matrix};

const FEATURE_SIZE: usize = 1024;
const MAX_FEATURES: usize = 3000;
const MIN_INLIERS: usize = 15;
const BLOCK: usize = 32;

pub struct Alignment {
    /// Suspect resampled into the frame of the (work-resolution) original.
    pub aligned: Plane,
    pub valid: Vec<bool>,
    pub matches: usize,
    pub inliers: usize,
    pub refine_points: usize,
    pub refine_rms: f64,
    pub coverage: f64,
}

pub fn align(orig: &Plane, suspect: &Plane) -> Result<Alignment> {
    let (of, ofx, ofy) = orig.downscale_to_max(FEATURE_SIZE);
    let (sf, sfx, sfy) = suspect.downscale_to_max(FEATURE_SIZE);
    let (fa, fb) = rayon::join(|| extract(&of, MAX_FEATURES), || extract(&sf, MAX_FEATURES));
    let matches = match_features(&fa, &fb);
    let src: Vec<Pt> = matches.iter().map(|&(i, _)| fa.points[i]).collect();
    let dst: Vec<Pt> = matches.iter().map(|&(_, j)| fb.points[j]).collect();

    let coarse = ransac(&src, &dst, 3.0, 4000);
    let inliers = coarse.as_ref().map_or(0, |r| r.count);
    if inliers < MIN_INLIERS {
        bail!(
            "registration failed: only {inliers} of {} feature matches are geometrically consistent \
             (is this really a copy of the original?)",
            matches.len()
        );
    }
    let coarse = coarse.unwrap();

    // `h` maps original work coordinates to full-resolution suspect coordinates.
    let to_suspect = scale_matrix(sfx, sfy).try_inverse().unwrap();
    let mut h = to_suspect * coarse.h * scale_matrix(ofx, ofy);

    // Pre-shrink the suspect if it is much larger than the original frame to avoid aliasing.
    let ratio = local_scale(&h, orig.w as f64 / 2.0, orig.h as f64 / 2.0);
    let (source, pre) = if ratio > 1.3 {
        let nw = (suspect.w as f64 / ratio).round().max(2.0) as usize;
        let nh = (suspect.h as f64 / ratio).round().max(2.0) as usize;
        let pre = scale_matrix(nw as f64 / suspect.w as f64, nh as f64 / suspect.h as f64);
        (suspect.resize_area(nw, nh), pre)
    } else {
        (suspect.clone(), Hom::identity())
    };

    let mut refine_points = 0;
    let mut refine_rms = f64::NAN;
    for iter in 0..4 {
        let (warped, valid) = warp(&source, &(pre * h), orig.w, orig.h);
        let radius = if iter == 0 { 6 } else { 3 };
        let pairs = block_shifts(orig, &warped, &valid, radius);
        if pairs.len() < 8 {
            break;
        }
        let (s, d): (Vec<Pt>, Vec<Pt>) = pairs.into_iter().unzip();
        let Some(r) = ransac(&s, &d, 1.0, 500) else { break };
        if r.count < 8 {
            break;
        }
        let shifts: Vec<f64> = s
            .iter()
            .zip(&r.inliers)
            .filter(|(_, b)| **b)
            .map(|(p, _)| {
                let q = apply(&r.h, *p);
                (q.0 - p.0).hypot(q.1 - p.1)
            })
            .collect();
        let mean_shift = shifts.iter().sum::<f64>() / shifts.len() as f64;
        let residuals: f64 = s
            .iter()
            .zip(&d)
            .zip(&r.inliers)
            .filter(|(_, b)| **b)
            .map(|((p, q), _)| {
                let e = apply(&r.h, *p);
                (e.0 - q.0).powi(2) + (e.1 - q.1).powi(2)
            })
            .sum();
        h *= r.h;
        refine_points = r.count;
        refine_rms = (residuals / r.count as f64).sqrt();
        if mean_shift < 0.05 {
            break;
        }
    }

    let (aligned, valid) = warp(&source, &(pre * h), orig.w, orig.h);
    let coverage = valid.iter().filter(|&&v| v).count() as f64 / valid.len() as f64;
    Ok(Alignment { aligned, valid, matches: matches.len(), inliers, refine_points, refine_rms, coverage })
}

fn local_scale(h: &Hom, x: f64, y: f64) -> f64 {
    let p0 = apply(h, (x, y));
    let px = apply(h, (x + 1.0, y));
    let py = apply(h, (x, y + 1.0));
    let det = (px.0 - p0.0) * (py.1 - p0.1) - (px.1 - p0.1) * (py.0 - p0.0);
    det.abs().sqrt()
}

/// Resamples `src` into a `w × h` frame; `map` takes frame to `src` coordinates.
pub fn warp(src: &Plane, map: &Hom, w: usize, h: usize) -> (Plane, Vec<bool>) {
    let mut data = vec![0.0f32; w * h];
    let mut valid = vec![false; w * h];
    data.par_chunks_mut(w).zip(valid.par_chunks_mut(w)).enumerate().for_each(|(y, (row, vrow))| {
        for x in 0..w {
            let (sx, sy) = apply(map, (x as f64, y as f64));
            if let Some(v) = src.sample(sx as f32, sy as f32) {
                row[x] = v;
                vrow[x] = true;
            }
        }
    });
    (Plane { w, h, data }, valid)
}

/// Local translation of `p` relative to `o` per textured block (normalised
/// cross-correlation of high-passed content, parabolic sub-pixel peak).
fn block_shifts(o: &Plane, p: &Plane, valid: &[bool], r: i32) -> Vec<(Pt, Pt)> {
    // Band-pass (difference of Gaussians) keeps enough structure even in blurry copies.
    let ho = o.gaussian_blur(1.0).sub(&o.gaussian_blur(4.0));
    let hp = p.gaussian_blur(1.0).sub(&p.gaussian_blur(4.0));
    let (w, h, ru) = (o.w, o.h, r as usize);
    let mut blocks = Vec::new();
    let mut by = ru;
    while by + BLOCK + ru <= h {
        let mut bx = ru;
        while bx + BLOCK + ru <= w {
            blocks.push((bx, by));
            bx += BLOCK;
        }
        by += BLOCK;
    }

    blocks
        .par_iter()
        .filter_map(|&(bx, by)| {
            for y in by - ru..by + BLOCK + ru {
                if valid[y * w + bx - ru..y * w + bx + BLOCK + ru].iter().any(|v| !v) {
                    return None;
                }
            }
            let n = (BLOCK * BLOCK) as f32;
            let mut tpl = Vec::with_capacity(BLOCK * BLOCK);
            for y in by..by + BLOCK {
                tpl.extend_from_slice(&ho.data[y * w + bx..y * w + bx + BLOCK]);
            }
            let mean = tpl.iter().sum::<f32>() / n;
            tpl.iter_mut().for_each(|v| *v -= mean);
            let var_o: f32 = tpl.iter().map(|v| v * v).sum();
            if var_o / n < 1.0 {
                return None;
            }

            let size = (2 * r + 1) as usize;
            let mut scores = vec![-1.0f32; size * size];
            for dy in -r..=r {
                for dx in -r..=r {
                    let (mut sp, mut spp, mut sop) = (0.0f32, 0.0f32, 0.0f32);
                    for y in 0..BLOCK {
                        let row = ((by + y) as i32 + dy) as usize * w;
                        let start = row + (bx as i32 + dx) as usize;
                        let prow = &hp.data[start..start + BLOCK];
                        for (a, b) in tpl[y * BLOCK..(y + 1) * BLOCK].iter().zip(prow) {
                            sp += b;
                            spp += b * b;
                            sop += a * b;
                        }
                    }
                    let var_p = spp - sp * sp / n;
                    if var_p > 1e-6 {
                        scores[(dy + r) as usize * size + (dx + r) as usize] = sop / (var_o * var_p).sqrt();
                    }
                }
            }
            let (bi, &best) = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
            let (ix, iy) = (bi % size, bi / size);
            if best < 0.4 || ix == 0 || iy == 0 || ix == size - 1 || iy == size - 1 {
                return None;
            }
            let sub = |m: f32, c: f32, pl: f32| {
                let den = m - 2.0 * c + pl;
                if den.abs() < 1e-9 { 0.0 } else { (0.5 * (m - pl) / den).clamp(-0.5, 0.5) }
            };
            let dx = ix as f32 - r as f32 + sub(scores[bi - 1], best, scores[bi + 1]);
            let dy = iy as f32 - r as f32 + sub(scores[bi - size], best, scores[bi + size]);
            let c = (bx as f64 + BLOCK as f64 / 2.0 - 0.5, by as f64 + BLOCK as f64 / 2.0 - 0.5);
            Some((c, (c.0 + dx as f64, c.1 + dy as f64)))
        })
        .collect()
}
