//! Non-blind detection: align the suspect to the original, remove the image
//! content, and correlate what is left with each candidate's pattern.
//!
//! The residual is projected back onto the chip grid (`g`). For a pattern
//! `p` of independent ±1 chips the score `Σ g·p / ‖g‖` is ~N(0, 1) when the
//! pattern is not present, so the score is directly a z-value.

use std::path::Path;

use anyhow::{Result, bail};
use image::RgbImage;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::pattern::{ChipGrid, default_chip, derive_seed, pattern_values, perceptual_mask};
use crate::plane::Plane;
use crate::register::{Alignment, align};

const BLOCK: usize = 32;

pub struct DetectOptions<'a> {
    pub key: &'a str,
    pub ids: &'a [String],
    pub chip: Option<usize>,
    pub work_size: usize,
    pub nulls: usize,
    pub debug_dir: Option<&'a Path>,
}

pub struct Score {
    pub id: String,
    pub z: f64,
}

pub struct NullStats {
    pub mean: f64,
    pub std: f64,
    pub max_abs: f64,
}

pub struct DetectReport {
    pub alignment: Alignment,
    pub chip: usize,
    pub work: (usize, usize),
    pub used: f64,
    pub scores: Vec<Score>,
    pub null: Option<NullStats>,
}

pub fn detect(original: &RgbImage, suspect: &RgbImage, opt: &DetectOptions) -> Result<DetectReport> {
    let o_full = Plane::from_rgb_luma(original);
    let s_full = Plane::from_rgb_luma(suspect);
    let chip = opt.chip.unwrap_or_else(|| default_chip(o_full.w, o_full.h));

    let (o_w, sx, sy) = o_full.downscale_to_max(opt.work_size);
    let mask_full = perceptual_mask(&o_full, chip);
    let mask_w = if o_w.w == o_full.w && o_w.h == o_full.h {
        mask_full
    } else {
        mask_full.resize_area(o_w.w, o_w.h)
    };
    let (ww, wh) = (o_w.w, o_w.h);

    let al = align(&o_w, &s_full)?;

    let (residual, used) = content_residual(&o_w, &al);
    let chip_w = chip as f64 * sx.min(sy);
    let residual = whiten(&residual, &used, ((2.0 * chip_w).round() as usize).max(2));
    let used_fraction = used.iter().filter(|&&u| u).count() as f64 / used.len() as f64;

    if let Some(dir) = opt.debug_dir {
        std::fs::create_dir_all(dir)?;
        o_w.to_gray_image().save(dir.join("original.png"))?;
        al.aligned.to_gray_image().save(dir.join("aligned.png"))?;
        residual.map(|v| 128.0 + 32.0 * v).to_gray_image().save(dir.join("residual.png"))?;
    }

    // Back-project the weighted residual onto the chip grid.
    let grid = ChipGrid::new(o_full.w, o_full.h, chip);
    let cols: Vec<_> = (0..ww).map(|u| grid.locate_x(((u as f64 + 0.5) / sx - 0.5) as f32)).collect();
    let rows: Vec<_> = (0..wh).map(|v| grid.locate_y(((v as f64 + 0.5) / sy - 0.5) as f32)).collect();
    let mut g = vec![0.0f64; grid.len()];
    for (v, &(j, fy)) in rows.iter().enumerate() {
        for (u, &(i, fx)) in cols.iter().enumerate() {
            let k = v * ww + u;
            if !used[k] {
                continue;
            }
            let val = (residual.data[k] * mask_w.data[k]) as f64;
            let (fx, fy) = (fx as f64, fy as f64);
            let b = j * grid.gw + i;
            g[b] += val * (1.0 - fx) * (1.0 - fy);
            g[b + 1] += val * fx * (1.0 - fy);
            g[b + grid.gw] += val * (1.0 - fx) * fy;
            g[b + grid.gw + 1] += val * fx * fy;
        }
    }
    let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm <= 0.0 {
        bail!("no usable overlap between original and suspect");
    }

    let score = |id: &str| -> f64 {
        let p = pattern_values(&derive_seed(opt.key, id), grid.len());
        g.iter().zip(&p).map(|(a, &b)| a * b as f64).sum::<f64>() / norm
    };

    let mut scores: Vec<Score> = opt.ids.par_iter().map(|id| Score { id: id.clone(), z: score(id) }).collect();
    scores.sort_by(|a, b| b.z.total_cmp(&a.z));

    let null = (opt.nulls > 1).then(|| {
        let zs: Vec<f64> = (0..opt.nulls).into_par_iter().map(|i| score(&format!("\0null-{i}"))).collect();
        let mean = zs.iter().sum::<f64>() / zs.len() as f64;
        let std = (zs.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / (zs.len() - 1) as f64).sqrt();
        let max_abs = zs.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        NullStats { mean, std, max_abs }
    });

    Ok(DetectReport { alignment: al, chip, work: (ww, wh), used: used_fraction, scores, null })
}

/// Per block, fits `suspect ≈ a·original + c·highpass(original) + b` (absorbs
/// exposure, contrast, gamma and blur/sharpening locally) and returns what
/// the original cannot explain.
fn content_residual(o: &Plane, al: &Alignment) -> (Plane, Vec<bool>) {
    let w = o.w;
    let hp = o.sub(&o.gaussian_blur(1.0));
    let mut res = vec![0.0f32; o.data.len()];
    let mut used = vec![false; o.data.len()];
    res.par_chunks_mut(BLOCK * w).zip(used.par_chunks_mut(BLOCK * w)).enumerate().for_each(|(bi, (rr, ur))| {
        let y0 = bi * BLOCK;
        let rows = rr.len() / w;
        for bx in (0..w).step_by(BLOCK) {
            let x1 = (bx + BLOCK).min(w);
            let mut m = Matrix3::<f64>::zeros();
            let mut v = Vector3::<f64>::zeros();
            let mut n = 0usize;
            for yy in 0..rows {
                for x in bx..x1 {
                    let i = (y0 + yy) * w + x;
                    if al.valid[i] {
                        let f = Vector3::new(o.data[i] as f64, hp.data[i] as f64, 1.0);
                        m += f * f.transpose();
                        v += f * al.aligned.data[i] as f64;
                        n += 1;
                    }
                }
            }
            if n * 10 < rows * (x1 - bx) * 8 {
                continue;
            }
            m[(0, 0)] += 1e-2 * n as f64;
            m[(1, 1)] += 1e-2 * n as f64;
            let Some(c) = m.lu().solve(&v) else { continue };
            for yy in 0..rows {
                for x in bx..x1 {
                    let i = (y0 + yy) * w + x;
                    if al.valid[i] {
                        let pred = c[0] * o.data[i] as f64 + c[1] * hp.data[i] as f64 + c[2];
                        rr[yy * w + x] = (al.aligned.data[i] as f64 - pred) as f32;
                        ur[yy * w + x] = true;
                    }
                }
            }
        }
    });
    (Plane { w, h: o.h, data: res }, used)
}

/// Removes low frequencies and normalises the residual's local noise level so
/// that no single region (edges, misaligned parts) dominates the score.
fn whiten(r: &Plane, used: &[bool], radius: usize) -> Plane {
    let low = r.box_blur(radius);
    let mut out = r.sub(&low);
    for (v, &u) in out.data.iter_mut().zip(used) {
        if !u {
            *v = 0.0;
        }
    }
    let w = out.w;
    for by in (0..out.h).step_by(BLOCK) {
        for bx in (0..w).step_by(BLOCK) {
            let (y1, x1) = ((by + BLOCK).min(out.h), (bx + BLOCK).min(w));
            let (mut s2, mut n) = (0.0f64, 0usize);
            for y in by..y1 {
                for x in bx..x1 {
                    if used[y * w + x] {
                        s2 += (out.data[y * w + x] as f64).powi(2);
                        n += 1;
                    }
                }
            }
            let std = if n > 16 { (s2 / n as f64).sqrt() as f32 } else { 0.0 };
            for y in by..y1 {
                for x in bx..x1 {
                    let v = &mut out.data[y * w + x];
                    *v = if std > 1e-3 { (*v / std).clamp(-4.0, 4.0) } else { 0.0 };
                }
            }
        }
    }
    out
}

/// One-sided p-value of a standard normal z-score.
pub fn p_value(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    // Numerical Recipes `erfcc`, fractional error < 1.2e-7.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let r = t * poly.exp();
    if x >= 0.0 { r } else { 2.0 - r }
}
