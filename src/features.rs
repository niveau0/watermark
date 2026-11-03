//! Minimal ORB-style features: FAST corners on an image pyramid,
//! intensity-centroid orientation and rotated 256-bit BRIEF descriptors.

use std::sync::OnceLock;

use imageproc::corners::corners_fast9;
use rayon::prelude::*;

use crate::homography::Pt;
use crate::plane::Plane;

pub struct Features {
    /// Keypoint positions in level-0 coordinates.
    pub points: Vec<Pt>,
    pub desc: Vec<[u64; 4]>,
}

const LEVELS: usize = 6;
const LEVEL_SCALE: f64 = 1.3;
/// Must exceed the largest rotated BRIEF offset (13·√2) and the orientation radius.
const BORDER: u32 = 20;
const ORIENT_R: i32 = 15;
const CELL: u32 = 12;
const MAX_HAMMING: u32 = 64;
const RATIO: f32 = 0.8;

fn brief_pairs() -> &'static [[i8; 4]; 256] {
    static PAIRS: OnceLock<[[i8; 4]; 256]> = OnceLock::new();
    PAIRS.get_or_init(|| {
        let mut bytes = [0u8; 256 * 8];
        blake3::Hasher::new().update(b"watermark brief pattern v1").finalize_xof().fill(&mut bytes);
        let mut out = [[0i8; 4]; 256];
        for (k, pair) in out.iter_mut().enumerate() {
            for (c, v) in pair.iter_mut().enumerate() {
                let a = bytes[k * 8 + c * 2] as i32;
                let b = bytes[k * 8 + c * 2 + 1] as i32;
                // Triangular distribution in [-13, 13], denser near the centre.
                *v = ((a + b - 255) * 13 / 255) as i8;
            }
        }
        out
    })
}

pub fn extract(img: &Plane, max_features: usize) -> Features {
    let areas: Vec<f64> = (0..LEVELS).map(|l| LEVEL_SCALE.powi(-2 * l as i32)).collect();
    let total: f64 = areas.iter().sum();
    let mut points = Vec::new();
    let mut desc = Vec::new();

    for (l, area) in areas.iter().enumerate() {
        let s = LEVEL_SCALE.powi(l as i32);
        let nw = (img.w as f64 / s).round() as usize;
        let nh = (img.h as f64 / s).round() as usize;
        if nw < 4 * BORDER as usize || nh < 4 * BORDER as usize {
            break;
        }
        let level = if l == 0 { img.clone() } else { img.resize_area(nw, nh) };
        let (sx, sy) = (img.w as f64 / nw as f64, img.h as f64 / nh as f64);
        let budget = (max_features as f64 * area / total).ceil() as usize;

        let gray = level.to_gray_image();
        let mut corners = corners_fast9(&gray, 20);
        if corners.len() < budget * 2 {
            corners = corners_fast9(&gray, 8);
        }
        let (bw, bh) = (nw as u32 - BORDER, nh as u32 - BORDER);
        corners.retain(|c| c.x >= BORDER && c.y >= BORDER && c.x < bw && c.y < bh);
        corners.sort_by(|a, b| b.score.total_cmp(&a.score));

        // Keep the strongest corner per grid cell for an even spatial spread.
        let gw = nw as u32 / CELL + 1;
        let mut occupied = vec![false; (gw * (nh as u32 / CELL + 1)) as usize];
        let mut selected = Vec::with_capacity(budget);
        for c in corners {
            let cell = ((c.y / CELL) * gw + c.x / CELL) as usize;
            if !occupied[cell] {
                occupied[cell] = true;
                selected.push(c);
                if selected.len() >= budget {
                    break;
                }
            }
        }

        let smooth = level.gaussian_blur(1.6);
        let res: Vec<(Pt, [u64; 4])> = selected
            .par_iter()
            .map(|c| {
                let (x, y) = (c.x as i32, c.y as i32);
                let d = describe(&smooth, x, y, orientation(&smooth, x, y));
                (((c.x as f64 + 0.5) * sx - 0.5, (c.y as f64 + 0.5) * sy - 0.5), d)
            })
            .collect();
        for (p, d) in res {
            points.push(p);
            desc.push(d);
        }
    }
    Features { points, desc }
}

fn orientation(p: &Plane, x: i32, y: i32) -> f32 {
    let (mut m10, mut m01) = (0.0f32, 0.0f32);
    for dy in -ORIENT_R..=ORIENT_R {
        for dx in -ORIENT_R..=ORIENT_R {
            if dx * dx + dy * dy > ORIENT_R * ORIENT_R {
                continue;
            }
            let v = p.get((x + dx) as usize, (y + dy) as usize);
            m10 += dx as f32 * v;
            m01 += dy as f32 * v;
        }
    }
    m01.atan2(m10)
}

fn describe(p: &Plane, x: i32, y: i32, angle: f32) -> [u64; 4] {
    let (s, c) = angle.sin_cos();
    let at = |px: i8, py: i8| {
        let (px, py) = (px as f32, py as f32);
        let rx = (c * px - s * py).round() as i32;
        let ry = (s * px + c * py).round() as i32;
        p.get((x + rx) as usize, (y + ry) as usize)
    };
    let mut d = [0u64; 4];
    for (k, q) in brief_pairs().iter().enumerate() {
        if at(q[0], q[1]) < at(q[2], q[3]) {
            d[k / 64] |= 1 << (k % 64);
        }
    }
    d
}

#[inline]
fn hamming(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Mutual nearest neighbours that also pass Lowe's ratio test.
pub fn match_features(a: &Features, b: &Features) -> Vec<(usize, usize)> {
    if a.desc.is_empty() || b.desc.is_empty() {
        return Vec::new();
    }
    let ab: Vec<Option<usize>> = a
        .desc
        .par_iter()
        .map(|da| {
            let (mut b1, mut b2, mut bi) = (u32::MAX, u32::MAX, 0);
            for (j, db) in b.desc.iter().enumerate() {
                let d = hamming(da, db);
                if d < b1 {
                    (b2, b1, bi) = (b1, d, j);
                } else if d < b2 {
                    b2 = d;
                }
            }
            (b1 <= MAX_HAMMING && (b1 as f32) < RATIO * b2 as f32).then_some(bi)
        })
        .collect();
    let ba: Vec<usize> = b
        .desc
        .par_iter()
        .map(|db| a.desc.iter().enumerate().min_by_key(|(_, da)| hamming(da, db)).map_or(0, |(i, _)| i))
        .collect();
    ab.iter().enumerate().filter_map(|(i, m)| m.filter(|&j| ba[j] == i).map(|j| (i, j))).collect()
}
