use image::RgbImage;
use rayon::prelude::*;

use crate::pattern::{ChipGrid, pattern_values, perceptual_mask};
use crate::plane::Plane;

pub struct EmbedReport {
    pub chip: usize,
    pub psnr: f64,
    pub mean_abs_delta: f64,
    pub max_abs_delta: u8,
}

/// Adds `strength · mask · pattern` to the luminance (equally to R, G and B).
pub fn embed(img: &RgbImage, seed: &[u8; 32], chip: usize, strength: f32) -> (RgbImage, EmbedReport) {
    let luma = Plane::from_rgb_luma(img);
    let mask = perceptual_mask(&luma, chip);
    let (w, h) = (luma.w, luma.h);
    let grid = ChipGrid::new(w, h, chip);
    let vals = pattern_values(seed, grid.len());
    let cols: Vec<_> = (0..w).map(|x| grid.locate_x(x as f32)).collect();

    let mut out = img.clone();
    let stats = out
        .par_chunks_mut(w * 3)
        .enumerate()
        .map(|(y, row)| {
            let ry = grid.locate_y(y as f32);
            let (mut sq, mut abs, mut max) = (0.0f64, 0.0f64, 0u8);
            for (x, px) in row.chunks_mut(3).enumerate() {
                let d = strength * mask.get(x, y) * grid.interp(&vals, cols[x], ry);
                for v in px.iter_mut() {
                    let new = (*v as f32 + d).round().clamp(0.0, 255.0) as u8;
                    let diff = new.abs_diff(*v);
                    sq += (diff as f64).powi(2);
                    abs += diff as f64;
                    max = max.max(diff);
                    *v = new;
                }
            }
            (sq, abs, max)
        })
        .reduce(|| (0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2.max(b.2)));

    let n = (w * h * 3) as f64;
    let mse = stats.0 / n;
    let psnr = if mse > 0.0 { 10.0 * (255.0f64.powi(2) / mse).log10() } else { f64::INFINITY };
    let report = EmbedReport { chip, psnr, mean_abs_delta: stats.1 / n, max_abs_delta: stats.2 };
    (out, report)
}
