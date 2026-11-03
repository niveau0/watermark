//! Keyed pseudo-random chip pattern and the perceptual embedding mask.
//!
//! The pattern is a grid of ±1 "chips" (one value per `chip × chip` pixels of
//! the original) that is bilinearly interpolated between chip centres, which
//! avoids visible block edges. It depends only on the secret key and the copy
//! id, never on image content.

use crate::plane::Plane;

const KEY_CONTEXT: &str = "watermark 2026-10-30 pattern key v1";

/// Per-copy seed: `keyed_hash(derive_key(secret), id)`.
pub fn derive_seed(key: &str, id: &str) -> [u8; 32] {
    let k = blake3::derive_key(KEY_CONTEXT, key.as_bytes());
    *blake3::keyed_hash(&k, id.as_bytes()).as_bytes()
}

/// Default chip size in original pixels, scaled with image size so the
/// pattern survives downscaling and camera blur.
pub fn default_chip(w: usize, h: usize) -> usize {
    ((w.min(h) as f32 / 256.0).round() as usize).clamp(3, 32)
}

#[derive(Clone, Copy, Debug)]
pub struct ChipGrid {
    pub chip: f32,
    pub gw: usize,
    pub gh: usize,
}

impl ChipGrid {
    pub fn new(w: usize, h: usize, chip: usize) -> Self {
        Self { chip: chip as f32, gw: w / chip + 2, gh: h / chip + 2 }
    }

    pub fn len(&self) -> usize {
        self.gw * self.gh
    }

    /// Chip column and interpolation weight for an original x coordinate.
    #[inline]
    pub fn locate_x(&self, x: f32) -> (usize, f32) {
        locate(x, self.chip, self.gw)
    }

    #[inline]
    pub fn locate_y(&self, y: f32) -> (usize, f32) {
        locate(y, self.chip, self.gh)
    }

    #[inline]
    pub fn interp(&self, vals: &[f32], (i, fx): (usize, f32), (j, fy): (usize, f32)) -> f32 {
        let b = j * self.gw + i;
        let top = vals[b] * (1.0 - fx) + vals[b + 1] * fx;
        let bottom = vals[b + self.gw] * (1.0 - fx) + vals[b + self.gw + 1] * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

#[inline]
fn locate(x: f32, chip: f32, n: usize) -> (usize, f32) {
    let g = ((x + 0.5) / chip - 0.5).clamp(0.0, (n - 1) as f32 - 1e-4);
    let i = (g as usize).min(n - 2);
    (i, g - i as f32)
}

/// `n` chip values in {-1, +1}, derived from the seed.
pub fn pattern_values(seed: &[u8; 32], n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n.div_ceil(8)];
    blake3::Hasher::new_keyed(seed).update(b"chips").finalize_xof().fill(&mut bytes);
    (0..n).map(|i| if (bytes[i / 8] >> (i % 8)) & 1 == 1 { 1.0 } else { -1.0 }).collect()
}

/// Local embedding strength: high in textured regions (masking), low in
/// flat regions and near black/white (to avoid visible noise and clipping).
pub fn perceptual_mask(luma: &Plane, chip: usize) -> Plane {
    let r = chip.max(2);
    let mean = luma.box_blur(r);
    let sq = luma.map(|v| v * v).box_blur(r);
    let mut m = Plane::new(luma.w, luma.h);
    for i in 0..m.data.len() {
        let std = (sq.data[i] - mean.data[i] * mean.data[i]).max(0.0).sqrt();
        let texture = (std / 8.0).clamp(0.2, 2.5);
        let l = luma.data[i];
        let headroom = (l.min(255.0 - l) / 24.0).clamp(0.25, 1.0);
        m.data[i] = texture * headroom;
    }
    m.box_blur(r)
}
