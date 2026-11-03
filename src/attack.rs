//! Simulated "copy" distortions for testing: geometry, light, blur, noise.
//! JPEG compression happens when the result is saved as `.jpg`.

use image::RgbImage;
use rayon::prelude::*;

use crate::homography::{Pt, apply, fit};
use crate::plane::Plane;

pub struct AttackParams {
    pub scale: f64,
    pub rotate: f64,
    pub perspective: f64,
    pub crop: f64,
    pub gamma: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub blur: f32,
    pub noise: f32,
    pub seed: u64,
}

pub fn attack(img: &RgbImage, p: &AttackParams) -> RgbImage {
    let (w, h) = (img.width() as f64, img.height() as f64);
    let mut chans = Plane::channels(img);
    if p.scale < 1.0 {
        let sigma = (0.5 / p.scale - 0.5) as f32;
        chans = chans.map(|c| c.gaussian_blur(sigma));
    }

    // Corners of the source, moved by perspective (narrower top edge), scale and rotation.
    let src: [Pt; 4] = [(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)];
    let k = p.perspective;
    let jitter = [(k * w, k * h * 0.5), (-k * w, k * h * 0.5), (0.0, 0.0), (0.0, 0.0)];
    let (sn, cs) = p.rotate.to_radians().sin_cos();
    let (cx, cy) = (w / 2.0, h / 2.0);
    let mut dst: Vec<Pt> = src
        .iter()
        .zip(jitter)
        .map(|(&(x, y), (jx, jy))| {
            let (x, y) = ((x + jx - cx) * p.scale, (y + jy - cy) * p.scale);
            (cs * x - sn * y, sn * x + cs * y)
        })
        .collect();
    let minx = dst.iter().map(|d| d.0).fold(f64::MAX, f64::min);
    let maxx = dst.iter().map(|d| d.0).fold(f64::MIN, f64::max);
    let miny = dst.iter().map(|d| d.1).fold(f64::MAX, f64::min);
    let maxy = dst.iter().map(|d| d.1).fold(f64::MIN, f64::max);
    let (fw, fh) = (maxx - minx + 1.0, maxy - miny + 1.0);
    let (ow, oh) = ((fw * p.crop).round().max(8.0), (fh * p.crop).round().max(8.0));
    let (ox, oy) = (minx + (fw - ow) / 2.0, miny + (fh - oh) / 2.0);
    dst.iter_mut().for_each(|d| *d = (d.0 - ox, d.1 - oy));

    let inv = fit(&src, &dst).and_then(|m| m.try_inverse()).expect("invalid attack geometry");
    let (ow, oh) = (ow as usize, oh as usize);
    let mut out: [Plane; 3] = std::array::from_fn(|_| Plane::new(ow, oh));
    for (c, plane) in out.iter_mut().enumerate() {
        let source = &chans[c];
        plane.data.par_chunks_mut(ow).enumerate().for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                let (sx, sy) = apply(&inv, (x as f64, y as f64));
                *v = source.sample(sx as f32, sy as f32).unwrap_or(30.0);
            }
        });
    }

    let mut rng = p.seed.max(1);
    for plane in out.iter_mut() {
        *plane = plane.map(|v| {
            let v = 255.0 * (v / 255.0).clamp(0.0, 1.0).powf(p.gamma);
            (v - 128.0) * p.contrast + 128.0 + p.brightness
        });
        if p.blur > 0.0 {
            *plane = plane.gaussian_blur(p.blur);
        }
        if p.noise > 0.0 {
            for v in plane.data.iter_mut() {
                *v += p.noise * gaussian(&mut rng);
            }
        }
    }
    Plane::to_rgb(&out)
}

fn gaussian(state: &mut u64) -> f32 {
    let mut uniform = || {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let (u1, u2) = (uniform(), uniform());
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}
