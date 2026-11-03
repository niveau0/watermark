//! Single-channel float image ("plane") with the few filters we need.
//!
//! Coordinate convention everywhere: pixel `(x, y)` has its centre at the
//! integer position `(x, y)`. Scaling a plane by `s` therefore maps a
//! coordinate `x` to `(x + 0.5) * s - 0.5`.

use image::{GrayImage, Luma, RgbImage};
use nalgebra::Matrix3;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; w * h] }
    }

    pub fn from_rgb_luma(img: &RgbImage) -> Self {
        let (w, h) = img.dimensions();
        let data = img
            .pixels()
            .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
            .collect();
        Self { w: w as usize, h: h as usize, data }
    }

    pub fn channels(img: &RgbImage) -> [Plane; 3] {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mut out = [Plane::new(w, h), Plane::new(w, h), Plane::new(w, h)];
        for (i, p) in img.pixels().enumerate() {
            for (c, plane) in out.iter_mut().enumerate() {
                plane.data[i] = p[c] as f32;
            }
        }
        out
    }

    pub fn to_rgb(channels: &[Plane; 3]) -> RgbImage {
        let (w, h) = (channels[0].w, channels[0].h);
        RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let i = y as usize * w + x as usize;
            image::Rgb(std::array::from_fn(|c| quantize(channels[c].data[i])))
        })
    }

    pub fn to_gray_image(&self) -> GrayImage {
        GrayImage::from_fn(self.w as u32, self.h as u32, |x, y| {
            Luma([quantize(self.get(x as usize, y as usize))])
        })
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    /// Bilinear sample. Returns `None` outside the image.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> Option<f32> {
        let (maxx, maxy) = ((self.w - 1) as f32, (self.h - 1) as f32);
        if !(x >= 0.0 && y >= 0.0 && x <= maxx && y <= maxy) {
            return None;
        }
        let x0 = (x as usize).min(self.w - 2);
        let y0 = (y as usize).min(self.h - 2);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let i = y0 * self.w + x0;
        let (a, b) = (self.data[i], self.data[i + 1]);
        let (c, d) = (self.data[i + self.w], self.data[i + self.w + 1]);
        Some(a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy)
    }

    pub fn sub(&self, other: &Plane) -> Plane {
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Plane { w: self.w, h: self.h, data }
    }

    pub fn map(&self, f: impl Fn(f32) -> f32 + Sync) -> Plane {
        Plane { w: self.w, h: self.h, data: self.data.par_iter().map(|&v| f(v)).collect() }
    }

    /// Mean filter over a `(2r+1)²` window, borders use the available pixels.
    pub fn box_blur(&self, r: usize) -> Plane {
        if r == 0 {
            return self.clone();
        }
        self.separable(&move |src: &[f32], dst: &mut [f32]| {
            let n = src.len();
            let mut prefix = Vec::with_capacity(n + 1);
            let mut acc = 0.0f64;
            prefix.push(0.0);
            for &v in src {
                acc += v as f64;
                prefix.push(acc);
            }
            for (x, d) in dst.iter_mut().enumerate() {
                let a = x.saturating_sub(r);
                let b = (x + r + 1).min(n);
                *d = ((prefix[b] - prefix[a]) / (b - a) as f64) as f32;
            }
        })
    }

    pub fn gaussian_blur(&self, sigma: f32) -> Plane {
        if sigma <= 0.0 {
            return self.clone();
        }
        let r = (3.0 * sigma).ceil() as isize;
        let mut kernel: Vec<f32> =
            (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
        let sum: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= sum);
        self.separable(&move |src: &[f32], dst: &mut [f32]| {
            let n = src.len() as isize;
            for (x, d) in dst.iter_mut().enumerate() {
                let mut s = 0.0;
                for (k, &kv) in kernel.iter().enumerate() {
                    let xi = (x as isize + k as isize - r).clamp(0, n - 1);
                    s += kv * src[xi as usize];
                }
                *d = s;
            }
        })
    }

    fn separable(&self, f: &(dyn Fn(&[f32], &mut [f32]) + Sync)) -> Plane {
        let rows = map_rows(&self.data, self.w, f);
        let t = transpose(&rows, self.w, self.h);
        let cols = map_rows(&t, self.h, f);
        Plane { w: self.w, h: self.h, data: transpose(&cols, self.h, self.w) }
    }

    /// Area-averaging resize (bilinear when upscaling).
    pub fn resize_area(&self, nw: usize, nh: usize) -> Plane {
        let wx = resample_weights(self.w, nw);
        let wy = resample_weights(self.h, nh);
        let mut tmp = vec![0.0f32; nw * self.h];
        tmp.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
            let src = &self.data[y * self.w..(y + 1) * self.w];
            for (x, ws) in wx.iter().enumerate() {
                row[x] = ws.iter().map(|&(i, wt)| src[i] * wt).sum();
            }
        });
        let mut out = vec![0.0f32; nw * nh];
        out.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
            for &(j, wt) in &wy[y] {
                let src = &tmp[j * nw..(j + 1) * nw];
                for (o, s) in row.iter_mut().zip(src) {
                    *o += s * wt;
                }
            }
        });
        Plane { w: nw, h: nh, data: out }
    }

    /// Downscales so that the longer side is at most `max`.
    /// Returns the plane and the effective x/y scale factors (new / old).
    pub fn downscale_to_max(&self, max: usize) -> (Plane, f64, f64) {
        let longest = self.w.max(self.h);
        if longest <= max {
            return (self.clone(), 1.0, 1.0);
        }
        let s = max as f64 / longest as f64;
        let nw = ((self.w as f64 * s).round() as usize).max(1);
        let nh = ((self.h as f64 * s).round() as usize).max(1);
        let sx = nw as f64 / self.w as f64;
        let sy = nh as f64 / self.h as f64;
        (self.resize_area(nw, nh), sx, sy)
    }
}

/// Matrix mapping full-resolution coordinates into a plane scaled by `(sx, sy)`.
pub fn scale_matrix(sx: f64, sy: f64) -> Matrix3<f64> {
    Matrix3::new(sx, 0.0, 0.5 * sx - 0.5, 0.0, sy, 0.5 * sy - 0.5, 0.0, 0.0, 1.0)
}

#[inline]
pub fn quantize(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

fn map_rows(data: &[f32], w: usize, f: &(dyn Fn(&[f32], &mut [f32]) + Sync)) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    out.par_chunks_mut(w).zip(data.par_chunks(w)).for_each(|(o, i)| f(i, o));
    out
}

fn transpose(data: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = data[y * w + x];
        }
    }
    out
}

fn resample_weights(n_src: usize, n_dst: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = n_src as f64 / n_dst as f64;
    (0..n_dst)
        .map(|d| {
            if scale <= 1.0 {
                let s = ((d as f64 + 0.5) * scale - 0.5).clamp(0.0, (n_src - 1) as f64);
                let i0 = s.floor() as usize;
                let i1 = (i0 + 1).min(n_src - 1);
                let f = (s - i0 as f64) as f32;
                vec![(i0, 1.0 - f), (i1, f)]
            } else {
                let a = d as f64 * scale;
                let b = a + scale;
                let mut v = Vec::new();
                let mut i = a.floor() as usize;
                while (i as f64) < b && i < n_src {
                    let lo = a.max(i as f64);
                    let hi = b.min(i as f64 + 1.0);
                    if hi > lo {
                        v.push((i, ((hi - lo) / scale) as f32));
                    }
                    i += 1;
                }
                v
            }
        })
        .collect()
}
