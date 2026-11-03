//! Homography estimation: normalised DLT plus RANSAC.

use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen, Vector3};

pub type Hom = Matrix3<f64>;
pub type Pt = (f64, f64);

#[inline]
pub fn apply(h: &Hom, p: Pt) -> Pt {
    let v = h * Vector3::new(p.0, p.1, 1.0);
    (v.x / v.z, v.y / v.z)
}

/// Hartley normalisation: centroid to origin, mean distance √2.
fn normalizer(pts: &[Pt]) -> Hom {
    let n = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (cx, cy) = (sx / n, sy / n);
    let md = pts.iter().map(|p| (p.0 - cx).hypot(p.1 - cy)).sum::<f64>() / n;
    let s = if md > 1e-12 { std::f64::consts::SQRT_2 / md } else { 1.0 };
    Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0)
}

/// Least-squares homography mapping `src` onto `dst` (at least 4 points).
pub fn fit(src: &[Pt], dst: &[Pt]) -> Option<Hom> {
    if src.len() < 4 || src.len() != dst.len() {
        return None;
    }
    let ts = normalizer(src);
    let td = normalizer(dst);
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for (p, q) in src.iter().zip(dst) {
        let (x, y) = apply(&ts, *p);
        let (u, v) = apply(&td, *q);
        let r1 = SVector::<f64, 9>::from_column_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
        let r2 = SVector::<f64, 9>::from_column_slice(&[0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v]);
        ata += r1 * r1.transpose() + r2 * r2.transpose();
    }
    let eig = SymmetricEigen::new(ata);
    let (idx, _) = eig.eigenvalues.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))?;
    let h = eig.eigenvectors.column(idx);
    let hn = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let hm = td.try_inverse()? * hn * ts;
    let z = hm[(2, 2)];
    if !z.is_finite() || z.abs() < 1e-12 {
        return None;
    }
    let hm = hm / z;
    hm.iter().all(|v| v.is_finite()).then_some(hm)
}

pub struct RansacResult {
    pub h: Hom,
    pub inliers: Vec<bool>,
    pub count: usize,
}

pub fn ransac(src: &[Pt], dst: &[Pt], thresh: f64, iters: usize) -> Option<RansacResult> {
    let n = src.len();
    if n < 4 {
        return None;
    }
    let t2 = thresh * thresh;
    let classify = |h: &Hom| -> Vec<bool> {
        src.iter()
            .zip(dst)
            .map(|(p, q)| {
                let (x, y) = apply(h, *p);
                let e = (x - q.0).powi(2) + (y - q.1).powi(2);
                e.is_finite() && e < t2
            })
            .collect()
    };
    let count = |m: &[bool]| m.iter().filter(|&&b| b).count();

    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut best: Option<(Hom, usize)> = None;
    for _ in 0..iters {
        let mut idx = [0usize; 4];
        let mut k = 0;
        while k < 4 {
            let c = (next() % n as u64) as usize;
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let s: Vec<Pt> = idx.iter().map(|&i| src[i]).collect();
        let d: Vec<Pt> = idx.iter().map(|&i| dst[i]).collect();
        if degenerate(&s) || degenerate(&d) {
            continue;
        }
        let Some(h) = fit(&s, &d) else { continue };
        let c = count(&classify(&h));
        if best.as_ref().is_none_or(|b| c > b.1) {
            best = Some((h, c));
        }
    }

    let (mut h, _) = best?;
    let mut inliers = classify(&h);
    for _ in 0..3 {
        let (s, d): (Vec<Pt>, Vec<Pt>) =
            src.iter().zip(dst).zip(&inliers).filter(|(_, b)| **b).map(|((p, q), _)| (*p, *q)).unzip();
        match fit(&s, &d) {
            Some(refined) => {
                let m = classify(&refined);
                if count(&m) < count(&inliers) {
                    break;
                }
                h = refined;
                inliers = m;
            }
            None => break,
        }
    }
    let c = count(&inliers);
    Some(RansacResult { h, inliers, count: c })
}

fn degenerate(p: &[Pt]) -> bool {
    let area = |a: Pt, b: Pt, c: Pt| ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs();
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)].iter().any(|&(i, j, k)| area(p[i], p[j], p[k]) < 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_known_homography() {
        let h = Matrix3::new(1.1, 0.05, 12.0, -0.03, 0.95, -7.0, 1e-4, -5e-5, 1.0);
        let src: Vec<Pt> = (0..20).map(|i| ((i * 37 % 500) as f64, (i * 91 % 400) as f64)).collect();
        let dst: Vec<Pt> = src.iter().map(|&p| apply(&h, p)).collect();
        let r = ransac(&src, &dst, 0.5, 200).unwrap();
        assert_eq!(r.count, src.len());
        for p in &src {
            let (a, b) = (apply(&r.h, *p), apply(&h, *p));
            assert!((a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);
        }
    }
}
