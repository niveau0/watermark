mod attack;
mod detect;
mod embed;
mod features;
mod homography;
mod io;
mod pattern;
mod plane;
mod register;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use crate::attack::AttackParams;
use crate::detect::{DetectOptions, p_value};
use crate::pattern::{default_chip, derive_seed};

#[derive(Parser)]
#[command(name = "watermark", version, about = "Invisible, copy-robust image watermarks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Embed an invisible watermark for one copy (e.g. one buyer) into an image.
    Embed(EmbedArgs),
    /// Check whether a suspect image (photo, scan, crop, ...) carries a copy's watermark.
    Detect(DetectArgs),
    /// Simulate a copy (perspective, scale, light, blur, noise, JPEG) for testing.
    Attack(AttackArgs),
}

#[derive(Args)]
struct EmbedArgs {
    /// Unmarked original image.
    #[arg(short, long)]
    input: PathBuf,
    /// Output path (.png recommended, .jpg uses --jpeg-quality).
    #[arg(short, long)]
    output: PathBuf,
    /// Secret key; all patterns are derived from it.
    #[arg(long, env = "WATERMARK_KEY", hide_env_values = true)]
    key: String,
    /// Identifier of this copy, e.g. buyer or edition number.
    #[arg(long)]
    id: String,
    /// Chip size in pixels [default: shorter side / 256, clamped to 3..=32].
    #[arg(long)]
    chip: Option<usize>,
    /// Embedding amplitude in 8-bit luminance steps (scaled by the texture mask).
    #[arg(long, default_value_t = 3.0)]
    strength: f32,
    #[arg(long, default_value_t = 95)]
    jpeg_quality: u8,
}

#[derive(Args)]
struct DetectArgs {
    /// Unmarked original image (the same one that was passed to `embed`).
    #[arg(long)]
    original: PathBuf,
    /// Image to check.
    #[arg(long)]
    suspect: PathBuf,
    #[arg(long, env = "WATERMARK_KEY", hide_env_values = true)]
    key: String,
    /// Candidate copy ids to test (repeatable).
    #[arg(long = "id", required = true)]
    ids: Vec<String>,
    /// Chip size used during embedding [default: same rule as `embed`].
    #[arg(long)]
    chip: Option<usize>,
    /// Longest side of the working resolution.
    #[arg(long, default_value_t = 2048)]
    work_size: usize,
    /// z-score above which a watermark counts as present.
    #[arg(long, default_value_t = 5.0)]
    threshold: f64,
    /// Number of random keys used as a sanity check of the null distribution.
    #[arg(long, default_value_t = 32)]
    nulls: usize,
    /// Write aligned and residual images into this directory.
    #[arg(long)]
    debug_dir: Option<PathBuf>,
}

#[derive(Args)]
struct AttackArgs {
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    scale: f64,
    /// Rotation in degrees.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    rotate: f64,
    /// Keystone strength as a fraction of the width (e.g. 0.08).
    #[arg(long, default_value_t = 0.0)]
    perspective: f64,
    /// Fraction of the result kept around the centre (1.0 = no crop).
    #[arg(long, default_value_t = 1.0)]
    crop: f64,
    #[arg(long, default_value_t = 1.0)]
    gamma: f32,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    brightness: f32,
    #[arg(long, default_value_t = 1.0)]
    contrast: f32,
    /// Gaussian blur sigma in output pixels.
    #[arg(long, default_value_t = 0.0)]
    blur: f32,
    /// Gaussian noise sigma in 8-bit steps.
    #[arg(long, default_value_t = 0.0)]
    noise: f32,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value_t = 85)]
    jpeg_quality: u8,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Embed(a) => run_embed(a),
        Command::Detect(a) => run_detect(a),
        Command::Attack(a) => run_attack(a),
    }
}

fn run_embed(a: EmbedArgs) -> Result<()> {
    let img = io::load_rgb(&a.input)?;
    let chip = a.chip.unwrap_or_else(|| default_chip(img.width() as usize, img.height() as usize));
    let (marked, r) = embed::embed(&img, &derive_seed(&a.key, &a.id), chip, a.strength);
    io::save_rgb(&marked, &a.output, a.jpeg_quality)?;
    println!("Embedded id '{}' into {}", a.id, a.output.display());
    println!(
        "  chip {} px, PSNR {:.1} dB, mean |Δ| {:.2}, max |Δ| {}",
        r.chip, r.psnr, r.mean_abs_delta, r.max_abs_delta
    );
    Ok(())
}

fn run_detect(a: DetectArgs) -> Result<()> {
    let start = Instant::now();
    let original = io::load_rgb(&a.original)?;
    let suspect = io::load_rgb(&a.suspect)?;
    let report = detect::detect(
        &original,
        &suspect,
        &DetectOptions {
            key: &a.key,
            ids: &a.ids,
            chip: a.chip,
            work_size: a.work_size,
            nulls: a.nulls,
            debug_dir: a.debug_dir.as_deref(),
        },
    )?;

    let al = &report.alignment;
    let refine = if al.refine_points > 0 {
        format!("refined with {} blocks (rms {:.2} px)", al.refine_points, al.refine_rms)
    } else {
        "no sub-pixel refinement possible".to_string()
    };
    println!(
        "Alignment: {} feature matches, {} inliers; {refine}; coverage {:.1} %",
        al.matches,
        al.inliers,
        100.0 * al.coverage
    );
    println!(
        "Analysis: {}x{} work resolution, chip {} px, {:.1} % of the original usable",
        report.work.0,
        report.work.1,
        report.chip,
        100.0 * report.used
    );
    if let Some(n) = &report.null {
        println!(
            "Null check ({} random keys): mean {:+.2}, std {:.2}, max |z| {:.2}  (expected ≈ 0, ≈ 1)",
            a.nulls, n.mean, n.std, n.max_abs
        );
    }
    println!();
    let width = report.scores.iter().map(|s| s.id.len()).max().unwrap_or(2).max(2);
    println!("{:<width$}  {:>8}  {:>10}  result", "id", "z-score", "p-value");
    for s in &report.scores {
        let verdict = if s.z >= a.threshold { "DETECTED" } else { "-" };
        println!("{:<width$}  {:>8.2}  {:>10.2e}  {verdict}", s.id, s.z, p_value(s.z));
    }
    println!("\n({:.1} s)", start.elapsed().as_secs_f64());
    Ok(())
}

fn run_attack(a: AttackArgs) -> Result<()> {
    let img = io::load_rgb(&a.input)?;
    let out = attack::attack(
        &img,
        &AttackParams {
            scale: a.scale,
            rotate: a.rotate,
            perspective: a.perspective,
            crop: a.crop,
            gamma: a.gamma,
            brightness: a.brightness,
            contrast: a.contrast,
            blur: a.blur,
            noise: a.noise,
            seed: a.seed,
        },
    );
    io::save_rgb(&out, &a.output, a.jpeg_quality)?;
    println!("Wrote {} ({}x{})", a.output.display(), out.width(), out.height());
    Ok(())
}
