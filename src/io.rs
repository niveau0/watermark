use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

/// Loads an image as RGB, honouring the EXIF orientation of camera photos.
pub fn load_rgb(path: &Path) -> Result<RgbImage> {
    let open = || -> Result<RgbImage> {
        let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
        let orientation = decoder.orientation()?;
        let mut img = DynamicImage::from_decoder(decoder)?;
        img.apply_orientation(orientation);
        Ok(img.to_rgb8())
    };
    open().with_context(|| format!("failed to load {}", path.display()))
}

pub fn save_rgb(img: &RgbImage, path: &Path, jpeg_quality: u8) -> Result<()> {
    let save = || -> Result<()> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if ext == "jpg" || ext == "jpeg" {
            let mut enc = JpegEncoder::new_with_quality(BufWriter::new(File::create(path)?), jpeg_quality);
            enc.encode_image(img)?;
        } else {
            img.save(path)?;
        }
        Ok(())
    };
    save().with_context(|| format!("failed to save {}", path.display()))
}
