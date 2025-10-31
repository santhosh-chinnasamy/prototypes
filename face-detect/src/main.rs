use anyhow::{Error, Ok};
use image::{
    DynamicImage,
    imageops::{self, FilterType},
};
use std::{time::Instant};
use tract_onnx::prelude::*;

fn main() -> Result<(), Error> {
    let t = Instant::now();
    let _model = tract_onnx::onnx()
        .model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?
        .into_runnable()?;

    println!("Model loaded in {:?}", t.elapsed());

    // load image and preprocess
    let raw_img = image::open("images/group.jpg")?;
    let (orig_w, orig_h) = (raw_img.width(), raw_img.height());

    println!("Image original size: {}x{}", orig_w, orig_h);

    // letterbox resize to 640x640
    let target = 640u32;
    let scale = target as f32 / orig_w.max(orig_h) as f32;
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    println!("Resized to: {}x{}", new_w, new_h);

    let resized = imageops::resize(&raw_img, new_w, new_h, FilterType::Triangle);

    let pad_x = (target - new_w) / 2 as u32;
    let pad_y = (target - new_h) / 2 as u32;

    println!("Padding: x={} y={}", pad_x, pad_y);

    // save resized image for debugging
    let mut letterboxed = DynamicImage::new_rgb8(target, target);
    imageops::replace(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);
    letterboxed.save("images/letterboxed.jpg")?;

    Ok(())
}
