use anyhow::{Error, Ok};
use image::{
    DynamicImage, GenericImageView,
    imageops::{self, FilterType},
};
use std::time::Instant;
use tract_ndarray::{Array2, Array4, Axis, Ix2};
use tract_onnx::prelude::*;

fn main() -> Result<(), Error> {
    let t = Instant::now();
    let model = tract_onnx::onnx()
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

    let pad_x = ((target - new_w) / 2) as i64;
    let pad_y = ((target - new_h) / 2) as i64;

    println!("Padding: x={} y={}", pad_x, pad_y);

    // save resized image for debugging
    let mut padded = DynamicImage::new_rgb8(target, target);
    imageops::replace(&mut padded, &resized, pad_x, pad_y);
    padded.save("images/padded.jpg")?;

    // create input tensor in NCHW format https://machinelearning.wtf/terms/nchw/, normalized to 0..1
    let mut input_arr = Array4::<f32>::zeros((1usize, 3usize, target as usize, target as usize));

    for y in 0..(target as usize) {
        for x in 0..(target as usize) {
            let px = padded.get_pixel(x as u32, y as u32);
            input_arr[[0, 0, y, x]] = px[0] as f32 / 255.0;
            input_arr[[0, 1, y, x]] = px[1] as f32 / 255.0;
            input_arr[[0, 2, y, x]] = px[2] as f32 / 255.0;
        }
    }

    let input_tensor: Tensor = input_arr.into();

    println!("Inference started at ... {:?}", t.elapsed());
    // run inference
    let outputs = model.run(tvec!(input_tensor.into()))?;
    println!("Inference done in {:?}", t.elapsed());

    println!("Output tensor shape: {:?}", outputs[0].shape());

    Ok(())
}
