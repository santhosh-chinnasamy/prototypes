use anyhow::{Error, Ok};
use image::DynamicImage;
use std::time::Instant;
use tract_onnx::prelude::*;

fn main() -> Result<(), Error> {
    let t = Instant::now();
    let _model = tract_onnx::onnx()
        .model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?
        .into_runnable()?;

    println!("Model loaded in {:?}", t.elapsed());

    Ok(())
}
