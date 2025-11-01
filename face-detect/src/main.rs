use image::{
    DynamicImage, GenericImageView,
    imageops::{self, FilterType},
};
use std::time::Instant;
use tract_ndarray::{Array2, Array4, Axis, Ix2};
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

impl Bbox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            confidence,
        }
    }
    /// Crop the box from an owned rgb image buffer (small copy).
    pub fn crop_bbox(&self, rgb_img: &mut image::RgbImage) -> Result<DynamicImage, anyhow::Error> {
        // clamp coordinates to image bounds & ensure non-negative
        let img_w = rgb_img.width();
        let img_h = rgb_img.height();

        let mut x1 = self.x1.max(0.0).min((img_w - 1) as f32) as u32;
        let mut y1 = self.y1.max(0.0).min((img_h - 1) as f32) as u32;
        let mut x2 = self.x2.max(0.0).min(img_w as f32) as u32;
        let mut y2 = self.y2.max(0.0).min(img_h as f32) as u32;

        // Ensure width/height non-zero (expand by 1 px if needed)
        if x2 <= x1 {
            if x1 > 0 {
                x1 -= 1;
            } else {
                x2 = (x1 + 1).min(img_w);
            }
        }
        if y2 <= y1 {
            if y1 > 0 {
                y1 -= 1;
            } else {
                y2 = (y1 + 1).min(img_h);
            }
        }

        let w = x2.saturating_sub(x1);
        let h = y2.saturating_sub(y1);

        if w == 0 || h == 0 {
            return Err(anyhow::anyhow!("crop would be empty: w={} h={}", w, h));
        }

        let sub = image::imageops::crop_imm(rgb_img, x1, y1, w, h);
        Ok(DynamicImage::ImageRgb8(sub.to_image()))
    }
}

fn canonicalize_detection_output(
    tval: &tract_onnx::prelude::TValue,
    preferred_cols: Option<usize>,
) -> Result<Array2<f32>, anyhow::Error> {
    let arr_view = tval.to_array_view::<f32>()?;
    let ndim = arr_view.ndim();

    match ndim {
        2 => {
            let a = arr_view.to_owned(); // owned dynamic-dim array
            let (r, c) = (a.shape()[0], a.shape()[1]);
            if let Some(cols) = preferred_cols {
                if r == cols {
                    // (cols, rows) -> transpose
                    let t = a.t().to_owned();
                    return Ok(t.into_dimensionality::<Ix2>().unwrap());
                } else {
                    return Ok(a.into_dimensionality::<Ix2>().unwrap());
                }
            } else {
                if r <= 8 && c > r {
                    let t = a.t().to_owned();
                    return Ok(t.into_dimensionality::<Ix2>().unwrap());
                } else {
                    return Ok(a.into_dimensionality::<Ix2>().unwrap());
                }
            }
        }
        3 => {
            let shape = arr_view.shape().to_vec();
            let (d0, d1, d2) = (shape[0], shape[1], shape[2]);

            // common case: [1, N, M] -> take slice at axis 0
            if d0 == 1 {
                let two = arr_view.index_axis(Axis(0), 0).to_owned();
                let (r, c) = (two.shape()[0], two.shape()[1]);
                if let Some(cols) = preferred_cols {
                    if r == cols {
                        let t = two.t().to_owned();
                        return Ok(t.into_dimensionality::<Ix2>().unwrap());
                    } else if c == cols {
                        return Ok(two.into_dimensionality::<Ix2>().unwrap());
                    } else {
                        if r <= 8 && c > r {
                            let t = two.t().to_owned();
                            return Ok(t.into_dimensionality::<Ix2>().unwrap());
                        } else {
                            return Ok(two.into_dimensionality::<Ix2>().unwrap());
                        }
                    }
                } else {
                    if r <= 8 && c > r {
                        let t = two.t().to_owned();
                        return Ok(t.into_dimensionality::<Ix2>().unwrap());
                    } else {
                        return Ok(two.into_dimensionality::<Ix2>().unwrap());
                    }
                }
            }

            // case: [N, 1, M] -> axis(1)
            if d1 == 1 {
                let two = arr_view.index_axis(Axis(1), 0).to_owned();
                return Ok(two.into_dimensionality::<Ix2>().unwrap());
            }

            // case: [N, M, 1] -> axis(2)
            if d2 == 1 {
                let two = arr_view.index_axis(Axis(2), 0).to_owned();
                return Ok(two.into_dimensionality::<Ix2>().unwrap());
            }

            Err(anyhow::anyhow!(
                "Unhandled 3-D detection output shape: {:?}",
                shape
            ))
        }
        other => {
            let shape_opt = tval.to_array_view::<f32>().ok().map(|a| a.shape().to_vec());
            return Err(anyhow::anyhow!(
                "Unhandled detection output ndim = {} shape = {:?}",
                other,
                shape_opt
            ));
        }
    }
}

/// Faster NMS: sort descending by confidence, then mark suppressed indices
fn non_max_suppression(mut boxes: Vec<Bbox>, iou_threshold: f32) -> Vec<Bbox> {
    boxes.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let no_of_boxes = boxes.len();
    let mut suppressed = vec![false; no_of_boxes];
    let mut keep = Vec::with_capacity(no_of_boxes);
    for i in 0..no_of_boxes {
        if suppressed[i] {
            continue;
        }
        let bi = &boxes[i];
        keep.push(bi.clone());
        for j in (i + 1)..no_of_boxes {
            if suppressed[j] {
                continue;
            }
            if calculate_iou(bi, &boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

fn calculate_iou(a: &Bbox, b: &Bbox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let intersection_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union_area = area_a + area_b - intersection_area;

    if union_area <= 0.0 {
        0.0
    } else {
        intersection_area / union_area
    }
}

fn main() -> Result<(), anyhow::Error> {
    let t = Instant::now();
    let model = tract_onnx::onnx()
        .model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?
        .into_runnable()?;

    println!("Model loaded in {:?}", t.elapsed());

    // create a directory for output images
    std::fs::create_dir_all("output")?;

    // iterate over all image files in the images/ directory
    let image_dir = std::fs::read_dir("images")?;
    let mut image_count = 0;
    for entry in image_dir {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !(ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" || ext == "gif") {
            continue;
        }
        image_count += 1;
        println!("\nProcessing image {}: {:?}", image_count, path);
        let raw_img = match image::open(&path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to open image {:?}: {}", path, e);
                continue;
            }
        };
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

        // Optionally save padded image for debugging
        // let mut padded = DynamicImage::new_rgb8(target, target);
        // imageops::replace(&mut padded, &resized, pad_x, pad_y);
        // padded.save(format!("images/padded_{}.jpg", image_count))?;

        // create input tensor in NCHW format, normalized to 0..1
        let mut padded = DynamicImage::new_rgb8(target, target);
        imageops::replace(&mut padded, &resized, pad_x, pad_y);
        let mut input_arr =
            Array4::<f32>::zeros((1usize, 3usize, target as usize, target as usize));
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

        let results = canonicalize_detection_output(&outputs[0], Some(5))?;
        let (rows, cols) = (results.shape()[0], results.shape()[1]);
        println!("Detections shape: {}x{}", rows, cols);
        if cols != 5 {
            eprintln!(
                "Unexpected detection output columns: expected 5, got {}",
                cols
            );
            continue;
        }

        // detect whether coordinates appear to be normalized (0..1) or absolute (pixel values)
        let sample_h = results[[0, 0]].abs();
        let sample_w = results[[0, 0]].abs();
        let likely_normalized = sample_h <= 1.01 && sample_w <= 1.01;
        println!(
            "Detections likely normalized: {} (sample values: h={} w={})",
            likely_normalized, sample_h, sample_w
        );

        let confidence_threshold = 0.5f32;
        let mut detections: Vec<Bbox> = Vec::with_capacity(256);
        for i in 0..rows {
            let confidence = results[[i, 4]];
            if confidence < confidence_threshold {
                continue;
            }
            let mut x = results[[i, 0]];
            let mut y = results[[i, 1]];
            let mut w = results[[i, 2]];
            let mut h = results[[i, 3]];
            // If normalized, convert to absolute coordinates
            if likely_normalized {
                x *= target as f32;
                y *= target as f32;
                w *= target as f32;
                h *= target as f32;
            }
            // YOLO gives center x,y. Convert to two corners (in absolute coordinates)
            let x1_model = x - (w / 2.0);
            let y1_model = y - (h / 2.0);
            let x2_model = x + (w / 2.0);
            let y2_model = y + (h / 2.0);
            // Map from model coords (with padding) to original image coords
            let pad_x_f = pad_x as f32;
            let pad_y_f = pad_y as f32;
            let inv_scale = 1.0 / scale;
            let bx1 = (x1_model - pad_x_f) * inv_scale;
            let by1 = (y1_model - pad_y_f) * inv_scale;
            let bx2 = (x2_model - pad_x_f) * inv_scale;
            let by2 = (y2_model - pad_y_f) * inv_scale;
            detections.push(Bbox::new(bx1, by1, bx2, by2, confidence));
        }
        println!(
            "Detections count: {}, (confidence >= {})",
            detections.len(),
            confidence_threshold
        );
        // apply NMS
        let final_boxes = non_max_suppression(detections, 0.4);
        println!("Final boxes after NMS: {}", final_boxes.len());
        println!("Processing done in {:?}", t.elapsed());

        let mut rgb_owned = raw_img.to_rgb8();
        for (idx, bbox) in final_boxes.iter().enumerate() {
            match bbox.crop_bbox(&mut rgb_owned) {
                Ok(c) => {
                    let out_path = format!(
                        "output/{}_face_{}.jpg",
                        path.file_stem().and_then(|s| s.to_str()).unwrap_or("img"),
                        idx + 1
                    );
                    c.save(&out_path)?;
                    println!("Cropped face saved to {}", out_path);
                }
                Err(e) => {
                    eprintln!("Failed to crop bbox {:?}: {}", idx, e);
                }
            }
        }
        println!(
            "All cropped faces saved in output/ directory. time elapsed: {:?}",
            t.elapsed()
        );
    }
    println!("\nProcessed {} images from images/ directory.", image_count);

    println!("Total time elapsed: {:?}", t.elapsed());
    Ok(())
}
