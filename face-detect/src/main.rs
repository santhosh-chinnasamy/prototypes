use anyhow::Result;
use image::{
    DynamicImage, GenericImageView,
    imageops::{self, FilterType},
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tract_ndarray::{Array2, Array4, Axis, Ix2};
use tract_onnx::prelude::*;

/// Reuse your existing Bbox impl (unchanged except Clone + Debug derived).
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

/// Reuse your canonicalize_detection_output, nms, iou, etc.
/// (For brevity in this example they are included as functions,
/// identical to your original; paste your original implementations.)
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
            Err(anyhow::anyhow!(
                "Unhandled detection output ndim = {} shape = {:?}",
                other,
                shape_opt
            ))
        }
    }
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

fn non_max_suppression(mut boxes: Vec<Bbox>, iou_threshold: f32) -> Vec<Bbox> {
    boxes.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = boxes.len();
    let mut suppressed = vec![false; n];
    let mut keep = Vec::with_capacity(n);
    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        let bi = &boxes[i];
        keep.push(bi.clone());
        for j in (i + 1)..n {
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

/// Pipeline context: stage inputs/outputs share a Context object.
pub struct Context {
    pub path: PathBuf,
    pub raw_img: Option<DynamicImage>,
    pub padded: Option<DynamicImage>,
    pub input_arr: Option<Array4<f32>>,
    pub scale: f32,
    pub pad_x: i64,
    pub pad_y: i64,
    pub target: u32,
    pub results: Option<Array2<f32>>,
    pub final_boxes: Option<Vec<Bbox>>,
}

impl Context {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            raw_img: None,
            padded: None,
            input_arr: None,
            scale: 1.0,
            pad_x: 0,
            pad_y: 0,
            target: 640,
            results: None,
            final_boxes: None,
        }
    }
}

/// Stage trait: each stage mutates Context.
/// `shared` holds read-only shared resources such as the loaded model.
pub trait Stage {
    fn name(&self) -> &str;
    fn run(&self, ctx: &mut Context, shared: &Shared) -> Result<()>;
}

/// Shared resources passed to stages.
pub struct Shared {
    pub model: Arc<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>,
}

impl Shared {
    pub fn new(
        model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    ) -> Self {
        Self {
            model: Arc::new(model),
        }
    }
}

/// Stage implementations follow. Small focused responsibilities.

/// Loads image from disk into Context::raw_img.
pub struct LoadImageStage;
impl Stage for LoadImageStage {
    fn name(&self) -> &str {
        "LoadImage"
    }
    fn run(&self, ctx: &mut Context, _shared: &Shared) -> Result<()> {
        let img = image::open(&ctx.path)?;
        ctx.raw_img = Some(img);
        Ok(())
    }
}

/// Prepares padded image and input tensor (same logic as your prepare_input).
pub struct PrepareInputStage;
impl Stage for PrepareInputStage {
    fn name(&self) -> &str {
        "PrepareInput"
    }
    fn run(&self, ctx: &mut Context, _shared: &Shared) -> Result<()> {
        let raw = ctx
            .raw_img
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("raw_img missing"))?;
        let target = ctx.target;
        let (orig_w, orig_h) = (raw.width(), raw.height());
        let scale = target as f32 / orig_w.max(orig_h) as f32;
        let new_w = (orig_w as f32 * scale).round() as u32;
        let new_h = (orig_h as f32 * scale).round() as u32;
        let resized = imageops::resize(raw, new_w, new_h, FilterType::Triangle);
        let pad_x = ((target - new_w) / 2) as i64;
        let pad_y = ((target - new_h) / 2) as i64;
        let mut padded = DynamicImage::new_rgb8(target, target);
        imageops::replace(&mut padded, &resized, pad_x, pad_y);

        // Build input array
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

        ctx.padded = Some(padded);
        ctx.input_arr = Some(input_arr);
        ctx.scale = scale;
        ctx.pad_x = pad_x;
        ctx.pad_y = pad_y;
        Ok(())
    }
}

/// Runs inference using shared.model, stores canonicalized Array2 into ctx.results.
pub struct InferenceStage;
impl Stage for InferenceStage {
    fn name(&self) -> &str {
        "Inference"
    }
    fn run(&self, ctx: &mut Context, shared: &Shared) -> Result<()> {
        let input_arr = ctx
            .input_arr
            .take()
            .ok_or_else(|| anyhow::anyhow!("input_arr missing"))?;
        let input_tensor: Tensor = input_arr.into();
        let outputs = shared.model.run(tvec!(input_tensor.into()))?;
        let results = canonicalize_detection_output(&outputs[0], Some(5))?;
        ctx.results = Some(results);
        Ok(())
    }
}

/// Postprocess detections -> boxes + NMS
pub struct PostprocessStage {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
}
impl Stage for PostprocessStage {
    fn name(&self) -> &str {
        "Postprocess"
    }
    fn run(&self, ctx: &mut Context, _shared: &Shared) -> Result<()> {
        let results = ctx
            .results
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("results missing"))?;
        let (rows, cols) = (results.shape()[0], results.shape()[1]);
        if cols != 5 {
            ctx.final_boxes = Some(vec![]);
            return Ok(());
        }

        let sample_h = results[[0, 0]].abs();
        let sample_w = results[[0, 0]].abs();
        let likely_normalized = sample_h <= 1.01 && sample_w <= 1.01;
        let target = ctx.target;
        let mut detections = Vec::new();

        for i in 0..rows {
            let confidence = results[[i, 4]];
            if confidence < self.confidence_threshold {
                continue;
            }
            let mut x = results[[i, 0]];
            let mut y = results[[i, 1]];
            let mut w = results[[i, 2]];
            let mut h = results[[i, 3]];
            if likely_normalized {
                x *= target as f32;
                y *= target as f32;
                w *= target as f32;
                h *= target as f32;
            }
            let x1_model = x - (w / 2.0);
            let y1_model = y - (h / 2.0);
            let x2_model = x + (w / 2.0);
            let y2_model = y + (h / 2.0);

            let pad_x_f = ctx.pad_x as f32;
            let pad_y_f = ctx.pad_y as f32;
            let inv_scale = 1.0 / ctx.scale;
            let bx1 = (x1_model - pad_x_f) * inv_scale;
            let by1 = (y1_model - pad_y_f) * inv_scale;
            let bx2 = (x2_model - pad_x_f) * inv_scale;
            let by2 = (y2_model - pad_y_f) * inv_scale;
            detections.push(Bbox::new(bx1, by1, bx2, by2, confidence));
        }

        let final_boxes = non_max_suppression(detections, self.iou_threshold);
        ctx.final_boxes = Some(final_boxes);
        Ok(())
    }
}

/// Save crops to output directory
pub struct SaveCropsStage {
    pub out_dir: PathBuf,
}
impl Stage for SaveCropsStage {
    fn name(&self) -> &str {
        "SaveCrops"
    }
    fn run(&self, ctx: &mut Context, _shared: &Shared) -> Result<()> {
        let raw = ctx
            .raw_img
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("raw_img missing"))?;
        let final_boxes = ctx
            .final_boxes
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("final_boxes missing"))?;
        let mut rgb_owned = raw.to_rgb8();

        std::fs::create_dir_all(&self.out_dir)?;
        let stem = ctx
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("img");

        for (idx, bbox) in final_boxes.iter().enumerate() {
            match bbox.crop_bbox(&mut rgb_owned) {
                Ok(c) => {
                    let out_path =
                        format!("{}/{}_face_{}.jpg", self.out_dir.display(), stem, idx + 1);
                    if let Err(e) = c.save(&out_path) {
                        eprintln!("Failed to save {}: {}", out_path, e);
                    } else {
                        println!("Saved {}", out_path);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to crop bbox {}: {}", idx, e);
                }
            }
        }
        Ok(())
    }
}

/// Helper: load model (same as your load_model)
fn load_model() -> Result<
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    anyhow::Error,
> {
    let model = tract_onnx::onnx()
        .model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

/// Build standard pipeline
fn build_pipeline(out_dir: PathBuf) -> Vec<Box<dyn Stage>> {
    vec![
        Box::new(LoadImageStage),
        Box::new(PrepareInputStage),
        Box::new(InferenceStage),
        Box::new(PostprocessStage {
            confidence_threshold: 0.5,
            iou_threshold: 0.4,
        }),
        Box::new(SaveCropsStage { out_dir }),
    ]
}

/// Run pipeline for a single image path
fn run_pipeline_for_path(stages: &[Box<dyn Stage>], shared: &Shared, path: PathBuf) {
    let mut ctx = Context::new(path.clone());
    let start = Instant::now();
    for stage in stages {
        let name = stage.name();
        let t0 = Instant::now();
        if let Err(e) = stage.run(&mut ctx, shared) {
            eprintln!("Stage {} failed for {:?}: {}", name, path, e);
            return; // choose to abort on stage failure; could also continue depending on policy
        }
        println!("Stage {} done in {:?}", name, t0.elapsed());
    }
    println!("Pipeline for {:?} finished in {:?}", path, start.elapsed());
}

fn main() -> Result<(), anyhow::Error> {
    let t = Instant::now();
    let model = load_model()?;
    println!("Model loaded in {:?}", t.elapsed());
    let shared = Shared::new(model);

    let out_dir = PathBuf::from("output");
    std::fs::create_dir_all(&out_dir)?;

    // build pipeline stages once
    let stages = build_pipeline(out_dir.clone());

    // iterate image files
    let image_dir = std::fs::read_dir("images")?;
    let mut image_count = 0usize;
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
        run_pipeline_for_path(&stages, &shared, path);
    }
    println!(
        "Processed {} images. Total elapsed: {:?}",
        image_count,
        t.elapsed()
    );
    Ok(())
}
