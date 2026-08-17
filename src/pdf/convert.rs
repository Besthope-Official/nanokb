use super::merge::{DocType, run_merge_with};
use super::ocr::{
    CacheLayout, DAILY_QUOTA_PAGES, OcrMetrics, StatusLine, ocr_metrics_summary, run_ocr_with,
};
use super::slice::{MAX_SLICE_BYTES, PdfDocument};
use crate::config::PdfConfig;
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::Path;

#[derive(clap::ValueEnum, Clone, Copy, PartialEq, Eq)]
pub enum ConvertStage {
    /// Slice the PDF into per-slice files in the cache (no API calls).
    Slice,
    /// Slice + OCR every slice via PaddleOCR, cache raw results.
    Ocr,
    /// Project cached OCR results into the md bundle (offline).
    Merge,
}

/// Top-level `nanokb convert` driver: slice → ocr → merge, resumable via the
/// cache. `--dry-run` prints the plan and quota estimate; `--stage` stops
/// after a stage; `out` is required unless the run stops before merge.
pub async fn convert(
    cfg: &PdfConfig,
    file: &Path,
    out: Option<&Path>,
    stage: Option<ConvertStage>,
    dry_run: bool,
    slice_pages: usize,
    doc_type: DocType,
) -> Result<()> {
    if dry_run {
        let pdf_doc = PdfDocument::open(file)?;
        let plan = pdf_doc.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
        let quota_pct = pdf_doc.page_count() as f64 * 100.0 / DAILY_QUOTA_PAGES as f64;
        eprintln!(
            "{} · {quota_pct:.0}% of daily quota",
            plan_summary(file, pdf_doc.page_count(), plan.len(), slice_pages)
        );
        return Ok(());
    }
    match (stage, out) {
        (Some(ConvertStage::Slice), _) => slice_to_cache(file, slice_pages, cfg).await,
        (Some(ConvertStage::Ocr), _) => {
            let metrics = run_ocr(cfg, file, slice_pages).await?;
            if let Some(summary) = ocr_metrics_summary(metrics) {
                eprintln!("{summary}");
            }
            Ok(())
        }
        (Some(ConvertStage::Merge), Some(out)) => run_merge(cfg, file, out, slice_pages, doc_type),
        (None, Some(out)) => {
            // Full pipeline: compute plan + cache layout once for both stages.
            let pdf_doc = PdfDocument::open(file)?;
            let plan = pdf_doc.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
            let layout = CacheLayout::for_pdf(file, slice_pages, &cfg.api_base, &cfg.model)?;
            eprintln!(
                "{}",
                plan_summary(file, pdf_doc.page_count(), plan.len(), slice_pages)
            );
            let metrics = run_ocr_with(cfg, &pdf_doc, &layout, &plan, &stem_label(file)?).await?;
            if let Some(summary) = ocr_metrics_summary(metrics) {
                eprintln!("{summary}");
            }
            run_merge_with(&layout, &plan, file, out, doc_type)
        }
        (Some(ConvertStage::Merge) | None, None) => {
            bail!("--out <dir> is required unless --stage stops before merge")
        }
    }
}

/// One-line plan summary, shared by dry-run and every stage entry point.
fn plan_summary(pdf_path: &Path, page_count: u32, slices: usize, slice_pages: usize) -> String {
    format!(
        "{}: {page_count} pages · {slices} slices (up to {slice_pages} per slice)",
        pdf_path.display()
    )
}

async fn slice_to_cache(pdf_path: &Path, slice_pages: usize, cfg: &PdfConfig) -> Result<()> {
    let pdf = PdfDocument::open(pdf_path)?;
    let plan = pdf.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, &cfg.api_base, &cfg.model)?;
    eprintln!("{}", plan_summary(pdf_path, pdf.page_count(), plan.len(), slice_pages));
    fs::create_dir_all(layout.slices_dir())
        .with_context(|| format!("failed to create {}", layout.slices_dir().display()))?;
    let label = pdf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let status = StatusLine::new(label);
    let mut written = 0usize;
    for (index, &(start, end)) in plan.iter().enumerate() {
        let dest = layout.slice_path(index);
        if dest.exists() {
            continue;
        }
        pdf.write_slice(start, end, &dest)?;
        written += 1;
        status.update(&format!("slices {written}/{}", plan.len()));
    }
    status.clear();
    Ok(())
}

async fn run_ocr(cfg: &PdfConfig, pdf_path: &Path, slice_pages: usize) -> Result<OcrMetrics> {
    let pdf = PdfDocument::open(pdf_path)?;
    let plan = pdf.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, &cfg.api_base, &cfg.model)?;
    eprintln!("{}", plan_summary(pdf_path, pdf.page_count(), plan.len(), slice_pages));
    run_ocr_with(cfg, &pdf, &layout, &plan, &stem_label(pdf_path)?).await
}

fn stem_label(pdf_path: &Path) -> Result<String> {
    pdf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .context("PDF path has no usable stem")
}

/// Merge stage: project cached raw OCR results into an md bundle (offline).
fn run_merge(
    cfg: &PdfConfig,
    pdf_path: &Path,
    out: &Path,
    slice_pages: usize,
    doc_type: DocType,
) -> Result<()> {
    let layout = CacheLayout::for_pdf(pdf_path, slice_pages, &cfg.api_base, &cfg.model)?;
    let pdf_doc = PdfDocument::open(pdf_path)?;
    let plan = pdf_doc.plan_slices(slice_pages, MAX_SLICE_BYTES)?;
    run_merge_with(&layout, &plan, pdf_path, out, doc_type)
}
