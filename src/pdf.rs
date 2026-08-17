mod convert;
mod merge;
mod ocr;
mod slice;

pub use convert::{ConvertStage, convert};
pub use merge::{
    BundleOutcome, DocType, DocTypeEvidence, FigureCrop, FigureRenderMetrics, ProjectReport,
    collect_diagnostics, frontmatter, pair_figures, project, render_figures, render_markdown,
    validate_structure, write_bundle,
};
pub use ocr::{
    ApiErrorKind, Bbox, BlockLabel, CacheLayout, JobState, OcrError, OcrMetrics, Page, PageBlock,
    PaddleOcrClient, cache_key, parse_jsonl,
};
pub use slice::{MAX_SLICE_BYTES, PdfDocument};

// Internals exercised by pdf_test.rs (kept as one file); re-exported under
// cfg(test) so its `use super::*` keeps resolving across the module split.
#[cfg(test)]
pub(crate) use merge::{
    book_degradation_warning, bundle_outcome_summary, detect_bundle_outcome, figure_metrics_summary,
    should_report_figure_progress, validate_figure_crops,
};
#[cfg(test)]
pub(crate) use ocr::{
    JournalJob, OcrJournal, classify_error, format_duration, ocr_metrics_summary,
    read_cached_slice, run_ocr_with, submit_all_slices, write_file_atomic,
};
// Names pdf_test.rs expects to inherit via `use super::*` from the former
// single-file module.
#[cfg(test)]
use crate::config::PdfConfig;
#[cfg(test)]
use governor::{Quota, RateLimiter};
#[cfg(test)]
use lopdf::Document;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;

#[cfg(test)]
#[path = "pdf_test.rs"]
mod tests;
