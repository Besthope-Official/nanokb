pub mod parser;
pub use parser::{parse_heading, parse_markdown, Section};

#[cfg(test)]
mod parser_test;
