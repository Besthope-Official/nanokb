use anyhow::Result;
use nanokb::parse_markdown;
use std::fs;

fn main() -> Result<()> {
    let content = fs::read_to_string("examples/example.md")?;
    let sections = parse_markdown(&content);

    for (idx, section) in sections.iter().enumerate() {
        let path_text = section.path(&sections).join(">");
        println!(
            "-----\nsection_idx: {}\nline {}-{}\nheading_level: {}\ntitle: {}\ncontent: {}\npath: {}\n-----",
            idx,
            section.source_span.start + 1,
            section.source_span.end + 1,
            section.heading_level,
            if !section.title.is_empty() { section.title.as_str() } else { "<NO TITLE>" },
            if !section.content.is_empty() { section.content.as_str() } else { "<NO CONTENT>" },
            path_text
        )
    }
    Ok(())
}
