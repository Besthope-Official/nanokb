use anyhow::Result;
use nanokb::parse_markdown;
use std::fs;

fn main() -> Result<()> {
    let content = fs::read_to_string("examples/example2.md")?;
    let sections = parse_markdown(&content);

    for (idx, section) in sections.iter().enumerate() {
        let path_text = section.path(&sections).join(">");
        println!(
            "-----\nsection_idx: {}\nstart_line_num: {}\nend_line_num: {}\ndepth: {}\ntitle: {}\nparagraphs: {}\npath: {}\n-----",
            idx,
            section.start_line_num,
            section.end_line_num,
            section.depth,
            section.title,
            section.paragraphs,
            path_text
        )
    }
    Ok(())
}
