use anyhow::Result;
use nanokb::parse_markdown;

fn main() -> Result<()> {
    let sections = parse_markdown("examples/example.md")?;

    for (idx, section) in sections.iter().enumerate() {
        let path_text = section.path(&sections).join(">");
        println!(
            "-----\nsection_idx: {}\nstart_line_num: {}\nend_line_num: {}\ndepth: {}\ncontent: {}\nparagraphs: {}\npath: {}\n-----",
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
