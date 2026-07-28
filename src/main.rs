use anyhow::Result;
use std::fs;

#[derive(Default)]
struct Section {
    depth: usize,
    title: String,
    start_line_num: usize,
    end_line_num: usize,
    paragraphs: String,
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let depth = trimmed.chars().take_while(|&c| c == '#').count();
    if !(1..7).contains(&depth) {
        return None;
    }
    let rest = &trimmed[depth..];
    if !rest.starts_with(' ') {
        return None;
    }
    Some((depth, rest.trim_start()))
}

fn main() -> Result<()> {
    let path = "examples/example.md";
    let contents = fs::read_to_string(path)?;

    let mut in_fence = false;
    let mut section_list: Vec<Section> = vec![Section::default()];

    for (i, line) in contents.lines().enumerate() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        if !in_fence && let Some((depth, title)) = parse_heading(line) {
            section_list.push(Section {
                depth,
                title: title.to_string(),
                start_line_num: i,
                end_line_num: i,
                paragraphs: String::new(),
            });
        } else {
            if let Some(curr) = section_list.last_mut() {
                curr.end_line_num = i;
                curr.paragraphs += if curr.paragraphs.is_empty() { "" } else { "\n" };
                curr.paragraphs += line;
            }
        }
    }
    if let Some(curr) = section_list.last() && curr.paragraphs.is_empty() {
        section_list.pop();
    }

    for section in section_list {
        println!(
            "start_line_num: {}, end_line_num: {}, depth: {}, content: {}, paragraphs: {}",
            section.start_line_num,
            section.end_line_num,
            section.depth,
            section.title,
            section.paragraphs
        )
    }
    Ok(())
}
