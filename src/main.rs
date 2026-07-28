use anyhow::Result;
use std::{fs, usize::MAX, vec};

struct Section {
    depth: usize,
    title_content: String,
    start_line_num: usize,
    end_line_num: usize,
    paragraphs: String,
}

fn main() -> Result<()> {
    let path = "examples/example.md";
    let contents = fs::read_to_string(path)?;

    let mut in_fence = false;
    let mut prev_section_line_num: usize = MAX;

    let mut section_list: Vec<Section> = vec![];
    let mut heading_line_buffer: &str = "";

    let lines_list: Vec<&str> = contents.lines().collect();
    // println!("lines length: {}", lines_list.len());

    for (curr_section_line_num, line) in contents.lines().enumerate() {
        // println!("line: {}", curr_section_line_num + 1);
        if line.contains("```") {
            in_fence = !in_fence;
        }
        // last heading
        if curr_section_line_num + 1 == lines_list.len() {
            if prev_section_line_num != MAX {
                if let Some(buffer_line_heading_depth) = heading_line_buffer.rfind(' ') {
                    let (_, content) = heading_line_buffer.split_at(buffer_line_heading_depth + 1);
                    let paragraphs =
                        lines_list[prev_section_line_num + 1..curr_section_line_num].join("\n");
                    section_list.push(Section {
                        depth: buffer_line_heading_depth,
                        title_content: content.to_string(),
                        start_line_num: prev_section_line_num + 1,
                        end_line_num: curr_section_line_num,
                        paragraphs,
                    });
                }
            }
        }
        if line.starts_with('#')
            && !in_fence
            && let Some(heading_depth) = line.rfind(' ')
            && heading_depth < 7
        {
            if prev_section_line_num != MAX {
                if let Some(buffer_line_heading_depth) = heading_line_buffer.find(' ') {
                    let (_, content) = heading_line_buffer.split_at(buffer_line_heading_depth + 1);
                    let paragraphs =
                        lines_list[prev_section_line_num + 1..curr_section_line_num].join("\n");
                    section_list.push(Section {
                        depth: buffer_line_heading_depth,
                        title_content: content.to_string(),
                        start_line_num: prev_section_line_num + 1,
                        end_line_num: curr_section_line_num,
                        paragraphs,
                    });
                }
            }
            heading_line_buffer = line;
            prev_section_line_num = curr_section_line_num;
        }
    }

    for heading in section_list {
        println!(
            "start_line_num: {}, end_line_num: {}, depth: {}, content: {}, paragraphs: {}",
            heading.start_line_num,
            heading.end_line_num,
            heading.depth,
            heading.title_content,
            heading.paragraphs
        )
    }
    // println!("{}", contents);
    Ok(())
}
