use anyhow::Result;
use std::fs;

#[derive(Clone)]
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
    let mut section_list: Vec<Section> = [Section {
        depth: 0,
        title_content: String::new(),
        start_line_num: 0,
        end_line_num: 0,
        paragraphs: String::new(),
    }]
    .to_vec();
    let lines_list: Vec<&str> = contents.lines().collect();

    for (curr_section_line_num, line) in contents.lines().enumerate() {
        if line.contains("```") && line.starts_with("```") {
            in_fence = !in_fence;
        }
        let trimmed_heading_line = line.trim_start();
        if !trimmed_heading_line.is_empty() {
            let depth = trimmed_heading_line
                .chars()
                .take_while(|&c| c == '#')
                .count();
            let (_, rest) = trimmed_heading_line.split_at(depth);
            if !in_fence && (1..7).contains(&depth) && rest.starts_with(" ") {
                if let Some(last_section) = section_list.last_mut() {
                    last_section.end_line_num = curr_section_line_num;
                    last_section.paragraphs = lines_list
                        [last_section.start_line_num + 1..curr_section_line_num]
                        .join("\n");
                }
                section_list.push(Section {
                    depth,
                    title_content: rest.trim_start().to_string(),
                    start_line_num: curr_section_line_num,
                    end_line_num: curr_section_line_num,
                    paragraphs: String::new(),
                });
            }
        }
    }
    if let Some(final_section) = section_list.last_mut() {
        final_section.end_line_num = lines_list.len();
        final_section.paragraphs =
            lines_list[final_section.start_line_num + 1..lines_list.len()].join("\n");
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
