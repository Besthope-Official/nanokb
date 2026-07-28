use crate::Section;

pub enum Filter {
    /// Drop sections whose title matches reference/bibliography patterns
    /// (e.g. "References", "Bibliography", "参考文献").
    DropReference,
}

pub fn apply_filters(sections: Vec<Section>, filters: &[Filter]) -> Vec<Section> {
    let mut sections = sections;
    for filter in filters {
        sections = match filter {
            Filter::DropReference => drop_reference_sections(sections),
        };
    }
    sections
}

fn drop_reference_sections(sections: Vec<Section>) -> Vec<Section> {
    sections
        .into_iter()
        .filter(|s| !is_reference_section(s))
        .collect()
}

fn is_reference_section(section: &Section) -> bool {
    let title = section.title.trim().to_lowercase();

    matches!(
        title.as_str(),
        "references"
            | "bibliography"
            | "works cited"
            | "references cited"
            | "literature cited"
            | "参考"
            | "参考文献"
            | "参考资料"
    )
}

#[cfg(test)]
#[path = "filter_test.rs"]
mod tests;
