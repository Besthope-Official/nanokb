use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    Eq,
    NotEq,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    pub key: String,
    pub op: FilterOp,
    pub value: String,
}

impl FromStr for Filter {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let eq = raw.find('=').ok_or_else(|| {
            format!("invalid filter '{raw}': expected key=value or key!=value")
        })?;
        let (op, key_end) = if eq > 0 && raw.as_bytes()[eq - 1] == b'!' {
            (FilterOp::NotEq, eq - 1)
        } else {
            (FilterOp::Eq, eq)
        };
        let key = &raw[..key_end];
        if key.is_empty() {
            return Err(format!("invalid filter '{raw}': key is empty"));
        }
        Ok(Self {
            key: key.to_string(),
            op,
            value: raw[eq + 1..].to_string(),
        })
    }
}

impl Filter {
    pub fn predicate_sql(&self, key_param: usize, value_param: usize) -> String {
        let fragment = format!(
            "CASE WHEN jsonb_typeof(document.frontmatter -> ${key_param}) = 'array' \
             THEN document.frontmatter -> ${key_param} ? ${value_param} \
             ELSE document.frontmatter ->> ${key_param} = ${value_param} END"
        );
        match self.op {
            FilterOp::Eq => format!("({fragment}) IS TRUE"),
            FilterOp::NotEq => format!("({fragment}) IS NOT TRUE"),
        }
    }
}

pub fn where_clause(filters: &[Filter], start: usize) -> Option<String> {
    if filters.is_empty() {
        return None;
    }
    let fragments: Vec<String> = filters
        .iter()
        .enumerate()
        .map(|(i, filter)| filter.predicate_sql(start + 2 * i, start + 2 * i + 1))
        .collect();
    Some(format!("WHERE {}", fragments.join(" AND ")))
}

#[cfg(test)]
#[path = "filter_test.rs"]
mod tests;
