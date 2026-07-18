use crate::counter::Counter;

pub type Token = Vec<u16>;
pub const BASE_TOKENS: u16 = 256;

pub fn replace(s: &[u16], from: &[u16], to: u16) -> Vec<u16> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if i + from.len() <= s.len() && s[i..i + from.len()] == *from {
            result.push(to);
            i += from.len();
        } else {
            result.push(s[i]);
            i += 1;
        }
    }
    result
}

pub fn replace_in_library(library: &Counter<Token>, from: &[u16], to: u16) -> Counter<Token> {
    let mut new_library = Counter::with_capacity(library.len());
    for (key, count) in library {
        let new_key = replace(key, from, to);
        new_library[&new_key] = *count;
    }
    if let Some(cm) = &library.current_max {
        new_library.current_max = Some((replace(&cm.0, from, to), cm.1));
    }
    new_library
}

pub fn fertility(library: &Counter<Token>) -> f64 {
    let total_token_lengths: usize =
        library.into_iter().map(|(key, value)| key.len() * value).sum();
    total_token_lengths as f64 / library.total() as f64
}
