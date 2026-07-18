use rayon::prelude::*;
use std::collections::HashSet;

use crate::counter::Counter;

type Token = Vec<u16>;

#[derive(Clone, Debug)]
pub enum Event {
    Merge(Token, Token),
    Remove(Token),
}

pub struct PickyBpe {
    pub vocab: Vec<Token>,
    pub events: Vec<Event>,
    pub threshold: f64,
    token_expansion: Vec<Token>,
}

impl PickyBpe {
    pub fn new(threshold: f64) -> Self {
        let mut token_expansion = Vec::new();
        for i in 0..256u16 {
            token_expansion.push(vec![i]);
        }
        PickyBpe {
            vocab: Vec::new(),
            events: Vec::new(),
            threshold,
            token_expansion,
        }
    }

    pub fn train(&mut self, library: &Counter<Token>, new_token_count: u16) {
        let mut library = library.clone();
        let base_tokens = 256u16;
        let mut removed_ids: HashSet<u16> = HashSet::new();
        let mut effective_count: u16 = 0;

        loop {
            if effective_count >= new_token_count {
                break;
            }
            let mut newly_removed: Vec<u16> = Vec::new();

            let Some((pair, pair_freq)) = find_candidate(&library, &removed_ids) else {
                println!("no compression possible at {} effective tokens", effective_count);
                break;
            };

            let x1 = vec![pair[0]];
            let x2 = vec![pair[1]];
            let mut x3 = Vec::new();
            x3.extend_from_slice(&x1);
            x3.extend_from_slice(&x2);

            let x1_freq: usize = library.par_iter()
                .map(|(word, &count)| word.iter().filter(|&&t| t == pair[0]).count() * count)
                .sum();
            let x2_freq: usize = library.par_iter()
                .map(|(word, &count)| word.iter().filter(|&&t| t == pair[1]).count() * count)
                .sum();

            self.events.push(Event::Merge(x1.clone(), x2.clone()));

            if x1_freq > 0 {
                let ios_x1 = pair_freq as f64 / x1_freq as f64;
                if ios_x1 >= self.threshold && pair[0] >= base_tokens {
                    self.events.push(Event::Remove(x1.clone()));
                    removed_ids.insert(pair[0]);
                    newly_removed.push(pair[0]);
                }
            }

            if x2 != x1 && x2_freq > 0 {
                let ios_x2 = pair_freq as f64 / x2_freq as f64;
                if ios_x2 >= self.threshold && pair[1] >= base_tokens {
                    self.events.push(Event::Remove(x2.clone()));
                    removed_ids.insert(pair[1]);
                    newly_removed.push(pair[1]);
                }
            }

            let exp_x1 = self.token_expansion[pair[0] as usize].clone();
            let mut exp_x3 = exp_x1;
            exp_x3.extend_from_slice(&self.token_expansion[pair[1] as usize]);
            self.token_expansion.push(exp_x3);

            let new_token_id = self.vocab.len() as u16 + base_tokens;
            library = replace_in_library(&library, &x3, new_token_id);

            for &removed_id in &newly_removed {
                let expansion = self.token_expansion[removed_id as usize].clone();
                library = decompose_in_library(&library, removed_id, &expansion);
            }

            self.vocab.push(x3);
            effective_count += 1;
            effective_count -= newly_removed.len() as u16;

            if effective_count % 100 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
        }
        println!();
        eprintln!("Events: {} merges, {} removals, {} effective tokens",
            self.events.iter().filter(|e| matches!(e, Event::Merge(_, _))).count(),
            self.events.iter().filter(|e| matches!(e, Event::Remove(_))).count(),
            effective_count);
    }

    pub fn tokenize(&self, word: &[u16]) -> Vec<u16> {
        let mut current: Vec<u16> = word.to_vec();

        for event in &self.events {
            match event {
                Event::Merge(x1, x2) => {
                    let mut merged_token = Vec::new();
                    merged_token.extend_from_slice(x1);
                    merged_token.extend_from_slice(x2);
                    let token_id = 256 + self.vocab.iter().position(|v| *v == merged_token).unwrap_or(0) as u16;
                    let mut j = 0;
                    while j + x1.len() + x2.len() <= current.len() {
                        if current[j..j + x1.len()] == *x1 && current[j + x1.len()..j + x1.len() + x2.len()] == *x2 {
                            current[j] = token_id;
                            current.drain(j + 1..j + 1 + x2.len());
                        } else {
                            j += 1;
                        }
                    }
                }
                Event::Remove(token) => {
                    let token_id = token[0];
                    if token_id < 256 { continue; }
                    let expansion = self.token_expansion[token_id as usize].clone();
                    let mut j = 0;
                    while j < current.len() {
                        if current[j] == token_id {
                            let exp_len = expansion.len();
                            current[j] = expansion[0];
                            for k in 1..exp_len {
                                current.insert(j + k, expansion[k]);
                            }
                            j += exp_len;
                        } else {
                            j += 1;
                        }
                    }
                }
            }
        }

        current
    }
}

fn find_candidate(library: &Counter<Token>, removed_ids: &HashSet<u16>) -> Option<(Token, usize)> {
    let pair_counts: Counter<u32> =
        library.par_iter().fold(
            || Counter::new(),
            |mut counter, (t, &weight)| {
                if t.len() < 2 {
                    return counter;
                }
                counter.update_weighted(
                    t.windows(2).filter(|w| {
                        !removed_ids.contains(&w[0]) && !removed_ids.contains(&w[1])
                    }).map(|a| ((a[0] as u32) << 16) | a[1] as u32),
                    weight,
                );
                counter
            }
        ).sum();

    pair_counts.most_common().map(|(token, amount)| {
        let top_bits = (token >> 16) as u16;
        let bottom_bits = (token & 0xFFFF) as u16;
        (vec![top_bits, bottom_bits], amount)
    })
}

fn replace(s: &[u16], from: &[u16], to: u16) -> Vec<u16> {
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

fn replace_in_library(library: &Counter<Token>, from: &[u16], to: u16) -> Counter<Token> {
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

fn decompose_in_library(library: &Counter<Token>, token_id: u16, expansion: &[u16]) -> Counter<Token> {
    let mut new_library = Counter::with_capacity(library.len());
    for (key, count) in library {
        let new_key: Vec<u16> = key.iter().flat_map(|&t| {
            if t == token_id {
                expansion.to_vec()
            } else {
                vec![t]
            }
        }).collect();
        new_library[&new_key] = *count;
    }
    if let Some(cm) = &library.current_max {
        let new_cm: Vec<u16> = cm.0.iter().flat_map(|&t| {
            if t == token_id {
                expansion.to_vec()
            } else {
                vec![t]
            }
        }).collect();
        new_library.current_max = Some((new_cm, cm.1));
    }
    new_library
}

pub fn fertility(library: &Counter<Token>) -> f64 {
    let total_token_lengths: usize =
        library.into_iter().map(|(key, value)| key.len() * value).sum();
    total_token_lengths as f64 / library.total() as f64
}
