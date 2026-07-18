use rayon::prelude::*;
use std::collections::HashSet;

use crate::counter::Counter;
use crate::utils::{replace_in_library, Token, BASE_TOKENS, find_candidate};

pub struct ScaffoldBpe {
	all_vocab: Vec<Token>,
	is_scaffold: HashSet<u16>,
	normal_count: usize,
}

impl ScaffoldBpe {
	pub fn new() -> Self {
		ScaffoldBpe {
			all_vocab: Vec::new(),
			is_scaffold: HashSet::new(),
			normal_count: 0,
		}
	}

	pub fn train<F: Fn(u16)>(&mut self, library: &Counter<Token>, 
				target_normal_tokens: u16, progress_fn : F) {
		let mut library = library.clone();

		let mut i = 0u16;
		loop {
			if self.normal_count >= target_normal_tokens as usize {
				break;
			}

			let Some((pair, _pair_freq)) = find_candidate(&library) else {
				println!("no compression possible at {} iterations", i);
				break;
			};

			let mut x3 = Vec::new();
			x3.push(pair[0]);
			x3.push(pair[1]);

			if let Some(existing_idx) = self.all_vocab.iter().position(|v| *v == x3) {
				let existing_id = BASE_TOKENS + existing_idx as u16;
				if self.is_scaffold.contains(&existing_id) {
					self.is_scaffold.remove(&existing_id);
					self.normal_count += 1;
					library = replace_in_library(&library, &x3, existing_id);
					i += 1;
					continue;
				}
			}

			let token_id = BASE_TOKENS + self.all_vocab.len() as u16;

			self.all_vocab.push(x3.clone());
			self.normal_count += 1;

			library = replace_in_library(&library, &x3, token_id);

			let next_pair_freq = find_candidate(&library)
				.map(|(_, f)| f)
				.unwrap_or(0);

			let remaining_a = token_freq(&library, pair[0]);
			let remaining_b = token_freq(&library, pair[1]);

			if pair[0] >= BASE_TOKENS {
				if remaining_a < next_pair_freq && !self.is_scaffold.contains(&pair[0]) {
					self.is_scaffold.insert(pair[0]);
					self.normal_count -= 1;
				}
			}
			if pair[1] != pair[0] && pair[1] >= BASE_TOKENS {
				if remaining_b < next_pair_freq && !self.is_scaffold.contains(&pair[1]) {
					self.is_scaffold.insert(pair[1]);
					self.normal_count -= 1;
				}
			}

			i += 1;
			progress_fn(i);
		}
		println!();
		eprintln!("Vocab: {} normal, {} scaffold, {} total",
			self.normal_count, self.is_scaffold.len(), self.all_vocab.len());
	}

	pub fn tokenize(&self, word: &[u16]) -> Vec<u16> {
		let mut current: Vec<u16> = word.to_vec();

		for (i, vocab_entry) in self.all_vocab.iter().enumerate() {
			let token_id = BASE_TOKENS + i as u16;
			let mut j = 0;
			while j + vocab_entry.len() <= current.len() {
				if current[j..j + vocab_entry.len()] == *vocab_entry {
					current[j] = token_id;
					current.drain(j + 1..j + vocab_entry.len());
				} else {
					j += 1;
				}
			}
		}

		let mut result = Vec::new();
		for &token in &current {
			if token < BASE_TOKENS {
				result.push(token);
			} else if self.is_scaffold.contains(&token) {
				result.extend_from_slice(&self.demolish(token));
			} else {
				result.push(token);
			}
		}
		result
	}

	fn demolish(&self, token_id: u16) -> Token {
		if token_id < BASE_TOKENS {
			return vec![token_id];
		}
		if !self.is_scaffold.contains(&token_id) {
			return vec![token_id];
		}
		let idx = (token_id - BASE_TOKENS) as usize;
		let components = &self.all_vocab[idx];
		let mut result = Token::new();
		for &c in components {
			result.extend_from_slice(&self.demolish(c));
		}
		result
	}
}

fn token_freq(library: &Counter<Token>, token_id: u16) -> usize {
	library.par_iter()
		.map(|(word, &count)| word.iter().filter(|&&t| t == token_id).count() * count)
		.sum()
}
// vim: ts=2 sw=2
