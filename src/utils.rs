use std::io::{self, Write};

use rayon::prelude::*;
use crate::counter::Counter;

pub type Token = Vec<u16>;
pub const BASE_TOKENS: u16 = 256; //must be 256, since other code assumes it is 256

///turns several elements into one element
pub fn replace(s: &[u16], from: &[u16], to: u16) -> Vec<u16> {
	let mut result = Vec::with_capacity(s.len());
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

///turns one element into several elements
pub fn expand(s: &[u16], from: u16, to: &[u16]) -> Vec<u16> {
	let mut result = Vec::with_capacity(s.len());
	let mut i = 0;
	while i < s.len() {
		if s[i] == from {
			result.extend(to)
			
		} else {
			result.push(s[i]);
		}
		i += 1;
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

pub fn expand_in_library(library: &Counter<Token>, token_id: u16, expansion: &[u16]) -> Counter<Token> {
	library.map_keys(|key| expand(key, token_id, expansion))
}


pub fn fertility(library: &Counter<Token>) -> f64 {
	let total_token_lengths: usize =
		library.into_iter().map(|(key, value)| key.len() * value).sum();
	total_token_lengths as f64 / library.total() as f64
}

///find the most commonly occurring byte pair in the library
pub fn find_candidate(library : &Counter<Token>) -> Option<(Token, usize)> {
	//this is a hotpath, so we are optimizing
	//including packing the BPE pairs into a single u32 
	let pair_counts : Counter<u32> = 
		 library.par_iter().fold(
			|| Counter::new(),
			|mut counter, (t, &weight)|
			{ counter.update_weighted(
					t.windows(2).map(|a| ((a[0] as u32) << 16) | a[1] as u32)
					, weight
				);
				counter
			}
		).sum();
	//most_common() is a bit bug prone
	//the only lawful reason for most_common() to be none is if there are
	//no pairs left in the library (because every token is only 1 element long)
	assert!(pair_counts.most_common() == None || pair_counts.len() > 0);
	pair_counts.most_common().map(
	|(token, amount)| {
			let top_bits : u16 = (token >> 16 ) as u16;
			let bottom_bits : u16 = (token & 0xFFFF ) as u16;
			(vec![top_bits, bottom_bits], amount)
		}
	)
}

///count how often a token occurs inside the library
pub fn token_freq(library: &Counter<Token>, token_id: u16) -> usize {
	//TODO test that to see if par_iter actually speeds things here
	//because tokens always 2 characters long, no need for Vec 
	library.par_iter()
		.map(|(word, &count)| word.iter().filter(|&&t| t == token_id).count() * count)
		.sum()
}

///progress bar for training tokens
///that *very* roughly fills 80 columns with periods 
///as we progress
pub fn ticker<const MAX_ITEMS : u16>(i : u16) {
	if i % (1+(MAX_ITEMS / (80 - 1))) == 0 {
		print!(".");
		io::stdout().flush().unwrap();
	}
}
#[cfg(test)]
mod test {
	use super::*;
	
	#[test]
	fn test_replace() {
		let c = vec!(1,2,3,4);
		let a = replace(&c, &[2,3], 4);
		assert_eq!(a, [1,4,4]);
	}
	#[test]
	fn test_replace_no_match() {
		let c = vec!(1,2,3,4);
		let a = replace(&c, &[4,3], 4);
		assert_eq!(a, [1,2,3,4]);
		let a = replace(&c, &[1,2,3,4,5], 4);
		assert_eq!(a, [1,2,3,4]);
	}
	#[test]
	fn test_replace_double_replace() {
		let c = vec!(1,2,1,2);
		let a = replace(&c, &[1,2], 4);
		assert_eq!(a, [4,4]);
	}
	
	#[test]
	fn test_expand_multi() {
		let c = vec!(1,2,1,2);
		let a = expand(&c, 2, &[1,2]);
		assert_eq!(a, [1,1,2,1,1,2]);
	}

	#[test]
	fn test_expand_no_replace() {
		let c = vec!(1,4);
		let a = expand(&c, 2, &[1,2]);
		assert_eq!(a, [1,4]);
	}

	#[test]
	fn test_expand_3length() {
		let c = vec!(1,4,3);
		let a = expand(&c, 4, &[1,2,3]);
		assert_eq!(a, [1,1,2,3,3]);
	}

	#[test]
	fn test_find_candidate() {
		let mut c = Counter::new();
		c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
		let a = find_candidate(&c).unwrap();
		assert_eq!(a.0, vec!(1,1) );
		assert_eq!(a.1, 3 );
	}
}
// vim: ts=2 sw=2
