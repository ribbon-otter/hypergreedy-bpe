#![allow(unused)]

use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use rustc_hash::FxHashMap;

mod counter;
use counter::Counter;

use rayon::prelude::*; 
use std::env;

type Token = Vec<u16>;

///extention_population * EXT_RATIO > candidate_population 
///triggers using the extention
static EXT_RATIO : usize = 2;

///beyond the 256 tokens that already exist, how many should we add?
static NEW_TOKEN_COUNT : u16 = 1000;
static BASE_TOKENS : u16 = 256; //must be 256 by other assumptions in code
//token numbers from 0 to 255 (inclusive) represent the raw bytes
//while tokens greater than that represent compressions

fn main() -> io::Result<()> {
	let file_argument : String = env::args().nth(1).unwrap_or(String::from("./AliceInWonderland.txt"));

	let library = gen_word_counts(file_argument, sometimes_logger);
	println!();
	println!("word counts generated. distinct word count: {}", library.len());
	let (vocab, compressed_lib) = bpe_hypergreedy(library.clone(), ticker);
	let opt_fertility =  fertility(&compressed_lib);
	println!();
	println!("hypergreedy bpe : fertility {}", fertility(&compressed_lib));
	println!("hypergreedy bpe: {:?}", vocab.iter().take(10).map(to_string).collect::<Vec<_>>());
	let (vocab, compressed_lib) = bpe(library.clone(), ticker);
	let old_fertility =  fertility(&compressed_lib);
	println!();
	println!("bpe : fertility {}", old_fertility);
	println!("bpe: {:?}", vocab.iter().take(10).map(to_string).collect::<Vec<_>>());
	println!();
	println!("improvement ratio: {}", opt_fertility / old_fertility);
	Ok(())
}

fn ticker(i : u16) {
	if i % (NEW_TOKEN_COUNT / (80 - 1)) == 0 {
		print!(".");
		io::stdout().flush().unwrap();
	}
}

fn sometimes_logger(i : usize) {
	if i % 100_000 == 0 {
		print!("\rreading line: {} ", i);
		io::stdout().flush().unwrap();
	}
}

fn bpe<F : Fn(u16)>(mut library : Counter<Token>, progress_fn : F) -> (Vec<Token>, Counter<Token>){
	//vocab[i] is the expantion of token number (i - base_tokens)
	let mut vocab : Vec<Token> = Vec::with_capacity(NEW_TOKEN_COUNT.into());
	for i in 0..NEW_TOKEN_COUNT {
		let new_token = find_candidate(&library).0;
		library = replace_in_library(&library, &new_token, i + BASE_TOKENS);
		vocab.push(new_token);
		progress_fn(i);
	}
	(vocab, library)
}

fn bpe_hypergreedy<F : Fn(u16)>(mut library : Counter<Token>, progress_fn : F) -> (Vec<Token>, Counter<Token>) {
	//vocab[i] is the expantion of token number (i - base_tokens)
	let mut vocab : Vec<Token> = Vec::with_capacity(NEW_TOKEN_COUNT.into());
	for i in 0..NEW_TOKEN_COUNT {
		let new_token = find_best_token(&library).0;
		library = replace_in_library(&library, &new_token, i + BASE_TOKENS);
		vocab.push(new_token);
		progress_fn(i);
	}
	(vocab, library)
}

//replaces a string with a single value
fn replace(s : &[u16], from: &[u16], to : u16) -> Vec<u16> {
	assert!(from.len() > 0);
	let mut result : Vec<u16> = Vec::new();
	let mut i : usize = 0;
	while i < s.len() {
		if i+from.len()-1 < s.len() && s[i..i+from.len()] == *from {
			result.push(to);
			i += from.len()
		} else {
			result.push(s[i]);
			i+=1;
		}
	}
	result
}

fn replace_in_library(library : &Counter<Token>, from : &[u16], to : u16) -> Counter<Token> {
	let mut new_library : Counter<Token> = Counter::with_capacity(library.len());
	for (key, count) in library {
		let new_key = replace(key, from, to);
		new_library[&new_key] = *count;
	}
	if let Some(cm) = &library.current_max {
		new_library.current_max = Some((replace(&cm.0, from, to), cm.1));
		//replacements don't change maximums (but do change keys)
	}
	new_library
}

fn find_best_token(library : &Counter<Token>) -> (Token, usize) {
	let mut can = find_candidate(&library);
	loop {
		
		let maybe_ext = find_best_extention(&library, &can.0);
		//if we find at least one valid extention
		if let Some(ext) = maybe_ext {
			//if the extention occurs often enough
			if  ext.1 * EXT_RATIO > can.1 {
			//the candidate is the true extention
			can = ext
			} else {
				break can
			}
		} else {
			//otherwise we found the best candidate
			break can
		}
	}
}

fn find_candidate(library : &Counter<Token>) -> (Token, usize) {
	let pair_counts : Counter<Token> = 
 		library.into_iter().map(|(t, &weight)|
			t.windows(2).map(|a| a.to_vec()).collect::<Counter<Token>>() * weight
		).sum();
	pair_counts.most_common().unwrap().into()
}

fn find_best_extention(library : &Counter<Token>, candidate : &Token) -> Option<(Token, usize)> {
	let extention_counts : Counter<Token> = 
 		library.into_iter().map(|(t, &weight)|
			t.windows(candidate.len()+1)
			 .filter(
						|win| win[0..win.len()-1] == *candidate //try extending backwards
							|| win[1..win.len()] == *candidate //try extending forwards
				).map(|a| a.to_vec()).collect::<Counter<Token>>() * weight
		).sum();
	extention_counts.most_common().into()
}

fn echo<T> (a : T, prefix : &str) -> T 
	where T : std::fmt::Debug {
	println!("{:?}: {:?}", prefix, a);
	a
}

fn gen_word_counts<P, F : Fn(usize)>(filename : P, progress_fn : F) -> Counter<Token>
where P: AsRef<Path>{
	use unicode_segmentation::UnicodeSegmentation;
	let lines = read_lines(filename).unwrap();
	let word_counts : Counter<Token> =
		lines.map_while(Result::ok).enumerate().map(|(i,x)| {
			progress_fn(i);
			//unicode_words effectively strips all the punctuation and whitespace 
			// from the dataset and I expect it to behave preversely on chinese and japanese
			x.unicode_words().map( 
				//turn words into Vec<u16>s
				|s| s.as_bytes().into_iter().map(|&b| b as u16).collect::<Token>()
			).collect::<Counter<Token>>()
		}).sum();
	return word_counts;
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>> 
where P: AsRef<Path>, {
	let file = File::open(filename)?;
	Ok(io::BufReader::new(file).lines())
}

//encode a text by a vocabulary
fn decode(text : Vec<u16>, vocab : &Vec<Token>) -> Vec<u16> {
	text.iter().flat_map(|&c| if c < BASE_TOKENS {
		vec!(c)
	} else {
		decode(vocab[(c - BASE_TOKENS) as usize].clone(), vocab)
	}).collect()
}

fn find_prefix<'a>(map : &FxHashMap<&[u16], u16>, text: &'a Vec<u16>, start_idx: usize)
		-> &'a [u16] {
	let mut i : usize = 0;
	while i + start_idx < text.len()
		&& map.contains_key(&text[start_idx .. i + start_idx]) {
		
		i += 1;
	}
	&text[start_idx .. i + start_idx]
}

fn encode(text : &Vec<u16>, vocab : &Vec<Token>) -> Vec<u16> {
	let map : FxHashMap<&[u16], u16> = vocab.iter().enumerate()
		.map(|(idx, t)| (t.as_slice(),idx as u16)).collect::<FxHashMap<_, u16>>();
	let mut result : Vec<u16> = Vec::new();
	let mut i = 0;
	while i < text.len() {
		if text[i] < BASE_TOKENS {
			result.push(text[i]);
			i += 1;
		} else {
			let prefix = find_prefix(&map, &text, i);
			result.push(map[prefix]);
			i += prefix.len();
		}
	};
	result
}

///a simple token to string, displays ? for any meta tokens
fn to_string(t : &Token) -> String {
	let x = 
	t.iter().map(|&u| {
		if u < 256 { u as u8 } else {'?' as u8}
	}).collect::<Vec<u8>>();
	
	String::from_utf8_lossy(&x).to_string()
}

fn fertility(library : &Counter<Token>) -> f64 {
	//take a weighted average of the token lengths
	let total_token_lengths : usize =
		library.into_iter().map(|(key, value)| key.len() * value).sum();
	total_token_lengths as f64 / library.total() as f64
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
	fn test_find_candidate() {
		let mut c = Counter::new();
		c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
		let a = find_candidate(&c);
		assert_eq!(a.0, vec!(1,1) );
		assert_eq!(a.1, 3 );
	}

	#[test]
	fn test_find_best_extention_right() {
		let mut c = Counter::new();
		c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
		let a = find_best_extention(&c, &vec!(1,1));
		let aa = a.unwrap();
		assert_eq!(aa.0, vec!(1,1,2) );
		assert_eq!(aa.1, 2 );
	}

	#[test]
	fn test_find_best_extention_left() {
		let mut c = Counter::new();
		c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
		let a = find_best_extention(&c, &vec!(1,2));
		let aa = a.unwrap();
		assert_eq!(aa.0, vec!(1,1,2) );
		assert_eq!(aa.1, 2 );
	}

	#[test]
	fn test_find_best_extention_empty() {
		let mut c = Counter::new();
		c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
		let a = find_best_extention(&c, &vec!(1,1,2));
		std::assert_matches!(a, None);
	}

	#[test]
	fn test_find_best_token() {
		let mut c = Counter::new();
		c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
		let a = find_best_token(&c);
		assert_eq!(a.0, vec!(1,1,2) );
		assert_eq!(a.1, 2 );
	}

	#[test]
	fn test_replace_in_library() {
		let mut c = Counter::new();
		c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
		let a = replace_in_library(&c, &[1,2], 3);
		let mut b = Counter::new();
		b.update(vec!(vec!(1,3), vec!(1,1), vec!(1,3)));
		assert_eq!(a, b);
	}
	
	#[test]
	fn test_encode_decode() {
		let text = vec!(1,2,1);
		let vocab = vec!(vec!(1,2));
		let encoded_text = encode(&text, &vocab);
		let final_text = decode(encoded_text, &vocab);
		assert_eq!(text, final_text);
	}
}
// vim: ts=2 sw=2
