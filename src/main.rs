
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use rustc_hash::FxHashMap;

mod counter;
use counter::Counter;

mod utils;
use utils::{replace_in_library, fertility, Token, BASE_TOKENS, find_candidate, ticker};

mod picky_bpe;
use picky_bpe::PickyBpe;

mod scaffold_bpe;
use scaffold_bpe::ScaffoldBpe;

mod hypergreedy_bpe;
use hypergreedy_bpe::bpe_hypergreedy;

use rayon::prelude::*;
use std::env;

static NEW_TOKEN_COUNT : u16 = 1000;
#[allow(unused)]
static TOTAL_TOKENS : u16 = NEW_TOKEN_COUNT + BASE_TOKENS; 
//above checks if our desired number of tokens is possible inside a u16

//token numbers from 0 to 255 (inclusive) represent the raw bytes
//while tokens greater than that represent compressions

fn main() -> io::Result<()> {
	let file_argument : String = env::args().nth(1).unwrap_or(String::from("./AliceInWonderland.txt"));

	let library = gen_word_counts(file_argument, sometimes_logger);
	println!();
	println!("word counts generated. distinct word count: {}", library.len());
	
	let progress = ticker::<NEW_TOKEN_COUNT>;

	let (_vocab, compressed_lib) = bpe(library.clone(), progress);
	let bpe_fertility = fertility(&compressed_lib);
	println!();
	println!("bpe : fertility {}", bpe_fertility);
	
	let (_vocab, compressed_lib) = bpe_hypergreedy(library.clone(),
		NEW_TOKEN_COUNT, progress);
	let hypergreedy_fertility = fertility(&compressed_lib);
	println!();
	println!("hypergreedy bpe : fertility {}", hypergreedy_fertility);
	
	let threshold = 0.9;
	println!();
	println!("training picky bpe with threshold {}...", threshold);
	let mut picky = PickyBpe::new();
	let ( _, picky_lib ) = picky.train(library.clone(), threshold, NEW_TOKEN_COUNT, progress);
	
	//this just tests the tokenizing code (algorithm 2), it isn't needed.
	let mut picky_lib2: Counter<Token> = Counter::new();
	for (word, &count) in &library {
		let tokenized = picky.tokenize(word);
		picky_lib2[&tokenized] += count;
	}
	assert_eq!(picky_lib, picky_lib2);
	let picky_fertility = fertility(&picky_lib);
	println!("picky bpe (threshold={}) : fertility {}", threshold, picky_fertility);
	
	println!();
	println!("training scaffold bpe...");
	let mut scaffold = ScaffoldBpe::new();
	scaffold.train(&library, NEW_TOKEN_COUNT, progress);
	
	let mut scaffold_lib: Counter<Token> = Counter::new();
	for (word, &count) in &library {
		let tokenized = scaffold.tokenize(word);
		scaffold_lib[&tokenized] += count;
	}
	let scaffold_fertility = fertility(&scaffold_lib);
	println!("scaffold bpe : fertility {}", scaffold_fertility);
	println!();
	println!("improvement ratio (hypergreedy/bpe): {}", hypergreedy_fertility / bpe_fertility);
	println!("improvement ratio (picky/bpe): {}", picky_fertility / bpe_fertility);
	println!("improvement ratio (scaffold/bpe): {}", scaffold_fertility / bpe_fertility);
	
	Ok(())
}

///logs what line we are currently reading from the text file
///every once and a while
///
///you ought to print a new line before printing anything else
///because this function fails to print every time
fn sometimes_logger(i : usize) {
	//move to 1 based indexing
	let i = i + 1;
	if i % (1<<20) == 0 {
		print!("\rreading line: {} ", i);
		io::stdout().flush().unwrap();
	}
}

fn bpe<F : Fn(u16)>(mut library : Counter<Token>, progress_fn : F) -> (Vec<Token>, Counter<Token>){
	//vocab[i] is the expansion of token number (i - base_tokens)
	let mut vocab : Vec<Token> = Vec::with_capacity(NEW_TOKEN_COUNT.into());
	for i in 0..NEW_TOKEN_COUNT {
		let Some((new_token, _)) = find_candidate(&library) else {
			println!("no compression is possible at {} new tokens", i);
			break;
		};
		library = replace_in_library(&library, &new_token, i + BASE_TOKENS);
		vocab.push(new_token);
		progress_fn(i);
	}
	(vocab, library)
}

fn gen_word_counts<P>(filename : P, progress_fn : fn(usize)) -> Counter<Token>
where P: AsRef<Path>{
	use unicode_segmentation::UnicodeSegmentation;
	let lines = read_lines(filename).unwrap();
	let word_counts : Counter<Token> =
		lines.map_while(Result::ok).enumerate().par_bridge().fold(
			|| Counter::new(),
			|mut counter, (i,x)| {
			progress_fn(i);
			//WARNING:
			//unicode_words effectively strips all the punctuation and whitespace 
			// from the dataset and I expect it to behave perversely on Chinese and Japanese
			// consider switching with .split(' ')
			// if you are willing to pay the performance cost
			// and it is appropriate for your language
			counter.update(x.unicode_words().map( 
				//turn words into Vec<u16>s
				|s| s.as_bytes().into_iter().map(|&b| b as u16).collect::<Token>()
			));
			counter
		}).sum();
	return word_counts;
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>> 
where P: AsRef<Path>, {
	let file = File::open(filename)?;
	Ok(io::BufReader::new(file).lines())
}

#[allow(unused)]
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

#[allow(unused)]
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
#[allow(unused)]
fn to_string(t : &Token) -> String {
	let x = 
	t.iter().map(|&u| {
		if u < 256 { u as u8 } else {'?' as u8}
	}).collect::<Vec<u8>>();
	
	String::from_utf8_lossy(&x).to_string()
}

#[cfg(test)]
mod test {
	use super::*;

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
