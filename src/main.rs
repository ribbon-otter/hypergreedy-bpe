
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use rustc_hash::FxHashMap;

mod counter;
use counter::Counter;

mod utils;
use utils::{replace, replace_in_library, fertility, Token, BASE_TOKENS};

mod picky_bpe;
use picky_bpe::PickyBpe;

mod scaffold_bpe;
use scaffold_bpe::ScaffoldBpe;

use rayon::prelude::*;
use std::env;

///extension_population * EXT_RATIO > candidate_population 
///triggers using the extension
static EXT_RATIO : usize = 2;

///beyond the 256 tokens that already exist, how many should we add?
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
	
	let (vocab, compressed_lib) = bpe_hypergreedy(library.clone(), ticker);
	let opt_fertility = fertility(&compressed_lib);
	println!();
	println!("hypergreedy bpe : fertility {}", fertility(&compressed_lib));
	println!("hypergreedy bpe: {:?}", vocab.iter().take(10).map(to_string).collect::<Vec<_>>());
	
	let (vocab, compressed_lib) = bpe(library.clone(), ticker);
	let old_fertility = fertility(&compressed_lib);
	println!();
	println!("bpe : fertility {}", old_fertility);
	println!("bpe: {:?}", vocab.iter().take(10).map(to_string).collect::<Vec<_>>());
	
	// Sanity check: Picky BPE with no removals should match vanilla BPE exactly
	println!();
	println!("sanity check: picky bpe with no removals (threshold=inf)...");
	let mut picky_noremove = PickyBpe::new(f64::INFINITY);
	picky_noremove.train(&library, NEW_TOKEN_COUNT);
	
	let mut picky_noremove_lib: Counter<Token> = Counter::new();
	for (word, &count) in &library {
		let tokenized = picky_noremove.tokenize(word);
		picky_noremove_lib[&tokenized] += count;
	}
	let picky_noremove_fertility = fertility(&picky_noremove_lib);
	println!("picky bpe (no removals) : fertility {}", picky_noremove_fertility);
	println!("vanilla bpe             : fertility {}", old_fertility);
	let diff = (picky_noremove_fertility - old_fertility).abs();
	if diff < 0.0001 {
		println!("MATCH: picky bpe (no removals) matches vanilla bpe perfectly");
	} else {
		eprintln!("MISMATCH: difference = {}", diff);
	}

	let threshold = 0.9;
	println!();
	println!("training picky bpe with threshold {}...", threshold);
	let mut picky = PickyBpe::new(threshold);
	picky.train(&library, NEW_TOKEN_COUNT);
	
	let mut picky_lib: Counter<Token> = Counter::new();
	for (word, &count) in &library {
		let tokenized = picky.tokenize(word);
		picky_lib[&tokenized] += count;
	}
	let picky_fertility = fertility(&picky_lib);
	println!("picky bpe (threshold={}) : fertility {}", threshold, picky_fertility);
	
	println!();
	println!("improvement ratio (hypergreedy/bpe): {}", opt_fertility / old_fertility);
	println!("improvement ratio (picky/bpe): {}", picky_fertility / old_fertility);
	
	// Spot check: verify specific word tokenizations match between no-removal picky and vanilla BPE
	println!();
	println!("spot check: comparing tokenizations against vanilla BPE...");
	let (bpe_vocab, _bpe_compressed) = bpe(library.clone(), ticker);
	let mut sample_words: Vec<Token> = Vec::new();
	for (word, _) in (&library).into_iter().take(20) {
		sample_words.push(word.clone());
	}
	let mut all_match = true;
	for word in &sample_words {
		let picky_tokens = picky_noremove.tokenize(word);
		let compressed = apply_bpe_merges(word, &bpe_vocab);
		if picky_tokens != compressed {
			eprintln!("MISMATCH on {:?}: picky={:?} bpe={:?}",
				String::from_utf8_lossy(&word.iter().map(|&b| b as u8).collect::<Vec<_>>()),
				picky_tokens, compressed);
			all_match = false;
		}
	}
	if all_match {
		println!("all {} sample words match", sample_words.len());
	}
	println!();
	println!("vocabulary size comparison:");
	let picky_merges = picky.events.iter().filter(|e| matches!(e, picky_bpe::Event::Merge(_, _))).count();
	let picky_removals = picky.events.iter().filter(|e| matches!(e, picky_bpe::Event::Remove(_))).count();
	println!("  vanilla bpe: {} tokens", bpe_vocab.len());
	println!("  picky bpe (T={}): {} merges - {} removals = {} effective tokens", threshold,
		picky_merges, picky_removals, picky_merges - picky_removals);

	println!();
	println!("training scaffold bpe...");
	let mut scaffold = ScaffoldBpe::new();
	scaffold.train(&library, NEW_TOKEN_COUNT);
	
	let mut scaffold_lib: Counter<Token> = Counter::new();
	for (word, &count) in &library {
		let tokenized = scaffold.tokenize(word);
		scaffold_lib[&tokenized] += count;
	}
	let scaffold_fertility = fertility(&scaffold_lib);
	println!("scaffold bpe : fertility {}", scaffold_fertility);
	println!();
	println!("improvement ratio (scaffold/bpe): {}", scaffold_fertility / old_fertility);
	
	Ok(())
}

///progress bar for training tokens
///that *very* roughly fills 80 columns with periods 
///as we progress
fn ticker(i : u16) {
	if i % (1+(NEW_TOKEN_COUNT / (80 - 1))) == 0 {
		print!(".");
		io::stdout().flush().unwrap();
	}
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

fn bpe_hypergreedy<F : Fn(u16)>(mut library : Counter<Token>, progress_fn : F) -> (Vec<Token>, Counter<Token>) {
	//vocab[i] is the expansion of token number (i - base_tokens)
	let mut vocab : Vec<Token> = Vec::with_capacity(NEW_TOKEN_COUNT.into());
	for i in 0..NEW_TOKEN_COUNT {
		let Some((new_token, _)) = find_best_token(&library) else {
			println!("no compression is possible at {} new tokens", i);
			break;
		};
		library = replace_in_library(&library, &new_token, i + BASE_TOKENS);
		vocab.push(new_token);
		progress_fn(i);
	}
	(vocab, library)
}


///returns None if no more compression is possible because every word in the library
///in only one token long
///otherwise returns the best token
fn find_best_token(library : &Counter<Token>) -> Option<(Token, usize)> {
	let mut can = find_candidate(&library)?;
	loop {
		
		let maybe_ext = find_best_extention(&library, &can.0);
		//if we find at least one valid extension
		if let Some(ext) = maybe_ext {
			//if the extension occurs often enough
			if  ext.1 * EXT_RATIO > can.1 {
			//the candidate is the true extension
			can = ext
			} else {
				break Some(can)
			}
		} else {
			//otherwise we found the best candidate
			break Some(can)
		}
	}
}

///find the most commonly occurring byte pair in the library
fn find_candidate(library : &Counter<Token>) -> Option<(Token, usize)> {
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

///finds the bests extension; if an extension can't exist 
///(because this the candidate is already the longest)
fn find_best_extention(library : &Counter<Token>, candidate : &Token) -> Option<(Token, usize)> {
	let extention_counts : Counter<&[u16]> = 
 		library.par_iter().fold(
			|| Counter::new(),
			|mut counter, (t, &weight)| {
				counter.update_weighted(
					t.windows(candidate.len()+1)
				 .filter(
							|win| win[0..win.len()-1] == *candidate //try extending backwards
								|| win[1..win.len()] == *candidate //try extending forwards
					).map(|a| a), weight);
					counter
			}
		).sum();
	//convert to owned vector
	extention_counts.most_common().map(|(token, weight)| (token.to_vec(), weight))
}

fn apply_bpe_merges(word: &Token, vocab: &Vec<Token>) -> Token {
	let mut current = word.clone();
	for (i, token) in vocab.iter().enumerate() {
		current = replace(&current, token, i as u16 + BASE_TOKENS);
	}
	current
}

#[allow(unused)]
fn echo<T> (a : T, prefix : &str) -> T 
	where T : std::fmt::Debug {
	println!("{:?}: {:?}", prefix, a);
	a
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
		let a = find_candidate(&c).unwrap();
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
		let a = find_best_token(&c).unwrap();
		assert_eq!(a.0, vec!(1,1,2) );
		assert_eq!(a.1, 2 );
	}
	
	#[test]
	fn test_find_best_token_no_compression_possible() {
		let mut c = Counter::new();
		c.update(vec!(vec!(1), vec!(1), vec!(1)));
		let a = find_best_token(&c);
		assert_eq!(a, None);
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
