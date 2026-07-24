use crate::counter::Counter;
use crate::utils::{BASE_TOKENS, 
	Token, expand, expand_in_library,
	find_candidate, replace, replace_in_library,
	token_freq};

#[derive(Clone, Debug)]
pub enum Event {
	///fromA, fromB, to
	Merge(u16, u16, u16),
	///from, toA, toB
	Remove(u16, u16, u16),
}

pub struct PickyBpe {
	pub vocab: Vec<Token>,
	pub events: Vec<Event>,
}

impl PickyBpe {
	pub fn new() -> Self {
		PickyBpe {
			vocab: Vec::new(),
			events: Vec::new(),
		}
	}

	pub fn train<F : Fn(u16)>(&mut self, mut library: Counter<Token>,
				threshold: f64, new_token_count: u16, progress_fn : F) 
					-> (Vec<Token>, Counter<Token>) {
		
		let mut effective_count: u16 = 0;

		while effective_count < new_token_count {
			let mut newly_removed: Vec<u16> = Vec::new();

			let Some((pair, pair_freq)) = find_candidate(&library) else {
				println!("no compression possible at {} effective tokens", effective_count);
				break;
			};

			let x1 = pair[0]; 
			let x2 = pair[1];
			let x3 = pair;

			let x1_freq: usize = token_freq(&library, x1);
			let x2_freq: usize = token_freq(&library, x2);

			let new_token_id = self.vocab.len() as u16 + BASE_TOKENS;
			self.events.push(Event::Merge(x1, x2, new_token_id));
			self.vocab.push(x3);
			let x3 = self.vocab.last().unwrap();
			assert_ne!(x1_freq, 0);
			assert_ne!(x2_freq, 0);
			
			let ios_x1 = pair_freq as f64 / x1_freq as f64;
			//>= is what the psudocode in the paper uses, however, since ios can be 1, it means
			//their claim that if threshold is equal to 1, no removals are possible is false.
			if ios_x1 >= threshold && x1 >= BASE_TOKENS {
				//expantion of x1
				let [xx1, xx2] = self.expansion_of(x1);
				self.events.push(Event::Remove(x1,xx1,xx2));
				newly_removed.push(x1);
				//Note that unlike algorithm 1's pseudocode, we don't remove from vocab.
				//This is because the length of vocab keeps track of yet unused token IDs
				//and we use it as a map of "tokenId -> expansion"
				//so deleting elements messed up the map
			}

			if x2 != x1 {
				let ios_x2 = pair_freq as f64 / x2_freq as f64;
				if ios_x2 >= threshold && x2 >= BASE_TOKENS {

					//expantion of x2
					let [xx1, xx2] = self.expansion_of(x2);

					self.events.push(Event::Remove(x2,xx1,xx2));
					newly_removed.push(x2);
				}
			}

			library = replace_in_library(&library, &x3, new_token_id);

			for &removed_id in &newly_removed {
				library = expand_in_library(&library, removed_id, 
					&self.expansion_of(removed_id));
			}

			effective_count += 1;
			effective_count -= newly_removed.len() as u16;
			
			progress_fn(effective_count);
		}
		println!();
		eprintln!("Events: {} merges, {} removals, {} effective tokens",
			self.events.iter().filter(|e| matches!(e, Event::Merge(_, _,_))).count(),
			self.events.iter().filter(|e| matches!(e, Event::Remove(_,_,_))).count(),
			effective_count);
		
		//(self.vocab is NOT V from algorithm 1)
		//namely removed tokens are next removed from it. However, this shouldn't change behavior.
		(self.vocab.clone(), library)
	}

	pub fn tokenize(&self, word: &[u16]) -> Vec<u16> {
		let mut current: Vec<u16> = word.to_vec();
		//this is a simplified form of algorithm-2 which doesn't optimize 
		//by skipping steps. However, it should be equivalent.
		for event in &self.events {
			match event {
				Event::Merge(from1, from2, to) => {
					current = replace(&current, &[*from1, *from2], *to);
				}
				Event::Remove(from, to1, to2 ) => {
					current = expand(&current, *from, &[*to1, *to2]);
				}
			}
		}

		current
	}

	fn expansion_of(&self, token_id: u16) -> [u16; 2]{
		let index : usize = (token_id - BASE_TOKENS) as usize;
		let xx1 = self.vocab[index][0];
		let xx2 = self.vocab[index][1]; 
		return [xx1, xx2];
	}
}

#[cfg(test)]
mod test {
	use super::*;
	
	#[test]
	fn test_decompose_in_library() {
		let mut library : Counter<Token> = Counter::new();
		library.update([vec![1,2,3], vec![1,1], vec![4,4], vec![1,2,3]]);
		let mut library2 : Counter<Token> = Counter::new();
		library2.update([vec![1,2,3], vec![1,1], vec![3,2,3,2], vec![1,2,3]]);
		assert_eq!(expand_in_library(&library, 4, &[3,2]), library2);
	}
}
// vim: ts=2 sw=2
