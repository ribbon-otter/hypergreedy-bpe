// based on https://docs.rs/crate/counter
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap;
use std::ops::{MulAssign, AddAssign, Index, *};
use std::cmp::Eq;
use std::hash::Hash;
use std::iter::Sum;
use std::fmt::Debug;

use rayon::prelude::*; 

#[derive(Clone, Debug)]
pub struct Counter<T> {
	map: FxHashMap<T, usize>,
	//cache of the current most common value 
	//
	//it's public so you can manually fix it
	//if you break it with mutable indexing
	pub current_max : Option<(T, usize)>,
}

impl<T> IntoParallelIterator for Counter<T> 
where T: Hash + Eq + Clone + Send + Sync {

	type Iter = <FxHashMap<T, usize> as IntoParallelIterator>::Iter;
	type Item = <FxHashMap<T, usize> as IntoParallelIterator>::Item;

	fn into_par_iter(self) -> Self::Iter {
		self.map.into_par_iter()
	}
}

impl<'data, T> IntoParallelRefIterator<'data> for Counter<T>
where T: 'data + Hash + Eq + Clone + Sync {

	type Iter = <&'data FxHashMap<T, usize> as IntoParallelIterator>::Iter;
	type Item = <&'data FxHashMap<T, usize> as IntoParallelIterator>::Item;

	fn par_iter(&'data self) -> Self::Iter {
		self.map.par_iter()
	}
}


impl<'a, T> IntoIterator for &'a Counter<T>
where T: 'a + Hash + Eq + Clone {
		type Item = (&'a T, &'a usize);
		type IntoIter = std::collections::hash_map::Iter<'a, T, usize>;

		// Required method
		fn into_iter(self) -> Self::IntoIter {
			self.map.iter()
		}
}

impl<T: std::cmp::Eq> PartialEq for Counter<T>
where T: Hash + Eq
{
	fn eq(&self, other: &Self) -> bool {
		// ingore the zero
		self.map == other.map
	}
}

impl<T> Counter<T> 
where T: Hash + Eq + Clone + Sync
{
	pub fn new() -> Self {
		Counter {
			map: FxHashMap::<T, usize>::default(),
			current_max : None
		}
	}
	
	pub fn with_capacity(capacity: usize) -> Self {
		use std::collections::HashMap;
		Counter {
			map: HashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default()),
			current_max: None
		}
	}

	pub fn update<I>(&mut self, iterable: I) 
	where
		I: IntoIterator<Item = T>,
	{
		for item in iterable {
			let entry = self.map.entry(item.clone()).or_insert(0);
			*entry += 1;
			let value = *entry;
			private_check_current_max(&mut self.current_max, &item, value);
			}
	}

	pub fn update_weighted<I>(&mut self, iterable: I, weight : usize) 
	where
		I: IntoIterator<Item = T>,
	{
		for item in iterable {
			let entry = self.map.entry(item.clone()).or_insert(0);
			*entry += weight;
			let value = *entry;
			private_check_current_max(&mut self.current_max, &item, value);
		}
	}

	pub fn most_common(&self) -> Option<(T, usize)>
	{
		self.current_max.clone()
	}

	pub fn len(&self) -> usize {
		self.map.len()
	}

	pub fn is_empty(&self) -> bool {
		self.map.is_empty()
	}
	///total number of items which have been counted
	pub fn total(&self) -> usize {
		self.into_iter().map(|(_key, value)| value).sum()
	}
	
	pub fn into_vec(self) -> Vec<(T, usize)> {
		self.map.into_iter().collect()
	}
}

//used to restore current_max invariant in some functions
fn private_check_current_max<T>(current_max : &mut Option<(T, usize)>, key : &T, value: usize) where T: Hash + Eq + Clone {
	if let Some((_key, count)) = current_max {
		if value > *count {
			*current_max = Some((key.clone(), value));
		}
	} else {
			*current_max = Some((key.clone(), value));
	}
}

impl<T> Sum for Counter<T> 
where T: Hash + Eq + Clone + Debug + Sync {
	fn sum<I>(mut iter: I) -> Self
		where I: Iterator<Item = Counter<T>> {
		let mut result = iter.next().unwrap_or_else(Counter::new); 
		for i in iter {
			result += i;
		}
		result
	}
}

impl<T> AddAssign<Counter<T>> for Counter<T> 
where T: Hash + Eq + Clone + Debug {
	fn add_assign(&mut self, rhs: Counter<T>) {
		for (key, val) in rhs {
			self[&key] += val;
			let new_total = self[&key];
			private_check_current_max(&mut self.current_max, &key, new_total);
		}
	}
}

static ZERO : usize = 0;

impl<T> Index<&T> for Counter<T>
where T: Hash + Eq {
	type Output = usize;
	fn index(&self, index: &T) -> &Self::Output {
		&self.map.get(index).unwrap_or(&ZERO)
	}
}

///watch out! you may need to manually fix self.current_max
impl<T> IndexMut<&T> for Counter<T>
where T: Hash + Eq + Clone {
	fn index_mut(&mut self, index: &T) -> &mut Self::Output {
		self.map.entry((*index).clone()).or_insert(0)
	}
}

impl<T> MulAssign<usize> for Counter<T> {
	fn mul_assign(&mut self, rhs: usize) {
		for (_, val) in &mut self.map {
			*val *= rhs; 
		}
		if let Some(cm) =  &mut self.current_max {
			//multiplying preserves maximiums so we only need to update
			cm.1 *= rhs;
		}
	}
}

impl<T> Mul<usize> for Counter<T> {
	type Output = Counter<T>;
	fn mul(mut self, rhs: usize) -> Counter<T> {
		for (_, val) in &mut self.map {
			*val *= rhs
		}
		if let Some(max) = &mut self.current_max {
			(*max).1 *= rhs;
		}
		self
	}
}

impl<T> IntoIterator for Counter<T> {
	type Item = (T, usize);
	type IntoIter = std::collections::hash_map::IntoIter<T, usize>;
	fn into_iter(self) -> Self::IntoIter {
		self.map.into_iter()
	}
}

impl<'a, T, > IntoIterator for &'a mut Counter<T> {
	type Item = (&'a T, &'a mut usize);
	type IntoIter = std::collections::hash_map::IterMut<'a, T, usize>;
	fn into_iter(self) -> Self::IntoIter {
		self.map.iter_mut()
	}
}

impl<T> FromIterator<T> for Counter<T>
where T: Hash + Eq + Clone + Sync,
{
	fn from_iter<I: IntoIterator<Item = T>>(iterable: I) -> Self {
		let mut counter = Counter::new();
		counter.update(iterable);
		counter
	}
}

#[cfg(test)]
mod test {
	use super::*;
	
	#[test]
	fn test_basic() {
		let mut c = Counter::new();
		c.update(vec!("a", "b", "a"));
		assert_eq!(c.most_common().unwrap().1, 2);
		assert_eq!(c.most_common().unwrap().0, "a");
		c.update(vec!("b", "b", "b"));
		assert_eq!(c.most_common().unwrap().1, 4);
		assert_eq!(c.most_common().unwrap().0, "b");
	}
	#[test]
	fn test_weighted_update() {
		let mut c = Counter::new();
		c.update_weighted(vec!("a", "b", "a"), 3);
		assert_eq!(c.most_common().unwrap().1, 6);
		c.update_weighted(vec!("b","b", "b", "b"), 1);
		assert_eq!(c.most_common().unwrap().1, 7);
		assert_eq!(c[&"a"], 6);
	}
	#[test]
	fn test_index() {
		let mut c = Counter::new();
		c.update(vec!("a", "b", "a"));
		assert_eq!(c[&"a"], 2);
		assert_eq!(c[&"c"], 0);
		assert_eq!(c[&"b"], 1);
	}
	#[test]
	fn test_add() {
		let mut c = Counter::new();
		c.update(vec!("a", "b", "a"));
		let mut d = Counter::new();
		d.update(vec!("a", "b", "a", "a", "c"));
		c += d;
		assert_eq!(c[&"a"], 5);
		assert_eq!(c[&"b"], 2);
		assert_eq!(c[&"c"], 1);
		assert_eq!(c.most_common().unwrap().1, 5);
		assert_eq!(c.most_common().unwrap().0, "a");
	}
	#[test]
	fn test_mul() {
		let mut c = Counter::new();
		c.update(vec!("a", "b", "a"));
		c *= 2;
		assert_eq!(c.most_common().unwrap().1, 4);
		assert_eq!(c.most_common().unwrap().0, "a");
		c *= 3;
		assert_eq!(c.most_common().unwrap().1, 12);
	}
	#[test]
	fn test_sum() {
		let mut c = Counter::new();
		c.update(vec!("a", "b", "a"));
		let v : Vec<Counter<_>> = vec!(c.clone(), c.clone());
		let vv : Counter<_> = v.into_iter().sum();
		assert_eq!(vv.most_common().unwrap().1, 4);
		assert_eq!(vv.most_common().unwrap().0, "a");

	}
}

// vim: ts=2 sw=2
