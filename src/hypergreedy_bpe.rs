use crate::counter::Counter;
use crate::utils::{replace_in_library, Token, BASE_TOKENS, find_candidate};
use rayon::prelude::*;


pub fn bpe_hypergreedy<F: Fn(u16)>(mut library: Counter<Token>, 
        new_tokens : u16, progress_fn: F) -> (Vec<Token>, Counter<Token>) {

    let mut vocab: Vec<Token> = Vec::new();
    for i in 0..new_tokens {
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

fn find_best_token(library: &Counter<Token>) -> Option<(Token, usize)> {
    let mut can = find_candidate(library)?;
    loop {
        let maybe_ext = find_best_extention(library, &can.0);
        if let Some(ext) = maybe_ext {
            if ext.1 * 2 > can.1 { //if the extention occurs more than half as often
                can = ext;
            } else {
                break Some(can);
            }
        } else {
            break Some(can);
        }
    }
}

fn find_best_extention(library: &Counter<Token>, candidate: &Token) -> Option<(Token, usize)> {
    let extention_counts: Counter<&[u16]> =
        library.par_iter().fold(
            || Counter::new(),
            |mut counter, (t, &weight)| {
                counter.update_weighted(
                    t.windows(candidate.len() + 1)
                        .filter(|win| {
                            win[0..win.len() - 1] == *candidate
                                || win[1..win.len()] == *candidate
                        })
                        .map(|a| a),
                    weight,
                );
                counter
            }
        ).sum();
    extention_counts.most_common().map(|(token, weight)| (token.to_vec(), weight))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::Counter;

    #[test]
    fn test_find_best_extention_right() {
        let mut c = Counter::new();
        c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
        let a = find_best_extention(&c, &vec!(1,1));
        let aa = a.unwrap();
        assert_eq!(aa.0, vec!(1,1,2));
        assert_eq!(aa.1, 2);
    }

    #[test]
    fn test_find_best_extention_left() {
        let mut c = Counter::new();
        c.update(vec!(vec!(1,1,2), vec!(1,1), vec!(1,1,2)));
        let a = find_best_extention(&c, &vec!(1,2));
        let aa = a.unwrap();
        assert_eq!(aa.0, vec!(1,1,2));
        assert_eq!(aa.1, 2);
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
        assert_eq!(a.0, vec!(1,1,2));
        assert_eq!(a.1, 2);
    }

    #[test]
    fn test_find_best_token_no_compression_possible() {
        let mut c = Counter::new();
        c.update(vec!(vec!(1), vec!(1), vec!(1)));
        let a = find_best_token(&c);
        assert_eq!(a, None);
    }
}
