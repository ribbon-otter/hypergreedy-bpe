An improvement to BPE tokenisation (in the context of LLMs). Hypergreedy BPE tackles the challenge of 'junk' tokens in BPE [1](https://arxiv.org/pdf/2004.03720). Also called scaffolding tokens, these are tokens which primarily occur during the encoding and decoding steps of tokenisation but rarely in the final fulling tokenized LLM input. This results in LLMs being under-trained on these tokens, because of their scarcity in the training corpus. Moreover, these scaffolding tokens waste finite vocabulary space and thus weights during the embedding stage of a LLM.

In traditional BPE, each higher-level token is mapped to just 2 other tokens. Thus, to represent a common 3 character string, there must be tokens for a 2 character pair inside it, even if that 2 character pair rarely occurs outside of the larger string. This is the origin of the scaffolding tokens. 

At a high-level hypergreedy BPE work like thus: greedily extend a potential pairs to a longer string if the longer string also occurs frequently. Note that unlike BPE, hypergreedy maps each token to a string of tokens rather than just a pair. This makes the 'pair' part of hypergreedy BPE somewhat a misnomer.

More detailed:
1. find the most common pair, call that this the candidate.
2. Consider extending the candidate by one byte forwards or backwards. The one which occurs the most often is called the 'extension'
3. If the extension happens more than half as often as the candidate, the extension becomes candidate. Then go back to step 2. In other words, the extension becomes the candidate if the candidate mostly occurs inside the extension.
4. Once a maximally long candidate is found, we make it a token and replace all occurrences of it.
Repeat until you have your target number of tokens.

In my preliminary testing, Hypergreedy BPE appears to have a 1% to 2% improvement in token fertility (average number of tokens per word).

This code-base uses the optimization of first splitting by word and then working over that rather than the raw text. This means that tokens can not be found that cross word boundaries. This is not inherent to the abstract hypergreedy BPE algorithm, merely this implementation of it.

## how to use
```
cargo run --release -- <path to text file you wish to train on>
```
Currently, the two tokenisers are trained and stats are printed and then everything is thrown away. No tokeniser is saved to disk. 

We use `--release` because this significantly improves the speed of training.

### Cautions
You likely want to change `NEW_TOKEN_COUNT` in `src/main.rs` to a number appropriate for your text file. It represents the number of new tokens 

The code currently uses `.unicode_words()` to split the text into words first. This effectively strips all punctuation, and thus a different word splitting strategy should be used if punctuation is important. It may also fail to compress Chinese or Japanese writing meaningfully as each Chinese character is considered a separate word. (And tokens can not be constructed across word boundaries)

If your language has spaces between words, and you can tolerate the performance penalty of more words to tokenize over, `.split(' ')` can work as an alternative.

# related work
- [Picky-BPE](https://arxiv.org/html/2409.04599) - removes tokens if when building a later bigger token, those tokens are found to mostly occur inside this bigger token.  
- [Scaffold-BPE](https://arxiv.org/html/2404.17808v3) - explicitly keeps a labelling tokens as either scaffold or non-scaffold during training
- [LiteToken](https://arxiv.org/html/2602.04706v1) - recognizes that these scaffolding tokens are a problem and offers a method to removes them after training.

Hypergreedy BPE differs from all of these because as a greedy algorithm, it doesn't remove tokens, nor does it involve different lists of tokens used during tokenization and what is ultimately passed to the model.
