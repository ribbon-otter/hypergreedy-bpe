An potential improvement to BPE (in the context of LLMs). Hypergreedy BPE attempts to allocate tokens more efficiently to get more out of N tokens. In traditional BPE, each higher-level token is mapped to just 2 other tokens. Thus, to represent a common 3 character string, there must be tokens for a 2 character pair inside it, even if that 2 character pair rarely occurs outside of the larger string. This wastes token space since that 2 character pair won't happen.

At a high-level hypergreedy BPE work like thus: try to extend potential pairs into longer strings when those strings are common enough. Note that unlike BPE, hypergreedy maps each token to a string of tokens rather than just a pair. This makes the 'pair' part of hypergreedy somewhat a misnomer.

More detailed:
1. find the most common pair, call that this the candidate.
2. Consider extending the candidate by one byte forwards or backwards. The one which occurs the most often is called the 'extention'
3. If the extention happens more than half as often as the candidate, the extention becomes candidate. Then go back to step 2. In other words, the extention becomes the candidate if the candidate mostly occurs inside the extention.
4. Once a maximially long candidate is found, we make it a token and replace all occurences of it.
Repeat until you have your target number of tokens.

In my limited testing, Hypergreedy BPE appears to have a 3% to 4% improvement in token fertility (average number of tokens per word).
