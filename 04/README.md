## Architecture

[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
[Attention is All You Need](https://arxiv.org/pdf/1706.03762)

## Tokenization

[karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)
[Language Model Tokenizers Introduce Unfairness Between Languages](https://openreview.net/pdf?id=Pj4YYuxTq9)
[All languages are NOT created (tokenized) equal](https://www.artfish.ai/p/all-languages-are-not-created-tokenized)


Language  |  Tokens per Word (approx.)| Baseline (English=1.0)
----------|---------------------------|-------------------------
English   |  1.3                      | 1.0
French    |  2.0                      | 1.5
German    |  2.1                      | 1.6
Spanish   |  2.1                      | 1.6
Chinese   |  2.5                      | 1.9
Hindi     |  3.0–5.0+                 | 2.3

[Tokenization Algorithms compared](https://www.arxiv.org/pdf/2509.15255)

[openai counting tokens](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
. 1 token ≈ 4 characters

. 1 token ≈ ¾ of a word

. 100 tokens ≈ 75 words

. 1–2 sentences ≈ 30 tokens

. 1 paragraph ≈ 100 tokens

. ~1,500 words ≈ 2,048 tokens
[openai tokenizer](https://platform.openai.com/tokenizer)

## embeddings

[vector visualization](https://medium.com/the-muse-junction/understanding-vector-spaces-the-foundation-of-machine-learning-ai-faa3b4f50668)
[alphaearth](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/)
[choose the right embedding model](https://levelup.gitconnected.com/how-to-choose-the-right-embedding-model-for-your-rag-application-44e30876d382)