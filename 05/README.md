## From Text to Meaning: Tokenization & Embeddings

### Tokenization

Language models don't see text like you and I, instead they see a sequence of numbers (known as tokens). `Byte pair encoding (BPE)` is a way of converting text into tokens. It has a couple desirable properties:

1. It's **reversible** and **lossless**, so you can convert tokens back into the original text, and so is **deterministic**: the same text will always produce the same token sequence.
2. It works on **arbitrary** text, even text that is not in the tokeniser's training data, treating unknown words with special subword tokens, e.g. "[UNK]", or breaking them down into smaller known tokens.
3. It **compresses** the text: the token sequence is shorter than the bytes corresponding to the original text. On average, in practice, each token corresponds to about 4 bytes.
4. It attempts to let the model see **common subwords**. For instance, "ing" is a common subword in English, so  encodings (like BPE) will often split "encoding" into tokens like "encod" and "ing" (instead of e.g. "enc" and "oding"). Because the model will then see the "ing" token again and again in different contexts, it helps models generalise and better understand grammar.

**Apri bene le orecchie!👂 (sì, in italiano)**
> 1. It's not like encode/decode a text in UTF-8 or ASCII, which are byte-level encodings with a fixed mapping from characters to bytes.</br> 
Instead, tokenization algorithms **learn** a vocabulary of tokens from a **training corpus** of text, and use that vocabulary to encode text into tokens and decode tokens back into text.<br/>
Once trained, the tokenizer uses a fixed mapping and rules from vocabulary tokens to token IDs (integers) for encoding/decoding.

> 2. Tokenizers are typically **trained** on a corpus that is not necessarily the same as the training corpus for the language model itself, and can be used across different models.

#### Tokenization strategies

<details>
<summary>Different levels of tokenization</summary>

For a word with morphology like "unhappiness":
- **Word-level**: `["unhappiness"]` (1 token)
- **Character-level**: `["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"]` (11 tokens)
- **Subword-level**: `["un", "happiness"]` or `["un", "hap", "pi", "ness"]` (2-4 tokens)

For a sentence like "The cat sat on the mat.":
- **Word-level**: `["The", "cat", "sat", "on", "the", "mat", "."]` (7 tokens)
- **Character-level**: `["T", "h", "e", " ", "c", "a", "t", " ", "s", "a", "t", " ", "o", "n", " ", "t", "h", "e", " ", "m", "a", "t", "."]` (23 tokens)
- **Subword-level**: `["The", "cat", "sat", "on", "the", "mat", "."]` (7 tokens, same as word-level in this case)

Tokenization affects three critical aspects of LLM performance:

**1. Vocabulary Size vs. Sequence Length Trade-off**
- Smaller tokens (like characters) mean shorter vocabularies but longer sequences
- Larger tokens (like words) mean bigger vocabularies but shorter sequences
- Sequence length directly impacts computational cost (quadratic in attention mechanisms)

**2. Out-of-Vocabulary (OOV) Handling**
- Word-level tokenization struggles with unseen words, typos, or rare terms
- Character-level tokenization eliminates OOV but loses semantic meaning
- Subword tokenization balances both concerns

**3. Cross-lingual and Multilingual Support**
- Different languages have vastly different word formation rules
- Some languages don't use spaces (e.g., Chinese, Japanese)
- Subword tokenization provides language-agnostic solutions

</details>

#### Character-Level Tokenization

<details>
<summary>Keep it simple!</summary>

The simplest approach treats each character as a token.

**Advantages:**
- Extremely small vocabulary (typically <256 for ASCII, <150K for Unicode)
- No OOV problem—any text can be represented
- Works identically across all languages

**Disadvantages:**
- Very long sequences increase computational cost
- Loses semantic information—model must learn word boundaries and meanings from scratch
- Inefficient for languages with complex morphology

**Example Implementation:**
```python
def char_tokenize(text):
    return list(text)

text = "Hello, world!"
tokens = char_tokenize(text)
# ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']
```

</details>

#### Word-Level Tokenization

<details>
<summary>A word is a token</summary>

Treating words as tokens seems intuitive but presents challenges.

**Advantages:**
- Tokens have clear semantic meaning
- Shorter sequences than character-level
- Aligns with human linguistic intuition

**Disadvantages:**
- Massive vocabulary size (100K+ words for English)
- OOV problems with typos, neologisms, and rare words
- Difficult to handle morphologically rich languages
- Inconsistent handling of punctuation and special characters

**Example Implementation:**
```python
def word_tokenize(text):
    return text.split()

text = "Hello, world!"
tokens = word_tokenize(text)
# ['Hello,', 'world!']  # Note: punctuation attached
```

</details>


---

#### The Need for Subword Tokenization

Modern LLMs primarily use `subword` tokenization algorithms that strike a balance between the extremes of character and word-level approaches. These algorithms learn to break words into meaningful pieces (`morphemes`) that appear frequently in the training data.

[📋Tokenization Algorithms compared](https://www.arxiv.org/pdf/2509.15255)

<details>
<summary>Why not just use words or characters?</summary>

**Byte Pair Encoding (BPE)**

BPE, originally a data compression algorithm, has become one of the most popular tokenization methods for LLMs, used in models like GPT-2 and RoBERTa. 

**Algorithm:**

1. **Initialization**: Start with a vocabulary of individual characters
2. **Iteration**: 
   - Count all adjacent pairs of tokens in the corpus
   - Merge the most frequent pair into a new token
   - Add this new token to the vocabulary
   - Repeat until reaching desired vocabulary size

**Example:**
```
Corpus: "low", "low", "low", "lower", "newer", "wider"

Initial vocabulary: ['l', 'o', 'w', 'e', 'r', 'n', 'w', 'i', 'd']

After tokenization: ['l', 'o', 'w', '</w>', 'l', 'o', 'w', '</w>', ...]

Iteration 1: Most frequent pair is ('l', 'o')
Merge to create: 'lo'
New representation: ['lo', 'w', '</w>', 'lo', 'w', '</w>', ...]

Iteration 2: Most frequent pair is ('lo', 'w')
Merge to create: 'low'
New representation: ['low', '</w>', 'low', '</w>', ...]

Continue until vocabulary reaches target size...
```

**Advantages:**
- Purely data-driven—no linguistic assumptions
- Guaranteed to encode any text (falls back to characters)
- Frequent words remain as single tokens
- Rare words split into meaningful subwords

**Disadvantages:**
- Greedy algorithm may not find optimal tokenization, e.g., "unaffable" → `["un", "aff", "able"]` instead of `["un", "affable"]`
- Can create unintuitive splits for rare words
- Sensitive to corpus statistics, e.g., domain-specific terms, typos, etc.

**Implementation Pseudocode:**
```python
def train_bpe(corpus, num_merges):
    # Start with character-level vocabulary
    vocab = get_characters(corpus)
    splits = {word: list(word) for word in corpus}
    
    for i in range(num_merges):
        # Count all pairs
        pairs = count_pairs(splits)
        
        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        
        # Merge this pair throughout corpus
        splits = merge_pair(splits, best_pair)
        
        # Add to vocabulary
        vocab.append(''.join(best_pair))
    
    return vocab, splits
```
[📋Vocabulary sample](./tmp/bpe-tokenizer.json)

---

**WordPiece**

WordPiece, developed by Google and used in BERT and its variants, is similar to BPE but uses a different merge criterion.

**Key Difference from BPE:**
Instead of merging the most frequent pair, WordPiece merges the pair that **maximizes the likelihood of the training data**. It selects merges based on maximizing:

```
score(pair) = freq(pair) / (freq(first_token) * freq(second_token))
```

This likelihood-based approach tends to create tokens that better represent the statistical structure of language.

**Special Notation:**
WordPiece uses `##` to indicate subword tokens that don't start a word:
- "playing" → `["play", "##ing"]`
- "unaffable" → `["un", "##aff", "##able"]`

**Advantages:**
- Linguistically more principled than BPE
- Often creates more coherent subwords

**Disadvantages:**
- Computationally more expensive during training
- Similar sensitivity to corpus statistics as BPE

---

#### Unigram Language Model

The Unigram tokenization algorithm, used in models like T5 (Text-to-Text Transfer Transformer) and mBART (Multilingual BART), takes a fundamentally different approach from BPE and WordPiece.

**Algorithm:**

1. **Initialization**: Start with a large vocabulary (e.g., all characters plus all substrings up to a certain length)
2. **Iteration**:
   - Train a unigram language model on the current vocabulary
   - For each token, compute the loss increase if that token were removed
   - Remove tokens that increase loss least (contribute least to the model's likelihood), typically 10-20% per iteration
   - Repeat until reaching desired vocabulary size

**Key Concept:**
Unlike BPE (which builds up), Unigram starts with a large vocabulary and `prunes down`. It can find multiple possible segmentations and choose the most probable one using the Viterbi algorithm

**Example:**
For the word "unaffable", Unigram might consider:
- `["un", "aff", "able"]` with probability p1
- `["una", "ff", "able"]` with probability p2
- `["un", "affable"]` with probability p3

It selects the segmentation with highest probability.

**Advantages:**
- Considers multiple segmentation possibilities
- More theoretically grounded in probability theory
- Can better handle ambiguous segmentations

**Disadvantages:**
- More complex to implement
- Slower training process
- Requires more computational resources

---

#### SentencePiece

SentencePiece is not a tokenization algorithm per se, but rather a tokenization toolkit that implements BPE and Unigram in a language-agnostic way.

**Key Innovation:**
SentencePiece treats the input text as a raw stream of Unicode characters, including whitespace. 
Spaces are converted to a special character (▁) that indicates the beginning of a new word.

**Example:**

"Hello world" → `["▁Hello", "▁world"]`

"unaffable" → `["▁un", "aff", "able"]`

**Advantages:**
- Completely language-agnostic (no need for pre-tokenization, e.g., splitting on spaces)
- Reversible without information loss, since spaces are explicitly represented facilitating decoding back to original text
- Handles languages without spaces (Chinese, Japanese, Thai)
- Pre-compiled vocabulary for fast inference, no need for complex post-processing, 

</details>

---

#### Tokenization Techniques Recap

1. **Byte Pair Encoding (BPE)**: Merges the most frequent pairs of bytes or characters iteratively.
2. **WordPiece**: Similar to BPE but uses a different merge criterion based on maximizing the likelihood of the training data.
3. **Unigram Language Model**: Starts with a large vocabulary and prunes down based on token contributions to model likelihood.
4. **SentencePiece**: Treats input text as a raw stream of Unicode characters, including whitespace.

[🐍Tokenization Techniques Comparison](./tokenizer-comparison.ipynb)

[🐍BPE from scratch](./tokenizer-bpe.ipynb)

[🐍Unigram from scratch](./tokenizer-unigram.ipynb)

[📋Tokenization Algorithms compared](https://www.arxiv.org/pdf/2509.15255)

---

#### Tokenization in Production LLMs

<details>
<summary>Popular LLM tokenizers</summary>

##### GPT Models (GPT-2, GPT-3, GPT-4)

**Tokenization Method**: Byte-level BPE

OpenAI's GPT models use a byte-level variant of BPE that operates on UTF-8 bytes rather than Unicode characters. This approach:
- Guarantees any text can be represented with a fixed base vocabulary of 256 bytes
- Eliminates the need for special unknown tokens
- Handles emoji, special characters, and any Unicode naturally

**Vocabulary Size**:
- GPT-2: ~50,257 tokens
- GPT-3: ~50,257 tokens
- GPT-4: Uses a different tokenizer with improved efficiency

**Key Insight**: GPT-4's tokenizer is significantly more efficient, often requiring 30-40% fewer tokens for the same text compared to GPT-3, which means:
- Lower costs (pricing is per token)
- Longer effective context windows
- Faster processing

##### BERT and RoBERTa

**Tokenization Method**: WordPiece (BERT) and BPE (RoBERTa)

BERT introduced several special tokens:
- `[CLS]`: Classification token at start of sequence
- `[SEP]`: Separator between sentences
- `[MASK]`: Used during training for masked language modeling
- `[PAD]`: Padding token for batch processing

**Vocabulary Size**: ~30,000 tokens

##### LLaMA and LLaMA 2

**Tokenization Method**: SentencePiece with BPE

Meta's LLaMA models use SentencePiece, which provides:
- Better multilingual support
- More efficient tokenization for code
- Language-agnostic processing

**Vocabulary Size**: 32,000 tokens

#### Vocabulary Size Selection

Choosing the right vocabulary size involves trade-offs:

**Smaller Vocabularies (1K-10K tokens)**
- Pros: Simpler models, faster training, better generalization
- Cons: Longer sequences, more computation during inference

**Medium Vocabularies (30K-50K tokens)**
- Pros: Good balance for most languages
- Cons: Standard choice for many models

**Larger Vocabularies (100K+ tokens)**
- Pros: Very short sequences, efficient inference
- Cons: Larger embedding matrices, potential sparsity issues

The trend in modern LLMs is toward larger vocabularies (50K-150K+) as the benefits of shorter sequences outweigh the costs of larger embedding matrices.

</details>

<details>
<summary>Common special & Control Tokens</summary>

Most production tokenizers include special tokens:
- **`<BOS>`/`<EOS>`**: Beginning/end of sequence markers
- **`<PAD>`**: Padding for batch processing
- **`<UNK>`**: Unknown tokens (though modern systems avoid this)
- **`<SEP>`**: Separators for multi-part inputs
- **Control tokens**: For specific behaviors (e.g., `<CODE>`, `<PYTHON>`)

</details>

---

#### Tokenization and Model Performance

<details>
<summary>Why care about tokenization?</summary

**Impacts on Few-shot Learning:**
Tokenization affects how models perform on tasks with few examples. Subword tokenization helps models generalize from limited data by leveraging morphological structure.

**Arithmetic and Numbers:**
Standard tokenizers struggle with numbers. "1234" might be tokenized as `["12", "34"]` or `["1", "2", "3", "4"]`, making arithmetic challenging. Some modern approaches:
- Dedicated numeric encodings
- Character-level for numbers
- Specialized tokenizers for mathematical content

**Code Tokenization:**
Programming languages have different statistical properties than natural language:
- More consistent structure
- Importance of whitespace and indentation
- Special characters and operators
- Long identifiers

Models like Codex use tokenizers optimized for code, with:
- Better handling of camelCase and snake_case
- Preserved indentation
- Efficient encoding of common code patterns

</details>

#### Multilingual Considerations

<details>
<summary>Wait... 1 tokenizer, n languages?</summary>

Training tokenizers on multilingual data presents unique challenges:

**Vocabulary Allocation:**
In a fixed-size vocabulary, how much space should each language receive? Options include:
- Proportional to data volume (favors high-resource languages)
- Equal allocation (inefficient for high-resource languages)
- Optimized allocation based on language similarity

**Script Diversity:**
Different writing systems require different amounts of vocabulary:
- Alphabetic (Latin, Cyrillic): fewer tokens needed
- Logographic (Chinese, Japanese): more tokens needed
- Syllabic (Korean, Thai): intermediate requirements

**Shared Subwords:**
Languages from the same family often share morphemes (e.g., Romance languages), enabling better cross-lingual transfer.

**All languages are NOT created (tokenized) equal**

Language  |  Tokens per Word (approx.)| Baseline (English=1.0)
----------|---------------------------|-------------------------
English   |  1.3                      | 1.0
French    |  2.0                      | 1.5
German    |  2.1                      | 1.6
Spanish   |  2.1                      | 1.6
Chinese   |  2.5                      | 1.9
Hindi     |  3.0–5.0+                 | 2.3
[📋Language Model Tokenizers Introduce Unfairness Between Languages](https://openreview.net/pdf?id=Pj4YYuxTq9)

</details>

---

#### Performance Optimization

<details>
<summary>Caching and Batch Processing</summary>

**Caching:**
Tokenization can be cached since it's **deterministic**:
```python
from functools import lru_cache

@lru_cache(maxsize=10_000)
def tokenize_cached(text):
    return tokenizer.encode(text)
```

**Batch Processing:**
Process multiple texts together for efficiency:
```python
texts = ["Text 1", "Text 2", "Text 3"]
encoded = tokenizer.batch_encode_plus(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

</details>

#### Tokenization & LLMs implications

<details>
<summary>Why REALLY care about tokenization</summary>

1. **Input Representation**: The way text is tokenized affects how it's represented in the model. Different tokenization strategies can lead to different embeddings and, consequently, different model behaviors.

2. **Context Length**: Tokenization impacts the maximum context length of the model. Longer token sequences may lead to better understanding but also require more computational resources.

3. **Bias and Fairness**: Tokenization can introduce biases, especially if certain languages or dialects are underrepresented in the training data.

4. **Finally...**: [🐍LLMs behavior affected by tokenization](./tokenizer-limits.ipynb)

</details>

#### Token management

<details>
<summary>Keeping track of tokens</summary>

- token counting (rules of thumb for English):
    . 1 token ≈ 4 characters

    . 1 token ≈ ¾ of a word

    . 100 tokens ≈ 75 words

    . 1–2 sentences ≈ 30 tokens

    . 1 paragraph ≈ 100 tokens

    . ~1,500 words ≈ 2,048 tokens

    - tools:

        [openai tokenizer](https://platform.openai.com/tokenizer)

        [tiktokenizer](https://tiktokenizer.vercel.app)
- token usage
    . `Input` tokens – tokens in your request.

    . `Output` tokens – tokens generated in the response.

    . `Cached` tokens – reused tokens in conversation history (often billed at a reduced rate).

    . `Reasoning` tokens – in some advanced models, extra “thinking steps” are included internally before producing the final output.
    
    [🐍Counting tokens](./tokenizer-count.ipynb)
- token limits (input + output)
    [🔗openai models](https://platform.openai.com/docs/models/compare)
- cost management
    [🔗openai pricing](https://platform.openai.com/docs/pricing)

</details>

---

### Embeddings

**✨ ..next => [Embeddings](./embeddings.md)! ✨**


