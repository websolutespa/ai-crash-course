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

<details>
<summary>The semantic meaning of data</summary>

#### What are embeddings?
Embeddings are numerical representations of data that capture semantic meaning. In the context of language models, embeddings convert words, phrases, or entire documents into `dense vectors` in a `high-dimensional space`. These vectors are designed such that semantically similar items are located close to each other in this space.

- **dense vectors**: Unlike sparse representations (like one-hot encoding, e.g. in text classification), embeddings use continuous values, allowing them to capture nuanced relationships. E.g. the word "king" might be represented as a 768-dimensional vector like [0.25, -0.1, 0.4, ..., 0.05]. Each value contributes to the overall meaning.
- **high-dimensional space**: Embeddings typically have hundreds or thousands of dimensions, enabling them to represent complex semantic relationships. Dimensions can be thought of as latent `features` that capture various aspects of meaning, such as syntactic roles, semantic categories, or contextual nuances.
- **how many vectors?** Each word, phrase, or document gets its own vector representation. A vector can represent anything from a single word to an entire paragraph, depending on the embedding model used, even images or other data types. The same vector can store a single word like "cat" or a whole sentence like "The cat sat on the mat.", or the content of a research paper, but not an encyclopedia (-> Context Window).
- **distance metrics**: The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness. Measures like `cosine similarity` or `Euclidean distance` are commonly used to quantify similarity between embeddings. For example, the cosine similarity between the embeddings of "king" and "queen" would be high, indicating they are semantically related. More similarity = closer in the vector space.

#### Why are embeddings important?
Embeddings are crucial for several reasons:
- **Semantic Understanding**: They enable models to understand the meaning and context of words beyond their surface form.
- **Dimensionality Reduction**: They reduce the complexity of data while preserving important relationships. E.g. the words "cat" and "dog" might be closer in embedding space than "cat" and "car".
- **Facilitating Downstream Tasks**: They serve as input features for various machine learning tasks, such as classification, clustering, and recommendation systems.

#### How are embeddings created?
Embeddings are typically created using neural networks trained on large datasets. Common methods include:
- **Word2Vec**: A predictive model that learns word associations from large corpora.
- **GloVe**: A count-based model that captures global word-word co-occurrence statistics.
- **Transformers**: Models like BERT and GPT use attention mechanisms to create contextual embeddings.

#### Key concepts
1. **Context Window**
The context window is the maximum amount of text the model can handle at one time. For example, if a model has a context window of 512 tokens, it can only read 512 words or pieces of words at once. Longer texts must be split up. 
A larger context window lets you process bigger documents without losing information. This is helpful for tasks like searching long articles or research papers, or reports.

Typical context windows are:
- Small models: 512–1024 tokens
- Medium models: 2048–4096 tokens
- Large models: 8192–32768 tokens or more
2. **Tokenization Unit**
Tokenization is how the model breaks text into smaller parts (`tokens`). 
The way a model tokenizes text affects how well it understands different words, especially unusual ones. Most modern models use subword tokenization for better flexibility. See previous section on Tokenization for more details.

3. **Dimensionality**
Dimensionality is the size of the number list (vector) the model creates for each piece of text. For example, some models produce vectors with 768 numbers, others with 1024 or even 3072.
Higher-dimensional vectors can store more detailed information, but they need more computer power. Lower-dimensional vectors are faster but might miss some details.

4. **Vocabulary Size**
This is the number of unique tokens the model knows. Bigger vocabularies handle more words and languages but use more memory. Smaller vocabularies are quicker, but might not understand rare or specialized words.
Vocabulary sizes typically range between 30k-50k, tokens, with some models going up to 150k+.

5. **Training Data**
Training data is what the model learned from.

Models trained on General-Purpose: When trained on many types of text, like web pages or books the models become useful for general purposes. Good for broad tasks.
Domain-Specific: When trained on specialized texts, like medical or legal documents, the models become specialized in a specific domain. These are good for niche tasks (but it may not work as well for general tasks.)


[🔗Iris, MNIST as Embeddings](https://projector.tensorflow.org/)

</details>

#### Tokenizer vs Embedding

<details>
<summary>The tokenizer converts text into tokens (numbers) that the model can understand, while the embedding maps those tokens into dense vectors that capture their semantic meaning.</summary>

#### Core Definitions

##### Tokenizer
A **tokenizer** is a text preprocessing component that:
- Breaks raw text into discrete units (tokens)
- Maps tokens to unique integer IDs (vocabulary indices)
- Provides **bidirectional** mapping: text ↔ token IDs

##### Embedding
An **embedding** (or embedding layer) is a learned representation component that:
- Maps discrete token IDs to continuous dense vectors
- Captures semantic and syntactic relationships between tokens
- Provides **unidirectional** mapping: token IDs → dense vectors

---

##### Key Differences

| Aspect | Tokenizer | Embedding |
|--------|-----------|-----------|
| **Primary Function** | Text preprocessing & segmentation | Semantic representation learning |
| **Input** | Raw text strings | Token IDs (integers) |
| **Output** | Token IDs (integers) | Dense vectors (continuous values) |
| **Reversibility** | ✅ Fully reversible (encode/decode) | ❌ Not reversible (one-way transformation) |
| **Training** | Rule-based or statistical (BPE, WordPiece) | Learned via backpropagation with model |
| **Parameters** | No trainable parameters (uses vocabulary mapping) | Trainable weight matrix (vocab_size × embedding_dim) |
| **Scope** | Language/script-specific | Semantic space |
| **Deterministic** | ✅ Yes (same input → same output) | ✅ Yes after training (but learned, not rule-based) |

---

#### Tokenizer Architecture

```
Text Input: "Hello world"
      ↓
[Preprocessing: lowercasing, normalization]
      ↓
[Segmentation Algorithm: BPE/WordPiece/Unigram]
      ↓
Tokens: ["Hello", "world"]
      ↓
[Vocabulary Lookup]
      ↓
Token IDs: [5678, 1234]
```

**Key Components:**
- **Vocabulary**: Fixed dictionary mapping tokens ↔ IDs (e.g., 50k-100k entries)
- **Segmentation Algorithm**: BPE, WordPiece, SentencePiece, etc.
- **Special Tokens**: [PAD], [UNK], [CLS], [SEP], [MASK]
- **No learnable parameters** (vocabulary is fixed after creation)

#### Embedding Architecture

```
Token IDs: [5678, 1234]
      ↓
[Embedding Matrix Lookup: E[vocab_size, embedding_dim]]
      ↓
Vector 5678: [0.23, -0.45, 0.67, ..., 0.12]  (dim=768)
Vector 1234: [-0.34, 0.56, -0.23, ..., 0.89]  (dim=768)
      ↓
Dense Representations (continuous vectors)
```

**Key Components:**
- **Embedding Matrix**: Shape `[vocab_size, embedding_dim]`
  - Example: `[50000, 768]` = 38.4M parameters
- **Learned through backpropagation** during model training
- Acts as a lookup table: `embedding_vector = E[token_id]`

---

#### Tokenizer Training

**When:** Before model training (preprocessing step)

**How:**
1. Collect large corpus of text
2. Apply algorithm (e.g., Byte Pair Encoding):
   - Start with character-level tokens
   - Iteratively merge most frequent pairs
   - Build vocabulary of subword units
3. Result: Fixed vocabulary + merging rules

**Key Point:** Tokenizer is **not** trained with gradient descent. It uses statistical/rule-based methods.

**Example Algorithms:**
- **BPE (Byte Pair Encoding)**: Merges frequent character pairs
- **WordPiece**: Similar to BPE, used by BERT
- **SentencePiece**: Treats text as raw Unicode, used by T5
- **Unigram**: Probabilistic subword segmentation

#### Embedding Training

**When:** During model training (integral part)

**How:**
1. Initialize embedding matrix randomly (or with pre-trained values)
2. During forward pass:
   - Look up vectors for input token IDs
   - Pass vectors through model
3. During backward pass:
   - Compute gradients for embedding weights
   - Update via optimizer (Adam, SGD, etc.)
4. Embeddings learn to position semantically similar tokens close together

**Key Point:** Embeddings are **learned parameters** optimized for the model's task.

---

#### Why encode() and decode() vs. Just encode()

#### Tokenizer: Bidirectional Mapping

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# encode(): text → token IDs
text = "Hello world"
token_ids = tokenizer.encode(text)  # [101, 7592, 2088, 102]

# decode(): token IDs → text
reconstructed = tokenizer.decode(token_ids)  # "hello world"
```

**Why bidirectional?**
- Tokenization is a **lossless** transformation (with minor normalization)
- Essential for generation tasks (need to convert model output back to text)
- Vocabulary is a bijective mapping (one-to-one correspondence)
- Original information is preserved in token IDs

#### Embedding: Unidirectional Mapping

```python
embedding_layer = nn.Embedding(vocab_size=50000, embedding_dim=768)

# Only forward direction exists
token_ids = torch.tensor([5678, 1234])
vectors = embedding_layer(token_ids)  # Shape: [2, 768]

# No decode() or inverse operation!
# Cannot reliably recover token_ids from vectors
```

**Why only encode?**

1. **Information Loss**: Embedding is a **lossy** transformation
   - Maps discrete IDs to continuous space
   - Multiple similar tokens may have nearby vectors
   - Exact reconstruction is impossible

2. **Dimensionality**: Many-to-one mapping potential
   - Vocab size: ~50k discrete tokens
   - Embedding space: continuous 768-dim space
   - No unique inverse function exists

3. **Purpose**: Embeddings are designed for **semantic similarity**, not reconstruction
   - Goal: Place "cat" and "kitten" close together
   - Goal: Make "king" - "man" + "woman" ≈ "queen"
   - Reconstruction would undermine these properties

4. **Mathematical Nature**: 
   - Tokenizer: Dictionary lookup (discrete → discrete)
   - Embedding: Linear projection (discrete → continuous)
   - Going backward would require solving: "which token ID gives this vector?" (no unique answer)

---

#### Relationship & Execution Flow

#### Sequential Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     TOKENIZER STAGE                         │
│  Raw Text → [Tokenization] → Token IDs                      │
│  "The cat sat" → [encode()] → [101, 2182, 4937, 102]        │
└────────────────────┬────────────────────────────────────────┘
                     │ Token IDs passed to model
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    EMBEDDING STAGE                          │
│  Token IDs → [Lookup Table] → Dense Vectors                 │
│  [101, 2182, 4937, 102] → [[0.1,...], [0.3,...], ...]       │
└────────────────────┬────────────────────────────────────────┘
                     │ Vectors passed to model layers
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  TRANSFORMER LAYERS                         │
│  Dense Vectors → [Attention, FFN, ...] → Output Vectors     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
              [Task-specific head]
```

#### Example: Text Classification

```python
# 1. TOKENIZER: Text → Token IDs
text = "This movie is amazing!"
token_ids = tokenizer.encode(text)  
# → [101, 2023, 3185, 2003, 6429, 999, 102]

# 2. EMBEDDING: Token IDs → Dense Vectors
input_embeds = embedding_layer(token_ids)
# Shape: [7, 768] - 7 tokens, each a 768-dim vector

# 3. MODEL PROCESSING
hidden_states = transformer_layers(input_embeds)

# 4. OUTPUT
classification = classifier_head(hidden_states)  # "Positive"

# Note: tokenizer.decode() only used if generating text
# For classification, we never decode back to text
```

---

#### Truncation & Padding

#### Tokenizer-Level Operations

```python
# Tokenizers handle sequence length constraints
tokens = tokenizer.encode(
    long_text,
    max_length=512,      # Truncate to 512 tokens
    truncation=True,     # Enable truncation
    padding='max_length' # Pad short sequences
)
```

**Truncation**: Remove tokens beyond max_length
**Padding**: Add [PAD] tokens to reach max_length

#### Embedding-Level Considerations

```python
# Embeddings must handle special tokens
embedding_layer = nn.Embedding(
    vocab_size=50000,
    embedding_dim=768,
    padding_idx=0  # [PAD] token ID, not updated during training
)
```

**Key Point**: Padding tokens get embedded but are masked in attention layers to prevent them from affecting computations.

---

#### Parameter Count

#### Tokenizer: 0 Trainable Parameters

```python
# Tokenizer configuration (not parameters):
{
    "vocab_size": 50000,        # Dictionary size
    "model_max_length": 512,    # Max sequence length
    "special_tokens": {...}     # Special token definitions
}
```

#### Embedding Layer: vocab_size × embedding_dim Parameters

```python
# BERT-base embedding parameters:
vocab_size = 30522
embedding_dim = 768
parameters = 30522 × 768 = 23,440,896 parameters

# Plus positional and token type embeddings:
# - Position embeddings: 512 × 768 = 393,216
# - Token type embeddings: 2 × 768 = 1,536
# Total: ~23.8M parameters just for embeddings!
```

---

#### Tokenizer Scope & Capabilities

- **Language-dependent**: Different tokenizers for different languages
- **Script-aware**: Handles Unicode, special characters, whitespace
- **No semantic understanding**: "bank" (river) and "bank" (financial) → same token
- **Vocabulary coverage**: Out-of-vocabulary (OOV) handling via subwords or [UNK]

#### Embedding Scope & Capabilities

- **Semantic representation**: Captures meaning, context, relationships
- **Language-agnostic** (after tokenization): Same architecture works for any language
- **Contextual** (in modern models): Same token has different embeddings based on context
  - Static embeddings (Word2Vec): "bank" always same vector
  - Contextual embeddings (BERT): "bank" vector changes based on surrounding words

---

#### ❌ Misconception 1: "Embeddings are part of the tokenizer"
**Reality**: Tokenizers and embeddings are separate components. Tokenizer output (token IDs) is the input to embeddings.

#### ❌ Misconception 2: "Tokenizer learns during model training"
**Reality**: Tokenizer is fixed after creation. Only embeddings and model weights are trained.

#### ❌ Misconception 3: "We can decode embeddings back to text"
**Reality**: Embeddings → token IDs requires additional model components (like LM head), and even then, we decode token IDs → text using the tokenizer.

#### ❌ Misconception 4: "Larger vocabulary = better model"
**Reality**: Vocabulary size is a trade-off:
- **Larger vocab**: Fewer tokens per sequence, more embedding parameters
- **Smaller vocab**: More tokens per sequence, fewer OOV issues, less parameters

---

#### Practical Implications

#### For Model Development

1. **Tokenizer changes require retraining**: If you modify the tokenizer, you must retrain the entire model (embeddings will be misaligned)

2. **Vocabulary size affects model size**: 
   - Doubling vocab_size doubles embedding parameters
   - For BERT-base: ~23M of 110M parameters (21%) are in embeddings

3. **Multilingual models**: Use shared tokenizer (e.g., SentencePiece) across languages, but embeddings learn language-specific patterns

#### For Inference

1. **Tokenizer must match training**: Always use the same tokenizer the model was trained with

2. **Sequence length**: Tokenizer handles truncation; model processes up to max_length

3. **Generation tasks**: 
   - Tokenizer: text → IDs (input)
   - Model: IDs → IDs (processing)
   - Tokenizer: IDs → text (output)

---

#### Summary 

| Question | Tokenizer | Embedding |
|----------|-----------|-----------|
| **What does it do?** | Converts text to/from integer IDs | Converts integer IDs to dense vectors |
| **Is it reversible?** | Yes (encode + decode) | No (only forward) |
| **Does it learn?** | No (rule/statistics-based) | Yes (gradient descent) |
| **Has parameters?** | No | Yes (vocab_size × embedding_dim) |
| **When is it created?** | Before model training | Initialized before, trained during |
| **Can it change meaning?** | No (mechanical mapping) | Yes (learns semantic relationships) |
| **Output type** | Discrete (integers) | Continuous (floats) |
| **Typical size** | Vocabulary: 30k-150k entries | Matrix: 30k-150k × 768-4096 |

</details>

---

#### Context size implications

<details>
<summary>How embeddings can store a token, a sentence, a book or more?</summary>

Embeddings can represent varying lengths of text by adjusting the way text is processed and encoded into vectors. Here's how embeddings can handle different levels of text granularity:
1. **Token-Level Embeddings**:
   - Each individual token (word or subword) is mapped to its own embedding vector.
   - Example: The word "cat" might be represented as a 768-dimensional vector.
    - Use Case: Useful for tasks like part-of-speech tagging or named entity recognition.
2. **Sentence-Level Embeddings**:
   - Sentences are encoded into single vectors that capture the overall meaning.
   - Techniques: Models like Sentence-BERT or Universal Sentence Encoder generate sentence embeddings by averaging or pooling token embeddings, or using specialized architectures.
   - Example: The sentence "The cat sat on the mat." might be represented as a single 768-dimensional vector.
    - Use Case: Ideal for tasks like semantic search or sentence similarity.
3. **Paragraph-Level Embeddings**:
   - Paragraphs are encoded into vectors that capture the broader context.
   - Techniques: Similar to sentence embeddings but may involve hierarchical models or attention mechanisms to capture inter-sentence relationships.
   - Example: A paragraph discussing cats might be represented as a single vector.
    - Use Case: Useful for document classification or topic modeling.
4. **Document-Level Embeddings**:
   - Entire documents are encoded into vectors that summarize the content.
   - Techniques: Models like Doc2Vec or transformer-based models can generate document embeddings by aggregating sentence or paragraph embeddings.
   - Example: A research paper on feline behavior might be represented as a single vector.
    - Use Case: Suitable for tasks like document retrieval or clustering.
5. **Multi-Document or Corpus-Level Embeddings**:
   - Large collections of documents can be represented in a shared embedding space.
   - Techniques: Techniques like topic modeling or clustering can be applied to group similar documents based on their embeddings.
   - Example: A corpus of articles about cats might be analyzed to find common themes.
    - Use Case: Useful for large-scale information retrieval or recommendation systems.    

[🐍Handle large documents](embeddings-capacity.ipynb)        
</details>

#### Common task types (also used in benchmarks like MTEB)

<details>
<summary>Embedding tasks</summary>

- Text Classification: Assigning predefined categories to text based on its content.
- Clustering: Grouping similar texts together based on their content.
- Semantic Textual Similarity: Measuring how similar two pieces of text are in meaning.
- Bitext Mining: Finding parallel sentences in different languages.
- Reranking: Ordering a list of texts based on relevance to a query.
- Pair Classification: Determining if two texts are related or not.
- Multilabel Classification: Assigning multiple categories to a single piece of text.
- Instruction Reranking: Ranking texts based on how well they follow given instructions.

[🐍Common tasks](./embeddings-tasks.ipynb)
</details>

---

#### Real use case

- classifier [🐍Qwen + logistic regression](../03/qwen-logistic.ipynb)

- recommendation system [🐍Qwen movie suggester](./embeddings-use-cases.ipynb)

---

#### Embedding Api

[🐍OpenAI Embedding API](embeddings-api.ipynb)

[🔗OpenAI pricing for embeddings](https://openai.com/api/pricing/)


#### Benchmarking

Massive Text Embedding Benchmark (MTEB)
MTEB is a community-run leaderboard. It compares over 100 text and image embedding models across more than 1,000 languages. Everything is in one place — evaluation metrics, different types of tasks, and a wide range of domains. This makes it a useful starting point when you’re deciding which model to use. 

[🔗MTEB](https://huggingface.co/spaces/mteb/leaderboard)

