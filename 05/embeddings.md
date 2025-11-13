## From Text to Meaning: Tokenization & Embeddings

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
