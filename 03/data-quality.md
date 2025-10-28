# Data Cleaning & Deduplication Techniques for LLM Pre-training

Pre-training data quality is critical - the saying **"garbage in, garbage out"** strongly applies. Here are the main techniques used to clean and deduplicate large internet-scale corpora:

---

## 1. Deduplication Techniques

### A. Exact Deduplication

Removes identical documents or segments.

**Methods:**
- **Hash-based matching**: Generate hash (MD5, SHA-256) for each document
  - If two documents have the same hash → duplicate
- **Line-level deduplication**: Remove duplicate lines within documents

**Example:**
```
Document A: "The quick brown fox jumps over the lazy dog."
Document B: "The quick brown fox jumps over the lazy dog."
Hash(A) = Hash(B) → Remove one copy
```

**Tools**: Bloom filters or hash tables for efficient checking at scale

---

### B. Near-Deduplication (Fuzzy Matching)

Removes *similar* but not identical documents (more challenging at scale).

#### MinHash & LSH (Locality-Sensitive Hashing)

Most popular approach for web-scale deduplication.

**How it works:**

1. **Shingling**: Break text into n-grams (e.g., 5-word sequences)
2. **MinHash**: Create compact "fingerprint" representing document similarity
3. **LSH**: Group similar documents into buckets efficiently
4. Compare only documents in same bucket

**Example:**

```text
Doc A: "Climate change affects global temperatures significantly"
Doc B: "Climate change impacts worldwide temperatures greatly"

→ 80% similar n-grams → Flag as near-duplicate
```

**Used by**: CommonCrawl processing, C4 dataset, many LLM projects

#### Suffix Arrays & Edit Distance

- Build suffix arrays for substring matching
- Calculate Levenshtein distance for similarity
- More accurate but computationally expensive

---

### C. Repetitive Content Within Documents

Remove documents with excessive internal repetition.

**Techniques:**

- **N-gram repetition ratio**: Flag documents where same phrases repeat too often
- **Line repetition detection**: Remove documents with repeated lines
- **Compression ratio heuristic**: Highly compressible text often indicates repetition

**Example of bad content:**

```text
"Buy now! Buy now! Buy now! Click here! Click here!..." 
→ Flagged as spam/low-quality due to repetition
```

---

## 2. Quality Filtering Techniques

### A. Language Identification

Filter for target language(s) only.

**Tools:**

- **fastText** language classifier (by Meta)
- **langdetect** library
- **CLD2/CLD3** (Compact Language Detector)

**Threshold**: Keep documents with >90% confidence in target language

---

### B. Heuristic-Based Filtering

#### Text Statistics

- **Average word length**: Too short/long suggests non-natural text
- **Word count**: Filter very short documents (<50 words)
- **Special character ratio**: High ratio suggests code, logs, or garbage
- **Uppercase ratio**: Too much suggests spam/shouting
- **Digit ratio**: High percentage might indicate tables, lists, or data dumps
- **Punctuation ratio**: Extreme values suggest formatting issues

**Example thresholds:**

```python
# Pseudo-code for filtering
if doc.avg_word_length < 3 or doc.avg_word_length > 15:
    reject()
if doc.special_char_ratio > 0.3:
    reject()
if doc.uppercase_ratio > 0.5:
    reject()
```

#### Document Structure

- **Line count limits**: Very few lines might be low-quality
- **Paragraph structure**: Presence of paragraph breaks indicates structure
- **Sentence boundaries**: Proper punctuation and sentence structure

---

### C. Content-Based Filtering

#### Blocklist/Denylist Filtering

Remove documents containing:

- Profanity and offensive language (often using word lists)
- Personally identifiable information (PII): emails, phone numbers, addresses
- URLs to known low-quality or harmful sites
- Spam keywords ("Click here!", "Make money fast!")

#### Boilerplate Removal

Remove common website elements:

- Navigation menus
- Cookie notices
- "Copyright 2024..."
- Advertisements
- Social media share buttons
- Comment sections (often low-quality)

**Tools:**

- jusText algorithm
- Trafilatura library
- Custom regex patterns

**Example:**

```html
Before: "Home | About | Contact | [Article content] |
         Subscribe to newsletter! | Copyright 2024"
After: "[Article content]"
```

---

### D. Classifier-Based Quality Filtering

Train a classifier to predict document quality.

**Approach 1: Using High-Quality Reference Data**

- Train classifier on Wikipedia, books, high-quality sources (positive examples)
- Train on spam, bot-generated text (negative examples)
- Score all web documents and keep top percentiles

**Approach 2: Perplexity Filtering**

- Use a language model to calculate perplexity
- High perplexity = unusual/low-quality text
- Keep documents with perplexity in reasonable range

**Example (GPT-3 approach):**

- Trained classifier on curated positive examples
- Filtered CommonCrawl to keep only documents similar to high-quality sources

---

### E. Toxicity & Safety Filtering

**Techniques:**

- **Perspective API**: Google's toxicity detection
- **Custom classifiers**: Trained on hate speech, violence, explicit content
- **Regex patterns**: Detect specific harmful patterns

**Trade-offs:**

- Too aggressive → lose valuable data
- Too lenient → include harmful content
- Often use multiple passes with different thresholds

---

## 3. Format-Specific Cleaning

### HTML Processing

- Remove HTML tags and JavaScript
- Extract main content (article text)
- Parse structured data (JSON-LD, microdata)

**Tools**: BeautifulSoup, lxml, html2text, Trafilatura

### PDF Processing

- Extract text while preserving structure
- Handle multi-column layouts
- Remove headers/footers/page numbers

### Code Filtering

- Identify and optionally separate code blocks
- Remove auto-generated code
- Filter out minified JavaScript

---

## 4. Privacy & Legal Compliance

### PII Removal

Patterns to detect and remove/mask:

- **Email addresses**: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`
- **Phone numbers**: various international formats
- **Social Security Numbers**
- **Credit card numbers**
- **IP addresses**

### Copyright Filtering

- Remove content with explicit copyright notices (context-dependent)
- Filter books/articles from certain sources
- Remove paywalled content
- Handle robots.txt and crawling ethics

---

## 5. Real-World Pipeline Example

Here's how datasets like **C4 (Colossal Clean Crawled Corpus)** process CommonCrawl:

```text
1. Download CommonCrawl (petabytes of web data)
   ↓
2. Language Filtering (keep English only)
   - fastText classifier
   - 99% confidence threshold
   ↓
3. Quality Filtering
   - Remove lines without ending punctuation
   - Remove pages with <3 sentences
   - Remove pages with bad word ratio > threshold
   - Remove "lorem ipsum" placeholder text
   ↓
4. Deduplication
   - MinHash + LSH for near-duplicates
   - 3-sentence window matching
   ↓
5. Final Corpus (~750GB from 20TB)
```

---

## 6. Advanced Techniques

### URL-Based Filtering

- Domain reputation scores
- URL structure analysis (too many parameters = low quality)
- Date in URL (helps with temporal filtering)

### Metadata Signals

- HTTP headers (content-type verification)
- Crawl metadata (fetch time, status codes)
- WARC record analysis

### Multi-Pass Filtering

- First pass: Aggressive filtering (remove obvious junk)
- Second pass: Moderate filtering (quality scoring)
- Third pass: Fine-grained deduplication

---

## 7. Scaling Challenges

When processing **trillion-token** datasets:

- **Distributed computing**: Use Spark, Ray, Dask
- **Efficient algorithms**: MinHash O(n) vs all-pairs O(n²)
- **Approximate methods**: Trade accuracy for speed
- **Checkpointing**: Save intermediate results
- **Monitoring**: Track filtering statistics

**Example statistics** (typical web crawl):

```text
Original:     100% (20TB)
After lang:   40% (8TB)  - kept English only
After quality: 20% (4TB)  - removed low-quality
After dedup:   10% (2TB)  - removed duplicates
```

---

## Key Takeaways

- **Deduplication is crucial** - prevents model from memorizing repeated content
- **Quality over quantity** - 1TB of clean data > 10TB of dirty data
- **Multiple filtering stages** - each removes different types of noise
- **Balance is important** - over-filtering loses valuable diversity
- **Domain-specific needs** - code LLMs need different cleaning than chat LLMs

The goal is to create a corpus that represents diverse, high-quality human knowledge and communication while removing noise that would degrade model performance.
