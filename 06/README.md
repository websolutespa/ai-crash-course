## Transformers

> **Why is language processing challenging?** 

    Because the `meaning` of a word often depends on the `context` provided by surrounding words. 
    Traditional models struggled with this because they processed text sequentially, making it hard to capture long-range dependencies.
    E.g., in the sentence "The bank will not approve your loan because your credit score is low," the word "bank" refers to a financial institution, not the side of a river. Understanding this requires considering the entire sentence.

<details>
<summary>NLP and LLMs</summary>

- **NLP (Natural Language Processing)** is the broader field focused on enabling computers to understand, interpret, and generate human language. NLP encompasses many techniques and tasks such as sentiment analysis, named entity recognition, and machine translation.
- **LLMs (Large Language Models)** are a powerful subset of NLP models characterized by their massive size, extensive training data, and ability to perform a wide range of language tasks with minimal task-specific training.
</details>

<details>
<summary>NLP Tasks 🚀</summary>

- **Text Classification**: Assigning categories to text (e.g., spam detection, sentiment analysis).
- **Zero-Shot Classification**: Classifying text into categories without prior training on those specific categories.
- **Feature Extraction**: Extracting meaningful vector representations from text => embeddings.
- **Text Generation**: Producing coherent and contextually relevant text (e.g., chat bots, story generation).
- **Sentiment Analysis**: Determining the sentiment expressed in text (e.g., positive or negative).
- **Named Entity Recognition (NER)**: Identifying and classifying entities in text (e.g., names, dates, locations).
- **Part-of-Speech Tagging**: Assigning grammatical tags to words (e.g., noun, verb, adjective).
- **Machine Translation**: Translating text from one language to another (e.g., English to French).
- **Text Summarization**: Generating concise summaries of longer texts.
- **Question Answering**: Providing answers to questions based on a given context or document.

[🐍NLP tasks ](nlp-tasks.ipynb)
</details>  

--- 

### Why Transformers?

<details>
<summary>Previous Approaches</summary>

**RNN**s (Recurrent Neural Networks) and **LSTM**s (Long Short-Term Memory networks) were early attempts to address this by maintaining a form of memory. However, they still faced limitations in handling long sequences and parallel processing.<br/>
[🐍 RNN sentiment analysis](rnn.ipynb)

**CNN**s (Convolutional Neural Networks) were also used for text processing, but they primarily focused on local patterns and struggled with capturing global context.

**Seq2Seq** models improved upon these by using an encoder-decoder architecture, but they still relied on RNNs or LSTMs, which limited their efficiency and effectiveness.
</details>

<details>
<summary>Transformers vs RNNs/CNNs/Seq2Seq</summary>

Transformers are a type of deep learning model architecture introduced in 2017 that revolutionized natural language processing and other AI tasks. They're based on the `attention mechanism` which allows the model to dynamically focus on different parts of the input data when making predictions.

Key features:
- **Parallel processing**: Unlike recurrent models, Transformers can process all input elements (tokens) simultaneously
- **Self-attention**: Each position in the input can attend to all other positions
- **Multi-head attention**: Multiple attention "heads" work together to capture different types of relationships.

  Each “head” pays attention to different aspects of relationships between words, thinks of it like having multiple experts analyzing the same sentence from different angles:

  Head 1 might focus on subject-verb relationships

  Head 2 might track pronouns and their referents

  Head 3 might be all about adjective-noun pairs

  ...

  And then heads are pooling their findings together for a comprehensive understanding
- **Attention mechanism**: Enables the model to weigh the importance of different input elements when generating output.  
- **Long.Range dependencies**: Thanks to attention, transformers can connect words no matter how far apart they are in a sentence.

Transformers have become the foundation for many state-of-the-art (SOTA) models like BERT, GPT, and T5, and are now widely used not just for text processing but also for image, audio, and other types of data.

[🧠 Attention is all you need](https://arxiv.org/abs/1706.03762)

</details>

---

### Transformer Architecture

<details>
<summary>Overview</summary>

[🐍overview](transformer-overview.ipynb)
</details>

<details>
<summary>Attention is all you need</summary>

[🐍attention](transformer-attention.ipynb)
</details>

<details>
<summary>Architecture variants</summary>

- **Encoder-Only Models**: Focus on understanding and processing input text (e.g., BERT).
- **Decoder-Only Models**: Specialize in generating text based on input prompts (e.g., GPT series).
- **Encoder-Decoder Models**: Combine both encoding and decoding capabilities for tasks like translation or summarization (e.g., T5, BART).

[🐍variants](transformer-variants.ipynb)

</details>

<details>
<summary>Architecture improvements</summary>

- **Local/sparse attention**: Reduces computational complexity by limiting attention to nearby tokens, significantly improving efficiency for long sequences at cost of some context awareness.
- **Multi-Query Attention (MQA)**: MQA keeps *one* set of keys/values for **all** attention heads, while still letting each head have its own query matrix.  
- **Grouped-Query Attention (GQA)**: Reduces computational load by sharing key and value matrices among groups of attention heads.
  In GQA, every *group* shares a single set of keys/values, but each group still has its own queries.  
- **Flash Attention**: provides
significant speedups for both training and inference of Transformer LLMs
on GPUs. 
- **Mixture of Experts (MoE)**: Introduces specialized sub-networks (experts) that are activated based on the input, allowing the model to scale efficiently.

  [🐍 Mixture of Experts](../04/moe.ipynb)
- **Positional Embeddings**: Techniques like Rotary Positional Embeddings (RoPE) enhance the model's ability to understand token positions.

[🐍improvements](transformer-improvements.ipynb)

> Making generation less sequential is another research goals of ours.

</details>

<details>
<summary>Summary</summary>

#### Key Concepts:

- A Transformer LLM generates `one token at a time`.

- That output token is `appended to the prompt`, then this updated prompt is presented to the model again for another forward pass to
generate the next token.

- The three major components of the Transformer LLM are the
`tokenizer`, a stack of `Transformer blocks`, and a language modeling `head`.

- The tokenizer contains the `token vocabulary` for the model. The
model has `token embeddings` associated with those tokens.
Breaking the text into tokens and then using the embeddings of
these tokens is the first step in the token generation process.

- The `forward pass` flows through all the stages once, `one by one`.

- Near the end of the process, the LM head scores the `probabilities of the next possible token`. Decoding strategies inform which actual
token to pick as the output for this generation step (sometimes it’s the most probable next token, but not always).

  <details>
  <summary>LLMs are Randomized Algorithms? Connection with a 50-years old academic field</summary>

  [🧠 Randomized Algorithms](https://medium.com/areas-producers/llms-are-randomized-algorithms-c41e2eddedf4)
  </details>

- One reason the Transformer excels is its ability to `process tokens in parallel`. Each of the input tokens flow into their `individual tracks` or streams of processing. The number of streams is the model’s **context size** and this represents the max number of tokens the model can operate on.

- Because Transformer LLMs loop to generate the text one token at a time, it’s a good idea to `cache` the processing results of each step so we don’t duplicate the processing effort (these results are stored as
various matrices within the layers).

- The majority of processing happens within `Transformer blocks`.
These are made up of two components: 
  - `feedforward neural network`, which is able to store information and make predictions and interpolations from data it was trained on.
  - `attention layer`, that incorporates contextual information to allow the model to better capture the nuance of language.

- Attention happens in two major steps: 
  - scoring relevance 
  - combining information

- A Transformer attention layer conducts several attention operations
in parallel, each occurring inside an `attention head`, and their
outputs are aggregated to make up the output of the attention layer.

- Attention can be accelerated via sharing the keys and values
matrices between all heads, or groups of heads (`grouped-query attention`).

- `Flash Attention` speed up the attention calculation by optimizing how the operation is done on the different memory systems of a GPU.

#### Components:

- **Input Embedding**: Converts input tokens into dense vectors.
- **Positional Encoding**: Adds information about the position of tokens in the sequence. It's needed because Transformers do not have a built-in sense of order like RNNs, parallel processing means the model sees all tokens at once without inherent sequence information.
- **Encoder**: Consists of multiple layers, each with:
  - Multi-head Self-Attention: Allows the model to focus on different parts of the input.
  - Feed-Forward Neural Network: Processes the attention outputs.
  - Layer Normalization and Residual Connections: Help stabilize training.
- **Decoder**: Similar to the encoder but includes:
  - Masked Multi-head Self-Attention: Prevents attending to future tokens. This layer ensures that the prediction for a particular token only depends on the known outputs at previous positions, without "seeing" future tokens.
    - Encoder-Decoder Attention: Allows the decoder to focus on relevant parts of the encoder's output, cross-referencing the input sequence.
    - Feed-Forward Neural Network: like in the encoder, processes the attention outputs.
    - Layer Normalization and Residual Connections
- **Output Layer**: Produces the final predictions, typically using a softmax function for classification tasks, outputs probabilities for each token in the vocabulary, one token at a time.
</details>

---

[🐍 GPT-like demo](./demo.ipynb)

---

<details>
<summary>🧠 Attention Is All You Need (Resources /  Credits)</summary>

- Terms & Math
  - [📘Transformer taxonomy](https://kipp.ly/transformer-taxonomy/)
  - [📘Transformer inference math](https://kipp.ly/transformer-inference-arithmetic/)

- Visual Explanations
  - [📺Video Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M)
  - [📺Video Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc)

- Books & Articles
  - [📋Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)
  - [📘The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [🧠Attention is all you need](https://arxiv.org/abs/1706.03762)

- Demos
  - [📺LLM from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
  - [🐍The annotated transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

</details>


