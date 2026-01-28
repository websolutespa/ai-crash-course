# Prompt engineering

A prompt is an input provided to a LLM to generate a response or prediction. It serves as the instruction and context that guides the AI model's output generation process

> _Prompt engineering_ is the process of discovering prompts that reliably yield useful or desired results.

_Good news_: you don't need to be an engineer or data scientist to engineer effective prompts!

## Recap of LLM prediction process

LLMs generate text by predicting the next token based on the input prompt and previously generated tokens. They use probability distributions to select tokens, influenced by training data patterns. Understanding this helps in crafting prompts that guide models towards desired outputs.

### Sampling parameters

Sampling parameters (temperature, top-K, top-P) control how LLMs select tokens from probability distributions, determining output randomness and creativity. These parameters interact: at extreme settings, one can override others (temperature 0 makes top-K/top-P irrelevant). A balanced starting point is temperature 0.2, top-P 0.95, top-K 30 for coherent but creative results. Understanding their interactions is crucial for optimal prompting—use temperature 0 for factual tasks, higher values for creativity, and combine settings strategically based on your specific use case.

- **Temperature**

    Temperature controls the randomness in token selection during text generation. Lower values (0-0.3) produce deterministic, factual outputs. Medium values (0.5-0.7) balance creativity and coherence. Higher values (0.8-1.0) generate creative, diverse outputs but may be less coherent. Use low temperature for math/facts, high for creative writing.
- **Top-p**

    Top-P (nucleus sampling) selects tokens from the smallest set whose cumulative probability exceeds threshold P. Unlike Top-K's fixed number, Top-P dynamically adjusts based on probability distribution. Low values (0.1-0.5) produce focused outputs, medium (0.6-0.9) balance creativity and coherence, high (0.9-0.99) enable creative diversity.
- **Top-k**

    Top-K restricts token selection to the K most likely tokens from the probability distribution. Low values (1-10) produce conservative, factual outputs. Medium values (20-50) balance creativity and quality. High values (50+) enable diverse, creative outputs. Use low K for technical tasks, high K for creative writing.

### Output control

Output control encompasses techniques and parameters for managing LLM response characteristics including length, format, style, and content boundaries. Key methods include max tokens for length limits, stop sequences for precise boundaries, temperature for creativity control, and structured output requirements for format consistency. Effective output control combines prompt engineering techniques with model parameters to ensure responses meet specific requirements. This is crucial for production applications where consistent, appropriately formatted outputs are essential for user experience and system integration.

- **Max tokens**

    Max tokens setting controls the maximum number of tokens an LLM can generate in response, directly impacting computation cost, response time, and energy consumption. Setting lower limits doesn't make models more concise—it simply stops generation when the limit is reached. This parameter is crucial for techniques like ReAct where models might generate unnecessary tokens after the desired response. Balancing max tokens involves considering cost efficiency, response completeness, and application requirements while ensuring critical information isn't truncated.

- **Stop sequences**
    Stop sequences are specific strings that signal the LLM to stop generating text when encountered, providing precise control over output length and format. Common examples include newlines, periods, or custom markers like "###" or "END". This parameter is particularly useful for structured outputs, preventing models from generating beyond intended boundaries. Stop sequences are essential for ReAct prompting and other scenarios where you need clean, precisely bounded responses. They offer more control than max tokens by stopping at logical breakpoints rather than arbitrary token limits.

[🐍 detailed recap](../06/transformer-overview.ipynb)

[🐍 sampling params](./sampling.ipynb)

### Repetition penalty  

Repetition penalties discourage LLMs from repeating words or phrases by reducing the probability of selecting previously used tokens. This includes frequency penalty (scales with usage count) and presence penalty (applies equally to any used token). These parameters improve output quality by promoting vocabulary diversity and preventing redundant phrasing.

- **Frequency penalty**

    Frequency penalty reduces token probability based on how frequently they've appeared in the text, with higher penalties for more frequent tokens. This prevents excessive repetition and encourages varied language use. The penalty scales with usage frequency, making overused words less likely to be selected again, improving content diversity.

- **Presence penalty**
    Presence penalty reduces the likelihood of repeating tokens that have already appeared in the text, encouraging diverse vocabulary usage. Unlike frequency penalty which considers how often tokens appear, presence penalty applies the same penalty to any previously used token, promoting varied content and creativity.

[🐍 repetition penalty](./repeat-penality.ipynb)

---

## Core principles of prompting (five is enough!)

- **Give Direction**: Describe the desired style in detail, or reference a relevant persona
- **Specify Format**: Define what rules to follow, and the required structure of the response
- **Provide Examples**: Insert a diverse set of test cases where the task was done correctly
- **Evaluate Quality**: Identify errors and rate responses, testing what drives performance
- **Divide Labor**: Split tasks into multiple steps, chained together for complex goals

[🐍 core principles - text](./5-principles-text.ipynb)

[🐍 core principles - image](./5-principles-image.ipynb)

### Structured outputs

Structured output involve prompting LLMs to return responses in specific formats like JSON, XML, or other organized structures rather than free-form text. This approach forces models to organize information systematically, reduces `hallucinations` by imposing format constraints, enables easy programmatic processing, and facilitates integration with applications. For example, requesting movie classification results as JSON with specified schema ensures consistent, parseable responses. Structured outputs are particularly valuable for data extraction, API integration, and applications requiring reliable data formatting.

[🐍 structured output](./structured-output.ipynb)

---

## System / Role / Contextual prompting

- **System Prompting**: sets the overall context, purpose, and operational guidelines for LLMs. It defines the model's role, behavioral constraints, output format requirements, and safety guardrails. System prompts provide foundational parameters that influence all subsequent interactions, ensuring consistent, controlled, and structured AI responses throughout the session.
- **Role Assignment Prompting**: assigns a specific persona or role to the LLM, shaping its tone, style, and perspective. By defining characteristics like profession, expertise level, or emotional tone, role prompts help tailor responses to fit particular contexts or audiences. This technique enhances engagement and relevance by aligning the model's output with the desired identity or viewpoint.
- **Contextual Prompting**: provides specific background information or situational details relevant to the current task, helping LLMs understand nuances and tailor responses accordingly. Unlike system or role prompts, contextual prompts supply immediate, task-specific information that's dynamic and changes based on the situation.

[🐍 system vs user prompt](./system-vs-user-prompt.ipynb)

### System prompt & User prompt

Many model APIs give you the option to split a prompt into a system prompt and a user prompt. 

Think of the system prompt as the `task description` and the user prompt as the `task`.

Typically, the instructions provided by application developers are put into the system prompt, while the instructions provided by users are put into the user prompt. 
But given a system prompt and a user prompt, the model can **combine them** into a single prompt, typically following a template (model `chat template`, and each model can have its own).

Template such as:

```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>
{{ user_message }} [/INST]
```

or
```
<system>
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ user_message }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

...under the hood, the system prompt and the user prompt are concatenated into a single
final prompt before being fed into the model.

From the model’s perspective, system prompts and user prompts are processed the same way. 

- chat template sample
[🔗 huggingface](https://huggingface.co/ibm-granite/granite-4.0-micro?chat_template=default) | [🔗 ollama](https://ollama.com/library/granite4:3b/blobs/0f6ec9740c76)

But many model providers emphasize that well-crafted system prompts can improve performance, i.e. when assigning specific role or setting behavior guidelines, it can maintain that character
more effectively throughout the conversation, exhibiting more natural and creative
responses while staying in character.

Any performance boost that a system prompt can give is likely because of one or both of the following factors:

- The system prompt comes first in the final prompt, and the model might just be better at processing instructions that come first. [🔗 Lost in the middle](https://arxiv.org/pdf/2307.03172) | [🔗 Needle in the haystack](https://observablehq.com/@shreyashankar/needle-in-the-real-world-experiments) | [🐍 needle-in-haystack](./needle-in-haystack.ipynb)
- The model might have been post-trained to pay more attention to the system
    prompt, training LLMs to prioritize `privileged instructions`, that also helps mitigate prompt attacks.

[🐍 system vs user prompt](./system-vs-user-prompt.ipynb)

---

## Hallucinations

> Hallucinations are plausible but false statements generated by language models

[🔗 Karpathy on hallucination](https://x.com/karpathy/status/1733299213503787018)

| Capability | LLM | Search engine |
|------------|-----|---------------|
| Knowledge cutoff | Yes | No |
| Factual accuracy | Medium | High |
| Reasoning | High | Low |
| Creativity | High | Low |

> LLMs always hallucinate. Sometimes their hallicinations align with your reality.

[🔗 OpenAI: why llm hallucinate](https://openai.com/index/why-language-models-hallucinate/) | 
[🔗 Paper](https://arxiv.org/pdf/2509.04664)

> [📺Feynman: can machines think?](https://www.youtube.com/watch?v=ipRvjS7q1DI) ...we are getting close to intelligent machines, but they’re showing the necessary weaknesses of intelligence.

### Reduce hallucinations through prompting

To reduce hallucinations through prompting, use techniques such as:
- **Explicit Instructions**: Clearly instruct the model to avoid making up information and to stick to known facts.
- **Structured Prompts**: Request responses in specific formats (e.g., JSON) to enforce accuracy and consistency.
- **Contextual Information**: Provide relevant context or background information to guide the model's responses.
- **Verification Steps**: Ask the model to verify its answers or provide sources for factual claims. 

[🐍 reduce hallucinations](./reduce-hallucinations.ipynb)

---

## Prompting techniques

- **Zero-shot prompting**: Directly asking the model to perform a task without prior examples.
- **One-shot / Few-shot prompting**: Providing a few examples in the prompt to guide the model.
- **Chain-of-thought (CoT) prompting**: Encouraging the model to reason through problems step-by-step.
- **Self-consistency prompting**: Generating multiple reasoning paths and selecting the most consistent answer.
- **Step-back prompting**: Asking the model to abstrat the problem before solving it.
- **ReAct prompting**: Combining reasoning and action by allowing the model to interact with tools or external data.
- **Tree-of-thought (ToT) prompting**: Expanding CoT by exploring multiple reasoning branches.

[🐍 prompting techniques](./prompting-techniques.ipynb)

### Enhance creativity

- **Verbalized sampling**: ask the model to verbalize a probabily distribution over a set of responses, requiring multiple outputs for the same query [🔗 verbalized sampling](https://arxiv.org/pdf/2510.01171)
- **Reverse prompting**: asking the model to discover prompts that can generate a desired output, effectively working backwards from the output to the input 

[🐍 enhance creativity](./enhance-creativity.ipynb)

### TL;DR

- Keep your prompts short and concise 
- Prioritize giving clearer instructions over adding constraints (_if your collegue can't understand it, the model won't either_)
- Experiment with input formats and writing styles 
- Ask for structured output if it helps e.g. JSON, XML, Markdown, CSV etc 
- Provide few-shot examples for structure or output style you need 
- Delimit different sections with triple backticks or XML tags 
- Tune sampling (temperature, top-k, top-p) for determinism vs creativity
- Right model for the job (specialized vs general purpose) 
- Enhance reasoning with chain-of-thought or multi-step prompting
- Use variables / placeholders in your prompts for easier configuration / reuse
- Document and track prompt versions 
- Automate evaluation

---

## Automatic prompt engineering
Automatic Prompt Engineering (APE) uses LLMs to generate and optimize prompts automatically, reducing human effort while enhancing model performance. The process involves prompting a model to create multiple prompt variants, evaluating them using metrics like BLEU or ROUGE, then selecting the highest-scoring candidate. For example, generating 10 variants of customer order phrases for chatbot training, then testing and refining the best performers. This iterative approach helps discover effective prompts that humans might not consider, automating the optimization process.

### Common APE frameworks
- **Prompt optimization tools**: Platforms like OpenAI's Prompt Optimizer and Anthropic's prompt engineering tools automate prompt generation and evaluation, streamlining the APE process.

    [🔗 Prompt optimizer: OpenAI](https://platform.openai.com/chat/edit?models=gpt-5&optimize=true)

    [🔗 Prompt optimizer: Anthropic](https://platform.claude.com/dashboard)

    [🔗 OpenAI prompt optimizer](https://cookbook.openai.com/examples/optimize_prompts)


- **DSPy**: A high-level language for building reliable AI systems, including prompt engineering tasks.

    [🔗 DSPy docs](https://dspy.ai/)