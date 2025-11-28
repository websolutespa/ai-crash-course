# 🤖 ai-crash-course
### 🚀 Your Fast-Track to Becoming an AI Expert!

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Course Progress](https://img.shields.io/badge/Lessons-10%20Weeks-orange.svg)](#bootcamp-schedule)

---

> *"[Artificial intelligence is] the science and engineering of making intelligent machines, especially intelligent computer programs. It is related to the similar task of using computers to understand human intelligence, but AI does not have to confine itself to methods that are biologically observable."*
> **— John McCarthy, 2007**

---

## 🗺️ Reference Roadmaps

Explore these comprehensive learning paths to deepen your AI journey:

<table>
<tr>
<td align="center" width="25%">
<a href="https://roadmap.sh/ai-engineer">
<img src="https://api.iconify.design/mdi:robot.svg?color=%234285f4" width="48" height="48" alt="AI Engineer"/>
<br/><b>AI Engineer</b>
</a>
</td>
<td align="center" width="25%">
<a href="https://roadmap.sh/machine-learning">
<img src="https://api.iconify.design/mdi:brain.svg?color=%2334a853" width="48" height="48" alt="Machine Learning"/>
<br/><b>Machine Learning</b>
</a>
</td>
<td align="center" width="25%">
<a href="https://roadmap.sh/prompt-engineering">
<img src="https://api.iconify.design/mdi:message-text.svg?color=%23fbbc04" width="48" height="48" alt="Prompt Engineering"/>
<br/><b>Prompt Engineering</b>
</a>
</td>
<td align="center" width="25%">
<a href="https://roadmap.sh/ai-red-teaming">
<img src="https://api.iconify.design/mdi:shield-alert.svg?color=%23ea4335" width="48" height="48" alt="AI Red Teaming"/>
<br/><b>AI Red Teaming</b>
</a>
</td>
</tr>
</table>

---

## 📚 Bootcamp Schedule

### 🎯 10-week intensive AI training program

| 📅 Week | 📖 Module | 🎓 Learning Goals | 🔑 Key Topics |
|:-------:|-----------|-------------------|---------------|
| [**01**](./01/README.md) | **🔰 AI Engineer Basics** | Understand role of AI Engineer, history, terminology, and foundations | • 📜 AI brief history<br/>• 💻 Software evolution<br/>• 🖥️ OS evolution |
| [**02**](./02/README.md) | **🧠 ML training & Neural Networks** | Learn fundamentals of ML/DL and pre-trained models | • 🤖 ML types: supervised, unsupervised, RL<br/>• 🕸️ Neural nets, generative models, LLMs<br/> |
| [**03**](./03/README.md) | **🔌 LLM training** | Deep-dive in LLM 3 training phases |  • 🏭 Pre-trained models: HuggingFace<br/>• 🎯 Fine-tuning basics<br/>• 🔄 RLHF |
| [**04**](./04/README.md) | **📦 LLM Catalog & Classification** | Classify and compare LLMs | • 🗂️ Open vs Closed source<br/>• ⚙️ Quantization<br/>• 🧑‍🏫 Distillation<br/>• 📊 Benchmarking |
| [**05**](./05/README.md) | **🧬 Tokenizer & Embeddings** | Learn tokenization, embeddings | • 🎫 Token management, moderation<br/> • ❓ What are embeddings?<br/>• 🎯 Use cases: semantic search, recsys |
| [**06**](./06/README.md) | **🧠 LLM Architecture** | Learn LLM architecture | • 🗣️ NLP tasks<br/>• 🤔 Why Transformers?<br/>• 🧠 Attention Is All You Need |
| [**07**](./07/README.md) | **🔍 RAG & tools** | Build retrieval-augmented generation pipelines<br/>Add tools to LLM | • 📊 Vector DBs: Chroma, FAISS, Qdrant<br/>• ✂️ Chunking & embedding<br/>• 🔎 Retrieval process<br/>• ✨ Generation step<br/>• ⛓️ LangChain<br/>• 🛠️ Tool integration |
| [**08**](./08/README.md) | **🤖 AI Agents & Model Context Protocol** | Build AI agents with tool use capabilities and apply MCP for interoperability | • 🧩 Agent architectures<br/>• 🔄 Feedback loops<br/>• 📚 LangChain agents<br/>• 🏗️ MCP concept & architecture<br/>• 🔧 Tool/plugin integration<br/>• 🤖 Use cases in AI agents |
| [**09**](./09/README.md) | **✍️ Prompting Techniques I** | Apply core prompting strategies | • 🎯 Zero-shot, one-shot, few-shot<br/>• 🎭 Role/system/contextual prompting<br/> • 🧵 Chain of Thought (CoT)<br/>• 🌳 Tree of Thoughts (ToT)<br/> |
| [**10**](./10/README.md) | **🧩 Prompting Techniques II** | Advanced prompting patterns | • ⚡ ReAct<br/>• 🤖 Automatic prompt engineering<br/>• 🛡️ Prompting best practices<br/>• 🔒 Injection defenses<br/>• 🎯 Structured outputs<br/>• 🔴 AI Red Teaming: jailbreaks, defenses |

### 🚀 GitHub Copilot Overview

| 📅 Lesson | 📖 Module | 🎓 Learning Goals |
|:-------:|-----------|-------------------|
| [**01**](./gh-copilot/01/README.md) | **🤖 Copilot Basics** | Master Copilot features, customization and MCP |
| [**02**](./gh-copilot/02/README.md) | **🛠️ Advanced Workflows** | Code review, Coding Agents and legacy refactoring |

---

## ⚙️ Requirements & Setup

### 📋 Prerequisites

- 🐍 **Python 3.12+** - [Download here](https://www.python.org/downloads/)
- 📝 **Code Editor** - VS Code recommended
- 🔑 **API Keys** - OpenAI, Anthropic, or other LLM providers

### 🚀 Quick Start

```pwsh
# 🐍 uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# linux: curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 1️⃣ Clone the repository
git clone https://github.com/websolutespa/ai-crash-course.git
cd ai-crash-course

# 2️⃣ Create and activate virtual environment
uv venv

# Activate virtual environment:
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# install ipykernel
uv pip install ipykernel -U --force-reinstall

# 3️⃣ Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys and settings

# 4️⃣ Install dependencies
uv pip install -U -r requirements.txt

# 5️⃣ Start learning! 🎉
code .
```

- Note for CUDA users: torch installation &  set `TORCH_CUDA_ARCH_LIST` in `.env` according to your GPU
```sh
nvcc --version
nvidia-smi
#check CUDA capacity, e.g., (12, 1)
python -c "import torch; print(torch.cuda.get_device_capability())"
# Or use nvidia-smi to find your GPU model, then check: https://developer.nvidia.com/cuda-gpus
# set TORCH_CUDA_ARCH_LIST .env accordingly, e.g.:  TORCH_CUDA_ARCH_LIST=12.1
# finally:
uv pip install -U torch --index-url https://download.pytorch.org/whl/cu130
```

<div align="center">

**✨ ready to begin your AI journey? => [week 01](./01/README.md)! ✨**

</div>