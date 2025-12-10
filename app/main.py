import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from htbuilder.units import rem
from htbuilder import div, styles
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import datetime, time
import textwrap
import torch
import os, sys
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_community.retrievers import WikipediaRetriever, ArxivRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from dotenv import load_dotenv
load_dotenv()

VECTOR_STORE = "chroma"  # options: faiss, chroma
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B" #google/embeddinggemma-300m"
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2' # model for re-ranking
TOP_K = 10 # number of top documents to retrieve (after re-ranking, if used, otherwise from vector store)
LLM_PROVIDER = "ollama"  # options: openai, ollama
LLM_MODEL = "gpt-oss:20b" # gpt-4.1, granite4:3b-h , gpt-oss:20b

@st.cache_resource
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARN': '\033[33m',       # Yellow
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'FATAL': '\033[35m',      # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)
@st.cache_resource    
def configure_logging(level=logging.INFO):
    logger = logging.getLogger()
    if logger.handlers: # Prevent adding handlers multiple times
        return logger
    logger.setLevel(level)
    handler =logging.StreamHandler(sys.stderr)    
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
logger = configure_logging()

@st.cache_resource
def parse_args():
    """
    Parse command line arguments to override default configuration parameters.
    E.g. --vector_store=faiss --embedding_model=google/embeddinggemma-300m
    """
    params = ["VECTOR_STORE","EMBEDDING_MODEL","RERANK_MODEL","TOP_K","LLM_PROVIDER","LLM_MODEL"]
    settings = []
    for arg in sys.argv:
        for param in params:
            if arg.lower().startswith(f"--{param.lower()}="):
                _v = arg.split("=", 1)[1]
                if param == "TOP_K":
                    _v = int(_v)
                settings.append((param, _v))
                globals()[param] = _v
    if settings:
        logger.info(f"Configuration parameters overridden via command line arguments: {[{p:v} for p, v in settings]}")

@st.cache_resource
def get_llm(provider: str, model: str):
    return init_chat_model(
        model_provider=provider,
        model=model,
        # kwargs passed to the model:
        temperature=0,
        timeout=60,
        max_tokens=4_000,
)   
@st.cache_resource
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
@st.cache_resource
def get_embeddings(model_name: str, device: str):
    logger.info(f"Loading embeddings model: {model_name} on {device}")
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
@st.cache_resource
def get_reranker(model_name: str, device: str):
    from sentence_transformers import CrossEncoder
    logger.info(f"Loading re-ranker model: {model_name} on {device}")
    return CrossEncoder(model_name, device=device) 

@st.cache_resource
def get_retriever(vector_store: str, embedding_model: str, rerank_model: str, top_k: int, device: str):
    embeddings = get_embeddings(embedding_model, device)
    match vector_store.lower():
        case "faiss":
            from langchain_community.vectorstores import FAISS            
            vector_store = FAISS.load_local(f"./{os.path.dirname(__file__)}/tmp/db/{FAISS.__name__.lower()}", embeddings, allow_dangerous_deserialization=True)
        case "chroma":
            from langchain_chroma import Chroma as CHROMA
            vector_store = CHROMA(collection_name="default",embedding_function=embeddings,persist_directory=f"./{os.path.dirname(__file__)}/tmp/db/{CHROMA.__name__.lower()}")        
        case _:
            raise ValueError(f"Unsupported VECTOR_STORE: {vector_store}")
    #re-ranking wrapper
    model = get_reranker(rerank_model, device)
    def invoke(query: str):
        docs = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":min(top_k*5,100)}).invoke(query)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs, show_progress_bar=False)
        results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in results[:top_k]]
    return type(
        "RerankRetriever",
        (),
        {"invoke": staticmethod(invoke)},
    )()

parse_args()
device = get_device()
_kb_retriever = get_retriever(VECTOR_STORE, EMBEDDING_MODEL, RERANK_MODEL, TOP_K, device)
_wiki_retriever = WikipediaRetriever(
    lang="en", #language of the articles
    top_k_results=2, #max results to return
    load_max_docs=2, #max downloaded documents
    load_all_available_meta=False, #Published,Title,Summary
    )
_arxiv_retriever = ArxivRetriever(
    top_k_results=2, #max results to return
    load_max_docs=2, #max downloaded documents   
    load_all_available_meta=False, #Published,Title,Authors,Summary
    get_full_documents=True #fetch full text of the papers
    )
# Wrap synchronous retrievers in async
def _retrieve(retriever, query: str):
    """Run retriever in thread pool to avoid blocking"""
    retrieved_docs = retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs     
@tool(response_format="content_and_artifact")
def kb(query: str):
    """Retrieve information from course materials, documentation and code snippets about AI, LLMs and related topics."""
    return _retrieve(_kb_retriever, query)
@tool(response_format="content_and_artifact")
def wikipedia(query: str):
    """Retrieve information from Wikipedia for general knowledge and fact checking."""
    return _retrieve(_wiki_retriever, query)
@tool(response_format="content_and_artifact")
def arxiv(query: str):
    """Retrieve information from arXiv for academic research papers."""
    return _retrieve(_arxiv_retriever, query)

llm = get_llm(LLM_PROVIDER, LLM_MODEL)
SYSTEM_PROMPT = textwrap.dedent("""
    - You are an assistant of an AI engineering course, your goal is to help students to improve their skills.
    - Use ALWAYS the tool `kb`, that contains course material, documentation and code snippets about AI, LLMs and related topics.
    - When needed use other tools ['wikipedia', 'arxiv'] to access external resources for fact checking .
    - Use the same tool only once, NEVER re-call the same tool per request
    - ALWAYS cite the sources with links, in markdown format: [title of the source](source http link).
    - Use image, when available from materials, to explain concept
    - Prepend /app/static/ when sourcing local files, e.g. README.md => /app/static/README.md , 02/demo-nn.png => /app/static/02/demo-nn.png
    - Site host is http://localhost:8501 , adjust links accordingly for images and .md files, e.g => [some useful png image](http://localhost:8501/app/static/{relative_path})                               
    - For *.ipynb (notebook) files, link ALWAYS with 'open in Colab' button format:
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/websolutespa/ai-crash-course/blob/main/{notebook_file_name}.ipynb)
    - Always answer in markdown format, with proper syntax for code blocks, images and links.
    
""")

@st.cache_resource
def get_agent(llm,_tools: list): # do not hash the _tools argument, prepending underscore
    return create_agent(llm, _tools)
agent = get_agent(llm,[kb, wikipedia, arxiv])

st.set_page_config(page_title="🤖-crash-course assistant", page_icon="🚀", layout="wide")
executor = ThreadPoolExecutor(max_workers=5)

HISTORY_LENGTH = 5
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)
#https://github.com/streamlit/streamlit/blob/develop/lib/streamlit/material_icon_names.py
SUGGESTIONS = {
    ":blue[:material/model_training:] Over-underfitting in training": (
        "What is the primary purpose of a validation set during model training?"
        "Explain over and underfitting in machine learning. Cite course materials with links to file."
    ),
    ":green[:material/contextual_token:] Recap of tokenization": (
        "How tokenization influences LLM behavior? "
        "What are the relationships between tokens, words, and subwords?"
    ),
    ":orange[:material/prompt_suggestion:] Copilot prompts & variables": (
        "List Copilot built-in prompts and chat variables, with use cases."
    ),
        ":violet[:material/task:] Embeddings task": (
            "List common tasks suitable for embeddings."
        ),
    ":yellow[:material/flowchart:] Attention-head workflow": (
        "Explain attention-head workflow including images."
    ),
    ":red[:material/match_case:] Term-matching algorithms": (
        "What is the difference between TF-IDF and BM25?"
    ),
    ":blue[:material/chat:] Relevant tweet": (
        "What are the most relevant tweet from course materials? "
    ),
    ":grey[:material/gate:] MoE architecture": (
        "Explain MoE, including images from course materials"
    )    
}

@st.dialog("Legal disclaimer")
def show_disclaimer_dialog():
    st.caption("""
            This AI chatbot is powered by large language models (LLMs) and
            other AI technologies. While we strive to provide accurate and
            helpful information, please be aware that the responses generated
            by the AI may not always be correct or appropriate. The AI may
            produce responses that are biased, misleading, or offensive.
            Always verify the information provided by the AI with reliable
            sources. By using this chatbot, you acknowledge and accept these
            terms.
        """)
def show_feedback_controls(message_index):
    """Shows the "How did I do?" control."""
    st.write("")

    with st.popover("How did I do?"):
        with st.form(key=f"feedback-{message_index}", border=False):
            with st.container(gap=None):
                st.markdown(":small[Rating]")
                rating = st.feedback(options="stars")

            details = st.text_area("More information (optional)")

            if st.checkbox("Include chat history with my feedback", True):
                relevant_history = st.session_state.messages[:message_index]
            else:
                relevant_history = []

            ""  # Add some space

            if st.form_submit_button("Send feedback"):
                # TODO: Submit feedback here!
                pass    
def build_prompt(**kwargs):
    """Builds a prompt string with the kwargs as HTML-like tags.

    For example, this:

        build_prompt(foo="1\n2\n3", bar="4\n5\n6")

    ...returns:

        '''
        <foo>
        1
        2
        3
        </foo>
        <bar>
        4
        5
        6
        </bar>
        '''
    """
    prompt = []

    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")

    prompt_str = "\n".join(prompt)

    return prompt_str


# Just some little objects to make tasks more readable.
TaskInfo = namedtuple("TaskInfo", ["name", "function", "args"])
TaskResult = namedtuple("TaskResult", ["name", "result"])

def history_to_text(chat_history):
    """Converts chat history into a string."""
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)
def generate_chat_summary(messages):
    pass
def build_question_prompt(question):
    """Fetches info from different services and creates the prompt string."""
    old_history = st.session_state.messages[:-HISTORY_LENGTH]
    recent_history = st.session_state.messages[-HISTORY_LENGTH:]

    if recent_history:
        recent_history_str = history_to_text(recent_history)
    else:
        recent_history_str = None

    # Fetch information from different services in parallel.
    task_infos = []

    results = executor.map(
        lambda task_info: TaskResult(
            name=task_info.name,
            result=task_info.function(*task_info.args),
        ),
        task_infos,
    )

    context = {name: result for name, result in results}

    return build_prompt(
        instructions=SYSTEM_PROMPT,
        **context,
        recent_messages=recent_history_str,
        question=question,
    )   
def get_response(prompt):         
    """Sends the prompt to the LLM and returns the response generator."""
    try:
        for token, metadata in agent.stream(  
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="messages",
        ):
            if metadata['langgraph_node'] == 'model':       
                if token.content_blocks:     
                    _block = token.content_blocks[0]
                    if _block['type'] == 'text':
                        yield token         
    except Exception as e:
        print(f"Error during LLM call: {e}")
        yield f"Error: {e}"  

def send_telemetry(**kwargs):
    """Records some telemetry about questions being asked."""
    # TODO: Implement this.
    pass

# -----------------------------------------------------------------------------
# Draw the UI.


st.html(div(style=styles(font_size=rem(5), line_height=1))["🤖-crash-course"])

title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom",
)

with title_row:
    st.title(        
        "🐍-ic assistant",
        anchor=False,
        width="stretch",
    )

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = (
    user_just_asked_initial_question or user_just_clicked_suggestion
)

has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)

# Show a different UI when the user hasn't asked a question yet.
if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Ask a question...", key="initial_question")

        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: Legal disclaimer]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

# Show chat input at the bottom when a question has been asked.
user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]       

with title_row:

    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None

    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

# Display chat messages from history as speech bubbles.
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()  # Fix ghost message bug.

        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_feedback_controls(i)

if user_message:
    # When the user posts a message...

    # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
    # display math). The line below fixes it.
    user_message = user_message.replace("$", r"\$")

    # Display message as a speech bubble.
    with st.chat_message("user"):
        st.text(user_message)

    # Display assistant response as a speech bubble.
    with st.chat_message("assistant"):
        # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
        # display math). The line below fixes it.
        user_message = user_message.replace("$", r"\$")
        user_message = user_message.replace("'", "")
        
        # Rate-limit the input if needed.
        question_timestamp = datetime.datetime.now()
        time_diff = question_timestamp - st.session_state.prev_question_timestamp
        st.session_state.prev_question_timestamp = question_timestamp

        if time_diff < MIN_TIME_BETWEEN_REQUESTS:
            with st.spinner("Waiting..."):
                time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

        with st.spinner("Researching..."):
            full_prompt = build_question_prompt(user_message)
        
        # Send prompt to LLM and stream the response.
        with st.spinner("Thinking..."):
            response_gen = get_response(full_prompt)
            # Consume first token to show spinner until LLM actually responds
            try:
                first_token = next(response_gen)
            except StopIteration:
                first_token = None
        
        # Stream the LLM response (spinner is now gone)
        def stream_with_first():
            if first_token:
                yield first_token
            yield from response_gen
        
        response = st.write_stream(stream_with_first())

        # Add messages to chat history.
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Other stuff.
        show_feedback_controls(len(st.session_state.messages) - 1)
        send_telemetry(question=user_message, response=response)
