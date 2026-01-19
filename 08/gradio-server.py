# uv pip install "gradio[mcp]"
import gradio as gr
import torch
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map=device)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device_map=device)
nerer= pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple",  device_map=device)

def summarize_text(text: str) -> str:
    """
    Summarize the given text using a pre-trained transformer model.
    
    Args:
        text (str): The text to summarize

    Returns:
        str: The summarized text
    """
    return summarizer(text)[0]['summary_text']

def sentiment_analysis(text: str) -> str:
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: A JSON string containing:
            -   label : NEGATIVE or POSITIVE
            -   score : 0 to 1 confidence score (higher is more confident)
    """
    return classifier(text)[0]

def named_entity_recognition(text: str) -> str:
    """
    Perform named entity recognition on the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: A JSON string with an array of the recognized entities.
            Each entity has the following fields:
            -   entity_group : The type of entity (e.g., PER, ORG, LOC, MISC)
            -   score : 0 to 1 confidence score (higher is more confident)
            -   word : The text of the entity
            -   start : The start character index of the entity in the input text
            -   end : The end character index of the entity in the input text
    """
    return nerer(text) 
    #to highlight entities in the text
    #return {"text": text, "entities": nerer(text)}

sentiment = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Enter text to analyze..."),
    outputs=gr.JSON(),  # gr.JSON() / gr.Textbox()
    title="Text Sentiment Analysis",
    description="Analyze the sentiment of text using pre-trained transformer model"
)

summarize = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(placeholder="Enter text to summarize..."),
    outputs=gr.Textbox(),
    title="Text Summarization",
    description="Summarize text using pre-trained transformer model"
)

ner = gr.Interface(
    fn=named_entity_recognition,
    inputs=gr.Textbox(placeholder="Enter text for named entity recognition..."),
    outputs=gr.JSON(), #gr.HighlightedText(),
    title="Named Entity Recognition",
    description="Perform named entity recognition using pre-trained transformer model"
)

server = gr.TabbedInterface(
    interface_list=[sentiment, summarize, ner],
    tab_names=["Sentiment Analysis", "Text Summarization", "Named Entity Recognition"],
    title="NLP Tasks",
    analytics_enabled=False
)

# launch interface and mcp server, sharing it publicly
if __name__ == "__main__":
    server.launch(mcp_server=True,share=False)