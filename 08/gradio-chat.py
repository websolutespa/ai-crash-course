import gradio as gr
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient 
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

async def predict(message, history):    
    history_format = [{"role": "system", "content": prompt}]
    for msg in history:
        history_format.append(msg)
    history_format.append({"role": "user", "content": message})     
    async for chunk in agent.astream({"messages": history_format}, stream_mode="updates"):
        try:
            for step, data in chunk.items():
                # prod mode: extract only model response
                """
                if step == 'model' and "messages" in data and len(data["messages"]) > 0:
                    last_message = data["messages"][-1].content_blocks[0]
                    if last_message['type'] == 'text':
                        yield last_message['text']  
                """    
                # debug mode                   
                _msg = f"**{step}**: "
                if "messages" in data and len(data["messages"]) > 0:
                    last_message = data["messages"][-1].content_blocks[0]
                    _msg += f"[`{last_message['type']}`] - "
                    if last_message['type'] == 'text':
                        _msg += last_message['text']
                    else:
                        _msg += str(last_message)
                yield _msg
                await asyncio.sleep(2)  # simulate delay for better UX
        except Exception as e:
            yield f"Error processing chunk: {e}"
        

client = MultiServerMCPClient(
    {
        "mcp-server": {
            "url": "http://127.0.0.1:7860/gradio_api/mcp",
            "transport": "http",
        }
    }        
)
try:
    tools = asyncio.run(client.get_tools())
    model = init_chat_model(model="granite4:3b", model_provider="ollama")
    prompt = """You are an AI agent that uses the tools to perform nlp tasks."""
    agent = create_agent(model=model, tools=tools, system_prompt=prompt)

    chat = gr.ChatInterface(
        fn=predict,        
        examples=["Analyze the sentiment of the following text 'This is awesome'",
                "Summarize the following text: 'Gradio is an open-source Python library that simplifies the process of \
                    creating user interfaces for machine learning models. It allows developers to quickly build and share web-based \
                    applications that showcase their models, making it easier for others to interact with and understand the capabilities'",
                "Perform named entity recognition on the following text: 'My name is Massimo and I work at Websolute in Italy.'"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
        show_progress="full"
    )
    chat.launch()
finally:
    if client:  
        client.connections.clear()