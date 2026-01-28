from fastapi import FastAPI
from pydantic import BaseModel
from langchain.messages import AIMessage

app = FastAPI()


class GenerateRequest(BaseModel):
    query: str

@app.post("/generate")
def _generate(payload: GenerateRequest) -> AIMessage:
    from langchain.agents import create_agent
    from langchain.agents.middleware import after_agent, AgentState
    from langgraph.runtime import Runtime
    from langchain.messages import AIMessage
    from langchain.chat_models import init_chat_model
    from typing import Any

    @after_agent(can_jump_to=["end"])
    def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Model-based guardrail: Use an LLM to evaluate response safety."""
        # Get the final AI response
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a model to evaluate safety, or alternative approach like external API, ML classifier, etc.
        safety_prompt = f"""Evaluate if this response is safe and appropriate.
        Respond with only 'SAFE' or 'UNSAFE'.

        Response: {last_message.content}"""

        result = safety_model.invoke([{"role": "user", "content": safety_prompt}])
        print(f"Content: {last_message.content[:30]}...\n===\nEvaluation: {result.content}")

        if "UNSAFE" in result.content:
            last_message.content = "I cannot provide that response. Please rephrase your request."

        return None

    safety_model = init_chat_model("ollama:gemma3:4b") 
    agent = create_agent(
        model="ollama:granite4:3b",
        middleware=[safety_guardrail],
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": payload.query}]
    })
    return result["messages"][-1]

if __name__ == "__main__":	
	import uvicorn
	uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )