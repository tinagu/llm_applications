import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from agents.coordinator_agent.agent import CoordinatorAgent

load_dotenv()

coordinator_agent: CoordinatorAgent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global coordinator_agent
    print("Starting up: Initializing coordinator_agent agent...")
    coordinator_agent = await CoordinatorAgent.create([
        f"http://{os.environ["SERVER_URL"]}:{os.environ["INTENT_PARSING_AGENT_PORT"]}",
        f"http://{os.environ["SERVER_URL"]}:{os.environ["RESEARCH_AGENT_PORT"]}",
    ])
    coordinator_agent.create_agent()
    print("Coordinator agent initialized.")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/message")
async def handle_message(request: Request):
    print("Agent: Processing request, please wait.")

    data = await request.json()
    user_message = data.get("message")

    if not user_message:
        return {"error": "No message provided."}
    
    try:
        response = await coordinator_agent.process_user_message(user_message)

    except Exception as e:
        return {"error": f"Failed to process message: {str(e)}"}
    
    return {"response": response}

@app.get("/health")
async def health_check():
    return {"status": "Coordinator agent is running!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("COORDINATOR_AGENT_PORT", "10004"))
    uvicorn.run("coordinator_agent.server:app", host="127.0.0.1", port=port, reload=True)
