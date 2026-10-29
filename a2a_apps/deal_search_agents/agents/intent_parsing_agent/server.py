import os
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv
from agents.intent_parsing_agent.agent_executor import create_foundry_agent_executor
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

load_dotenv()

host = os.environ.get("SERVER_URL", "localhost")
port = int(os.environ.get("INTENT_PARSING_AGENT_PORT", "10008"))

# Define agent skills
skills = [
    AgentSkill(
        id='intent_parsing_agent',
        name='Parse user query',
        description='Parse user query to help personalize shopping with budget',
        tags=['intent_parsing_agent'],
        examples=[
            'Discounts on walking shoes',
        ],
    ),
]

# Create agent card
agent_card = AgentCard(
    name='ParsingAgent',
    description='Agent to parse the user query'
    'I can help you with the shopping with deal',
    url=f'http://{host}:{port}/',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=skills,
)

# Create agent executor
agent_executor = create_foundry_agent_executor(agent_card)

# Create request handler
request_handler = DefaultRequestHandler(
    agent_executor=agent_executor, task_store=InMemoryTaskStore()
)

# Create A2A application
a2a_app = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)

# Get routes
routes = a2a_app.routes()

# Add health check endpoint
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse('AI Foundry IntentParsing Agent is running!')

routes.append(Route(path='/health', methods=['GET'], endpoint=health_check))

# Create Starlette app
app = Starlette(routes=routes)

def main():
    # Run the server
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()
