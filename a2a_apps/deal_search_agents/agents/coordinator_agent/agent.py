import asyncio
import os
import uuid
import httpx

from typing import Any
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool
from collections.abc import Callable
from dotenv import load_dotenv
from openai.types.responses.response_input_param import FunctionCallOutput
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
import json
import logging

load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinatorAgentConnections:

    def __init__(self, agent_card: AgentCard, agent_url: str):
        self._httpx_client = httpx.AsyncClient(timeout=30)
        self.a2a_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message_request: SendMessageRequest) -> SendMessageResponse:
        return await self.a2a_client.send_message(message_request)


class CoordinatorAgent:

    def __init__(self, task_callback: TaskUpdateCallback | None = None):

        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, CoordinatorAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''

        self.project_client : AIProjectClient = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential()
        )
        self.openai_client = self.project_client.get_openai_client()

        self.agent = None
        self.conversation = None

    @classmethod
    async def create(cls, remote_agent_addresses: list[str],
                     task_callback: TaskUpdateCallback | None = None) -> 'CoordinatorAgent':
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def list_remote_agents(self) -> str:
        if not self.remote_agent_connections:
            return "[]"

        lines = []
        for card in self.cards.values():
            lines.append(f"{card.name}: {card.description}")

        return "[\n  " + ",\n  ".join(lines) + "\n]"

    async def _async_init_components(self, remote_agent_addresses: list[str]) -> None:

        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()

                    remote_connection = CoordinatorAgentConnections(agent_card=card, agent_url=address)
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card

                except httpx.ConnectError as e:
                    print(f'ERROR: Failed to get agent card from {address}: {e}')
                except Exception as e:  # Catch other potential errors
                    print(f'ERROR: Failed to initialize connection for {address}: {e}')
            print(f"Found remote agents: {self.list_remote_agents()}")

    async def send_message(self, agent_name: str, task: str):
        client = None
        if "parsing" in agent_name.lower():
            client = self.remote_agent_connections.get("ParsingAgent")

        if not client:
            client = self.remote_agent_connections["ResearchAgent"]

        message_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{
                    "type": "text",
                    "kind": "text",
                    "text": task
                }],
                'messageId': message_id,
                "contextId": str(uuid.uuid4()),
            }
        }

        message_request = SendMessageRequest(
            id=message_id,
            params=MessageSendParams.model_validate(payload, strict=False)
        )

        send_response: SendMessageResponse = await client.send_message(message_request=message_request)

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            print('received non-success response. Aborting get task ')
            return

        if not isinstance(send_response.root.result, Task):
            print('received non-task response. Aborting get task ')
            return

        return send_response.root.result

    def create_agent(self):

        try:
            self.agent = self.project_client.agents.create_version(
                agent_name="coordinator-agent",
                definition=PromptAgentDefinition(
                    model=os.environ["MODEL_DEPLOYMENT_NAME"],
                    instructions=f"""
            You are an expert Coordinator that helps users with requests.

            Your role:
                1. Call intent-parsing agent to parse intent
                2. Call researcher to get the discounts for the output from the intent-parsing agent.
                3. Synthesize the answer with urls + confidence

            Available Agents: {self.list_remote_agents()}

            Always be helpful and route requests to the most appropriate agent.""",
                    tools=[FunctionTool(
                        name="invoke_agent",
                        description="The most appropriate agent to process requests.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "agent_name": {"type": "string"},
                                "task": {"type": "string"}
                            },
                            "required": ["agent_name", "task"],
                            "additionalProperties": False,
                        },
                        strict=True
                    )]
                )
            )
            self.conversation = self.openai_client.conversations.create()

            return self.agent

        except Exception as e:
            print(f"Error creating Azure AI agent: {e}")
            raise

    async def process_user_message(self, user_message: str) -> str:

        if not hasattr(self, 'agent') or not self.agent:
            return json.dumps({
                "status": "error",
                "error": "Azure AI Agent not initialized. Please ensure the agent is properly created."}
            )

        if not hasattr(self, 'conversation') or not self.conversation:
            return json.dumps({
                "status": "error",
                "error": "Azure AI Conversation not initialized. Please ensure the agent is properly created."}
            )

        try:
            conversation = self.openai_client.conversations.create()

            timeout = 30
            text_output = ""
            input = user_message
            while True:

                if not input:
                    break

                response = self.openai_client.responses.create(
                    input=input,
                    conversation=conversation.id,
                    extra_body={
                        "agent_reference": {
                            "name": self.agent.name,
                            "type": "agent_reference"
                        }
                    },
                    background=True
                )

                _timeout = asyncio.get_event_loop().time() + timeout
                while response.status in ["queued", "in_progress"]:
                    if asyncio.get_event_loop().time() > _timeout:
                        return json.dumps({"status": "timeout", "agent_name": self.agent.name})

                    await asyncio.sleep(5)
                    response = self.openai_client.responses.retrieve(
                        response_id=response.id,
                    )

                function_outputs = []
                if response.status != "completed":
                    return json.dumps({"status": "error",
                                       "agent_name": self.agent.name,
                                       "reason": response.status})

                for item in response.output:
                    if item.type == "message":
                        for part in item.content:
                            if hasattr(part, "text"):
                                text_output += part.text
                    elif item.type == "function_call" and item.name == "invoke_agent":
                        args = json.loads(item.arguments)

                        result = await self.send_message(
                            agent_name=args.get("agent_name"),
                            task=args.get("task"),
                        )

                        function_outputs.append(FunctionCallOutput(
                            type="function_call_output",
                            call_id=item.call_id,
                            output=json.dumps(result)
                        ))

                input = function_outputs

            return json.dumps({
                "status": "success",
                "agent_name": self.agent.name,
                "text_output": text_output
            })

        except Exception as e:
            error_msg = f"Error in process_user_message: {e}"
            logger.exception(error_msg)
            return json.dumps({"status": "error", "error": error_msg})


async def _get_initialized_coordinator_agent_sync() -> CoordinatorAgent:
    async def _async_main() -> CoordinatorAgent:
        coordinator_agent_instance = await CoordinatorAgent.create(
            remote_agent_addresses=[
                f"http://{os.environ["SERVER_URL"]}:{os.environ["INTENT_PARSING_AGENT_PORT"]}",
                f"http://{os.environ["SERVER_URL"]}:{os.environ["RESEARCH_AGENT_PORT"]}",
            ]
        )
        # Create the Azure AI agent
        coordinator_agent_instance.create_agent()
        return coordinator_agent_instance

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        raise


# Initialize the coordinator agent
coordinator_agent = _get_initialized_coordinator_agent_sync()
