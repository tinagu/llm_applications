import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, AgentVersionDetails
from openai.types.responses import EasyInputMessageParam
from azure.identity import DefaultAzureCredential
import logging
from agents.cmn.utils import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

class IntentParsingAgent:

    def __init__(self):

        # Create the agents client
        self.project_client = AIProjectClient(
            endpoint=os.environ['PROJECT_ENDPOINT'],
            credential=DefaultAzureCredential()
        )
        self.openai_client = self.project_client.get_openai_client()
        self.agent: AgentVersionDetails | None = None


    async def create_agent(self) -> AgentVersionDetails:
        if self.agent:
            return self.agent

        self.agent = self.project_client.agents.create_version(
            agent_name="intent-parsing-agent",
            definition=PromptAgentDefinition(
                model=os.environ['MODEL_DEPLOYMENT_NAME'],
                instructions="""You are a deals researcher. You extract the product from the user query. You also need to extract timing, budget, location or other constraints. If the user doesn't specify the constraint, put it as 'None' 

    - Output format:
       {
          "product": "<product_lower_case>",
          "budget": "None",
          "location": "None",
          "timing": "None",
          "other": "None"
       }
    """
            )
        )

        return self.agent


    async def run_conversation(self, user_message: str) -> list[dict]:
        print("IntentParsing: run_conversation: ", user_message)

        if not self.agent:
            await self.create_agent()

        # Create a thread for the chat session
        conversation = self.openai_client.conversations.create()

        response = await run_agent(self.openai_client,
                                        conversation,
                                       self.agent.name,
                                        [
                                            EasyInputMessageParam(
                                                role="user",
                                                content=user_message,
                                            )
                                        ])

        return [response]


async def create_foundry_intent_parsing_agent() -> IntentParsingAgent:
    agent = IntentParsingAgent()
    await agent.create_agent()
    return agent

