import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, AgentVersionDetails, FunctionTool
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from openai.types.responses.response_input_param import FunctionCallOutput
from azure.identity import DefaultAzureCredential
import logging
import asyncio
from typing import Annotated
from pydantic import Field
import numpy as np
import json
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def search_tool(
        product_name: Annotated[str, Field(description="Product name to search the deal")],
) -> list[dict]:
    return [{
          "url": "http://shopping1",
          "source": "shopping1.com",
          "product": product_name,
          "price_before": 100 * np.random.rand(),
          "price_after": np.random.rand(),
          "confidence": "high"
        },
        {
            "url": "http://shopping2",
            "source": "shopping2.com",
            "product": product_name,
            "price_before": 50 * np.random.rand(),
            "price_after": np.random.rand(),
            "confidence": "medium"
        }
    ]


deal_search_tool = FunctionTool(
    name="search_deal",
    description="Get the discounts for the product",
    parameters={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "the name to search the deal",
            },
        },
        "required": ["product_name"],
        "additionalProperties": False,
    },
    strict=True,
)


class ResearchAgent:

    def __init__(self):

        self.project_client : AIProjectClient = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential()
        )
        self.openai_client: OpenAI = self.project_client.get_openai_client()
        self.agent : AgentVersionDetails = None

    async def create_agent(self) -> AgentVersionDetails:
        if self.agent:
            return self.agent

        self.agent = self.project_client.agents.create_version(
            agent_name="research-agent",
            definition=PromptAgentDefinition(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                instructions="""Search deals, discounts, coupons, and promo codes for the query. Return structured results across multiple retailers — don't just return one source.""",
                tools=[deal_search_tool]
            )
        )

        return self.agent


    async def run_conversation(self, user_message: str) -> dict:
        if not self.agent:
            await self.create_agent()

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
                    return {"status": "timeout", "agent_name": self.agent.name}

                await asyncio.sleep(5)
                response = self.openai_client.responses.retrieve(
                    response_id=response.id,
                )

            function_outputs = []
            if response.status != "completed":
                return {
                    "status": "error",
                    "agent_name":  self.agent.name,
                    "reason": response.status
                }

            # step 4: read output items
            for item in response.output:
                if item.type == "message":
                    for part in item.content:
                        if hasattr(part, "text"):
                            text_output += part.text
                elif item.type == "function_call":
                    result = await search_tool(
                        **json.loads(item.arguments),
                    )
                    function_outputs.append(
                        FunctionCallOutput(
                            type="function_call_output",
                            call_id=item.call_id,
                            output=json.dumps(result)
                        )
                    )

            input = function_outputs

        return [{
            "status": "success",
            "agent_name": self.agent.name,
            "text_output": text_output
        }]


async def create_foundry_agent() -> ResearchAgent:
    agent = ResearchAgent()
    await agent.create_agent()
    return agent


