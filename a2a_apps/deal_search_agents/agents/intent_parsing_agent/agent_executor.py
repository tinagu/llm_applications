from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCard, Part, TaskState
from a2a.utils.message import new_agent_text_message
from a2a.utils import new_task
from agents.intent_parsing_agent.agent import IntentParsingAgent, create_foundry_intent_parsing_agent
import json
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentParsingAgentExecutor(AgentExecutor):

    def __init__(self, card: AgentCard):
        self._card = card
        self._foundry_agent: IntentParsingAgent | None = None

    async def _get_or_create_agent(self) -> IntentParsingAgent:
        if not self._foundry_agent:
            self._foundry_agent = await create_foundry_intent_parsing_agent()
        return self._foundry_agent

    async def _process_request(self, user_message: str, context_id: str, task_updater: TaskUpdater) -> None:
        # Process a user request through the Foundry agent

        try:
            agent = await self._get_or_create_agent()

            # Update the task status
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message('IntentParsing Agent is processing your request...', context_id=context_id)
            )

            # Run the conversation
            responses = await agent.run_conversation(user_message)

            # Update the task with responses
            for response in responses:
                message = response if isinstance(response, str) else json.dumps(response)
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(message, context_id=context_id)
                )

            # Mark the task as complete
            if response:
                message = responses[-1] if isinstance(responses[-1], str) else json.dumps(responses[-1])
            else:
                message = "Task completed."

            await task_updater.complete(
                message=new_agent_text_message(message, context_id=context_id)
            )

        except Exception as e:
            await task_updater.failed(
                message=new_agent_text_message('IntentParsing Agent failed to process the request.',
                context_id=context_id)
            )

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_input = context.get_user_input()
        task = context.current_task

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.submit()

        await updater.start_work()

        await self._process_request(user_input, context.context_id, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.failed(
            message=new_agent_text_message('Task cancelled by user', context_id=context.context_id)
        )

def create_foundry_agent_executor(card: AgentCard) -> IntentParsingAgentExecutor:
    return IntentParsingAgentExecutor(card)
