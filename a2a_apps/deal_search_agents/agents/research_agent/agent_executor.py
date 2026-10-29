from a2a.server.events.event_queue import EventQueue
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message
from a2a.types import AgentCard, Part, TaskState
from agents.research_agent.agent import ResearchAgent, create_foundry_agent
import json


class ResearchAgentExecutor(AgentExecutor):

    def __init__(self, card: AgentCard):
        self._card = card
        self._foundry_agent: ResearchAgent | None = None

    async def _get_or_create_agent(self) -> ResearchAgent:
        if not self._foundry_agent:
            self._foundry_agent = await create_foundry_agent()
        return self._foundry_agent

    async def _process_request(self, user_message, context_id: str, task_updater: TaskUpdater) -> None:
        try:
            agent = await self._get_or_create_agent()

            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message('ResearchAgent Agent is processing your request...', context_id=context_id),
            )

            responses = await agent.run_conversation(user_message)

            for response in responses:
                response = response if isinstance(response, str) else json.dumps(response)
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(response, context_id=context_id),
                )

            if responses:
                message = responses[-1] if isinstance(responses[-1], str) else json.dumps(responses[-1])
            else:
                message = 'Task completed.'

            await task_updater.complete(
                message=new_agent_text_message(message, context_id=context_id)
            )

        except Exception as e:
            print(f'ResearchAgent: Error processing request - {e}')
            await task_updater.failed(
                message=new_agent_text_message("ResearchAgent failed to process the request.",
                context_id=context_id)
            )

    async def execute(self, context: RequestContext, event_queue: EventQueue,):
       
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.submit()

        await updater.start_work()

        # Process the request
        await self._process_request(
            context.get_user_input(),
            context.context_id,
            updater
        )
        

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        print(f'ResearchAgent: Cancelling execution for context {context.context_id}')

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.failed(
            message=new_agent_text_message('Task cancelled by user', context_id=context.context_id)
        )

def create_foundry_agent_executor(card: AgentCard) -> ResearchAgentExecutor:
    return ResearchAgentExecutor(card)

