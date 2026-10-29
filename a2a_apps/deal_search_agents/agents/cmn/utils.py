import asyncio


async def run_agent(
        openai_client,
        conversation,
        agent_name: str,
        user_input: str,
        timeout: float | None = 30.0) -> dict:
    response = openai_client.responses.create(
        input=user_input,
        conversation=conversation.id,
        extra_body={
            "agent_reference": {
                "name": agent_name,
                "type": "agent_reference"
            }
        },
        background=True
    )

    deadline = asyncio.get_event_loop().time() + timeout
    while response.status in ["queued", "in_progress"]:
        if asyncio.get_event_loop().time() > deadline:
            return {"status": "timeout", "agent_name": agent_name}

        await asyncio.sleep(5)
        response = openai_client.responses.retrieve(
            response_id=response.id,
        )

    if response.status != "completed":
        return {"status": "error", "agent_name": agent_name, "reason": response.status}

    text_output = ""
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if hasattr(part, "text"):
                    text_output += part.text

    return {
        "status": "success",
        "agent_name": agent_name,
        "text_output": text_output
    }