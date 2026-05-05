import pytest
from livekit.agents import ChatMessage
from livekit.agents.llm import ImageContent

from agent import LLM_MODEL, PREEMPTIVE_GENERATION, Assistant


@pytest.mark.asyncio
async def test_attaches_latest_video_frame_to_completed_user_turn() -> None:
    assistant = Assistant()
    message = ChatMessage(role="user", content=["what is this?"])
    assistant._latest_frame = "data:image/jpeg;base64,AAAA"  # type: ignore[assignment]

    await assistant.on_user_turn_completed(turn_ctx=None, new_message=message)  # type: ignore[arg-type]

    assert isinstance(message.content[-1], ImageContent)
    assert message.content[-1].inference_width == 512
    assert message.content[-1].inference_height == 512
    assert message.content[-1].inference_detail == "low"
    assert assistant._latest_frame is None


@pytest.mark.asyncio
async def test_does_not_add_image_when_no_video_frame_is_available() -> None:
    assistant = Assistant()
    message = ChatMessage(role="user", content=["你好"])

    await assistant.on_user_turn_completed(turn_ctx=None, new_message=message)  # type: ignore[arg-type]

    assert message.content == ["你好"]


def test_defaults_are_for_local_vision_pipeline() -> None:
    assert LLM_MODEL == "Qwen/Qwen3-VL-8B-Instruct"
    assert PREEMPTIVE_GENERATION is False
