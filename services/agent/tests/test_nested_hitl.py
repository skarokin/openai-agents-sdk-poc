"""Deterministic verification that nested HITL bubbles to the root run."""

import json
import tempfile
import time
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agents import Runner
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.run import RunConfig
from agents.tool import Tool
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from agent.core.agent_factory import AgentFactory
from agent.core.config import load_service_config
from agent.core.models import (
    AgentContext,
    EventSource,
    IdentityContext,
    RunEvent,
    TextChunk,
)
from agent.core.runtime import create_run_config
from agent.core.session_manager import SessionManager
from agent.hooks import GLOBAL_DEADLINE_HOOK, SOFT_DEADLINE_MESSAGE


def _tool_call(name: str, call_id: str, arguments: dict[str, Any]):
    return ResponseFunctionToolCall(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=json.dumps(arguments),
    )


def _message(text: str, item_id: str):
    return ResponseOutputMessage(
        id=item_id,
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(
                type="output_text",
                text=text,
                annotations=[],
            )
        ],
    )


def _response(output: list[TResponseOutputItem]) -> Response:
    return Response(
        id="response-test",
        created_at=1,
        model="test",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        top_p=None,
        usage=ResponseUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_tokens_details=InputTokensDetails(
                cached_tokens=0,
                cache_write_tokens=0,
            ),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    )


class SequenceModel(Model):
    def __init__(self, outputs: list[list[TResponseOutputItem]]):
        self.outputs = outputs
        self.seen_inputs: list[str | list[TResponseInputItem]] = []

    def _next(self) -> list[TResponseOutputItem]:
        if not self.outputs:
            raise AssertionError("The test model received an unexpected extra turn")
        return self.outputs.pop(0)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        self.seen_inputs.append(input)
        del (
            system_instructions,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        return ModelResponse(
            output=self._next(),
            usage=Usage(),
            response_id="response-test",
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        del (
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        output = self._next()
        response = _response(output)
        sequence = 0
        yield ResponseCreatedEvent(
            type="response.created",
            response=response,
            sequence_number=sequence,
        )
        sequence += 1
        yield ResponseInProgressEvent(
            type="response.in_progress",
            response=response,
            sequence_number=sequence,
        )
        sequence += 1
        for index, item in enumerate(output):
            yield ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=item,
                output_index=index,
                sequence_number=sequence,
            )
            sequence += 1
            if isinstance(item, ResponseFunctionToolCall):
                yield ResponseFunctionCallArgumentsDoneEvent(
                    type="response.function_call_arguments.done",
                    item_id=item.call_id,
                    output_index=index,
                    name=item.name,
                    arguments=item.arguments,
                    sequence_number=sequence,
                )
                sequence += 1
            elif isinstance(item, ResponseOutputMessage):
                for content_index, content in enumerate(item.content):
                    if isinstance(content, ResponseOutputText):
                        yield ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            item_id=item.id,
                            output_index=index,
                            content_index=content_index,
                            delta=content.text,
                            logprobs=[],
                            sequence_number=sequence,
                        )
                        sequence += 1
            yield ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=item,
                output_index=index,
                sequence_number=sequence,
            )
            sequence += 1
        yield ResponseCompletedEvent(
            type="response.completed",
            response=response,
            sequence_number=sequence,
        )


class CollectingSink:
    def __init__(self):
        self.events: list[tuple[RunEvent, EventSource]] = []

    async def emit(self, event: RunEvent, *, source: EventSource) -> None:
        self.events.append((event, source))


class NestedHitlTest(unittest.IsolatedAsyncioTestCase):
    async def test_soft_deadline_steers_model_without_failing_run(self):
        model = SequenceModel(
            [
                [_message("deadline fallback", "deadline-message")],
            ]
        )
        factory = AgentFactory(load_service_config(), model=model)
        sink = CollectingSink()
        context = AgentContext(
            identity=IdentityContext(
                subject_id="deadline-user",
                roles=frozenset({"calculator"}),
            ),
            event_sink=sink,
            request_id="deadline-request",
            session_id="deadline-session",
            deadline_epoch_seconds=time.time() - 1,
        )
        agent = await factory.create(context)
        run_config = create_run_config(context)
        run_config.tracing_disabled = True

        result = await Runner.run(
            agent,
            "Calculate after the deadline",
            context=context,
            hooks=GLOBAL_DEADLINE_HOOK,
            run_config=run_config,
        )

        self.assertEqual(result.final_output, "deadline fallback")
        self.assertTrue(model.seen_inputs)
        first_input = model.seen_inputs[0]
        self.assertIsInstance(first_input, list)
        self.assertTrue(
            any(
                isinstance(item, dict) and item.get("content") == SOFT_DEADLINE_MESSAGE
                for item in first_input
            )
        )

    async def test_nested_approvals_bubble_to_root_state(self):
        model = SequenceModel(
            [
                [
                    _tool_call(
                        "protected_subagent",
                        "call-protected",
                        {"input": "Call approval_demo"},
                    )
                ],
                [_tool_call("approval_demo", "call-nested-approval", {})],
                [_message("nested complete", "nested-message")],
                [_message("root complete", "root-message")],
            ]
        )
        factory = AgentFactory(load_service_config(), model=model)
        sink = CollectingSink()
        context = AgentContext(
            identity=IdentityContext(
                subject_id="test-user",
                roles=frozenset({"calculator", "approval_user", "subagent_user"}),
            ),
            event_sink=sink,
            request_id="test-request",
            session_id="test-session",
            deadline_epoch_seconds=time.time() + 60,
        )
        agent = await factory.create(context)
        run_config = RunConfig(tracing_disabled=True)

        first = await Runner.run(
            agent,
            "Use the protected subagent",
            context=context,
            hooks=GLOBAL_DEADLINE_HOOK,
            run_config=run_config,
        )
        self.assertEqual(
            [item.name for item in first.interruptions],
            ["protected_subagent"],
        )

        directory = self.enterContext(tempfile.TemporaryDirectory())
        sessions = SessionManager(Path(directory))
        first_token = await sessions.save_run_state(
            first.to_state(),
            session_id=context.session_id,
            owner_subject_id=context.identity.subject_id,
        )
        with self.assertRaises(PermissionError):
            sessions.run_state_session_id(first_token, "different-user")
        agent = await factory.create(context)
        first_state, _ = await sessions.load_run_state(
            first_token,
            agent=agent,
            context=context,
        )
        sessions.apply_decisions(first_state, {"call-protected": "approve"})
        second = await Runner.run(
            agent,
            first_state,
            context=context,
            hooks=GLOBAL_DEADLINE_HOOK,
            run_config=run_config,
        )
        self.assertEqual(
            [item.name for item in second.interruptions],
            ["approval_demo"],
            f"final={second.final_output!r}, items={second.new_items!r}, remaining={len(model.outputs)}",
        )

        second_token = await sessions.save_run_state(
            second.to_state(),
            session_id=context.session_id,
            owner_subject_id=context.identity.subject_id,
        )
        agent = await factory.create(context)
        second_state, _ = await sessions.load_run_state(
            second_token,
            agent=agent,
            context=context,
        )
        sessions.apply_decisions(
            second_state,
            {"call-nested-approval": "approve"},
        )
        final = await Runner.run(
            agent,
            second_state,
            context=context,
            hooks=GLOBAL_DEADLINE_HOOK,
            run_config=run_config,
        )
        self.assertFalse(final.interruptions)
        self.assertEqual(final.final_output, "root complete")
        self.assertTrue(
            any(
                isinstance(event, TextChunk)
                and event.delta == "nested complete"
                and source.kind == "agent_tool"
                for event, source in sink.events
            )
        )

    async def test_open_subagent_only_interrupts_for_nested_tool(self):
        model = SequenceModel(
            [
                [
                    _tool_call(
                        "open_subagent",
                        "call-open",
                        {"input": "Call approval_demo"},
                    )
                ],
                [_tool_call("approval_demo", "call-open-nested-approval", {})],
                [_message("open nested complete", "open-nested-message")],
                [_message("open root complete", "open-root-message")],
            ]
        )
        factory = AgentFactory(load_service_config(), model=model)
        context = AgentContext(
            identity=IdentityContext(
                subject_id="test-user",
                roles=frozenset({"calculator", "approval_user", "subagent_user"}),
            ),
            event_sink=CollectingSink(),
            request_id="open-test-request",
            session_id="open-test-session",
            deadline_epoch_seconds=time.time() + 60,
        )
        agent = await factory.create(context)
        run_config = RunConfig(tracing_disabled=True)

        first = await Runner.run(
            agent,
            "Use the open subagent",
            context=context,
            hooks=GLOBAL_DEADLINE_HOOK,
            run_config=run_config,
        )
        self.assertEqual(
            [item.name for item in first.interruptions],
            ["approval_demo"],
        )

        state = first.to_state()
        state.approve(first.interruptions[0])
        final = await Runner.run(
            agent,
            state,
            context=context,
            hooks=GLOBAL_DEADLINE_HOOK,
            run_config=run_config,
        )
        self.assertFalse(final.interruptions)
        self.assertEqual(final.final_output, "open root complete")

    async def test_local_mcp_requires_approval_only_for_delete_note(self):
        config = load_service_config()
        model = SequenceModel(
            [
                [_tool_call("list_notes", "call-list", {})],
                [
                    _tool_call(
                        "delete_note",
                        "call-delete",
                        {"note_id": "1"},
                    )
                ],
                [_message("listed then deleted", "mcp-message")],
            ]
        )
        factory = AgentFactory(config, model=model)
        try:
            denied = await factory.create(
                AgentContext(
                    identity=IdentityContext(
                        subject_id="mcp-user",
                        roles=frozenset({"calculator"}),
                    ),
                    event_sink=CollectingSink(),
                    request_id="mcp-denied",
                    session_id="mcp-denied",
                    deadline_epoch_seconds=time.time() + 60,
                )
            )
            self.assertEqual(denied.mcp_servers, [])
            self.assertEqual(factory._mcp_servers, {})

            context = AgentContext(
                identity=IdentityContext(
                    subject_id="mcp-user",
                    roles=frozenset({"mcp_user"}),
                ),
                event_sink=CollectingSink(),
                request_id="mcp-request",
                session_id="mcp-session",
                deadline_epoch_seconds=time.time() + 60,
            )
            agent = await factory.create(context)
            run_config = RunConfig(tracing_disabled=True)

            first = await Runner.run(
                agent,
                "List notes, then delete note 1",
                context=context,
                run_config=run_config,
            )
            self.assertEqual(
                [item.name for item in first.interruptions],
                ["delete_note"],
            )

            state = first.to_state()
            state.approve(first.interruptions[0])
            final = await Runner.run(
                agent,
                state,
                context=context,
                run_config=run_config,
            )
            self.assertFalse(final.interruptions)
            self.assertEqual(final.final_output, "listed then deleted")
        finally:
            await factory.close()


if __name__ == "__main__":
    unittest.main()
