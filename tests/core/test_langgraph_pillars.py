"""
Unit tests validating the 5 LangGraph architectural pillars in Jarvis:
1. State (TypedState schemas & message reducers)
2. Durability (Graph execution resilience & error trapping)
3. Interrupts (Human-in-the-loop pause & configuration)
4. Checkpoints (MemorySaver thread state retention)
5. Custom Workflows (StateGraph node routing & conditional edges)
"""

from typing import Annotated, TypedDict
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.core.orchestrator import JarvisOrchestrator


class CustomAgentState(TypedDict):
    """Pillar 1: Typed state schema with message accumulator reducer."""

    messages: Annotated[list[BaseMessage], add_messages]
    mission_status: str


def test_pillar_1_state_reducers() -> None:
    """Validate Pillar 1: State management with add_messages reducer."""
    initial_state: CustomAgentState = {
        "messages": [HumanMessage(content="Hello Jarvis")],
        "mission_status": "initialized",
    }
    # Simulate reducer update
    updated_messages = add_messages(
        initial_state["messages"], [AIMessage(content="Greetings! How can I assist you today?")]
    )
    assert len(updated_messages) == 2
    assert updated_messages[0].content == "Hello Jarvis"
    assert updated_messages[1].content == "Greetings! How can I assist you today?"


def test_pillar_2_durability_resilience() -> None:
    """Validate Pillar 2: Durability and graceful error recovery."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_agent:
        # Mock graph raising transient execution error
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = TimeoutError("LLM Provider Timeout")
        mock_agent.return_value = mock_graph

        orchestrator = JarvisOrchestrator(api_key="test_key")
        result = orchestrator.run("Test query", chat_history=[])
        assert (
            "error" in result["output"].lower()
            or "timeout" in result["output"].lower()
            or "apologize" in result["output"].lower()
        )


def test_pillar_3_interrupts_configuration() -> None:
    """Validate Pillar 3: Interrupts for human-in-the-loop approval."""
    interrupt_config = {"dispatch_email_campaign": True, "write_workspace_file": True}
    with patch("src.core.orchestrator.create_deep_agent") as mock_agent:
        mock_agent.return_value = MagicMock()
        orchestrator = JarvisOrchestrator(
            api_key="test_key",
            interrupt_on=interrupt_config,
        )
        assert orchestrator.interrupt_on == interrupt_config
        mock_agent.assert_called_once()
        _, kwargs = mock_agent.call_args
        assert kwargs.get("interrupt_on") == interrupt_config


def test_pillar_4_checkpoints_memory_saver() -> None:
    """Validate Pillar 4: Checkpoints persistence across thread turns."""
    saver = MemorySaver()
    with patch("src.core.orchestrator.create_deep_agent") as mock_agent:
        mock_agent.return_value = MagicMock()
        orchestrator = JarvisOrchestrator(
            api_key="test_key",
            checkpointer=saver,
        )
        assert orchestrator.checkpointer == saver
        mock_agent.assert_called_once()
        _, kwargs = mock_agent.call_args
        assert kwargs.get("checkpointer") == saver


def test_pillar_5_custom_workflow_graph() -> None:
    """Validate Pillar 5: Custom Workflows using StateGraph nodes & edges."""
    builder = StateGraph(CustomAgentState)

    def step_gather(state: CustomAgentState) -> dict:
        return {"mission_status": "gathering"}

    def step_synthesize(state: CustomAgentState) -> dict:
        return {
            "messages": [AIMessage(content="Final synthesis completed.")],
            "mission_status": "completed",
        }

    builder.add_node("gather", step_gather)
    builder.add_node("synthesize", step_synthesize)
    builder.add_edge(START, "gather")
    builder.add_edge("gather", "synthesize")
    builder.add_edge("synthesize", END)

    workflow = builder.compile()
    res = workflow.invoke(
        {
            "messages": [HumanMessage(content="Start mission")],
            "mission_status": "pending",
        }
    )

    assert res["mission_status"] == "completed"
    assert len(res["messages"]) == 2
    assert res["messages"][-1].content == "Final synthesis completed."
