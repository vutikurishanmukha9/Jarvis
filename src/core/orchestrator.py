"""
Autonomous Multi-Modal Agent Orchestrator for Jarvis Super-Intelligence.
Powered by Deep Agents & LangGraph with Hierarchical Sub-Agents, Live Thought-Step Tracing,
and Sliding-Window Context Compaction.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

try:
    from deepagents import SubAgent, create_deep_agent
except ImportError:
    create_deep_agent = None  # type: ignore[assignment, misc]
    SubAgent = None  # type: ignore[assignment, misc]

try:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        AgentExecutor = None  # type: ignore[assignment, misc]
        create_tool_calling_agent = None  # type: ignore[assignment, misc]
        ChatPromptTemplate = None  # type: ignore[assignment, misc]
        MessagesPlaceholder = None  # type: ignore[assignment, misc]

from ..assistant.profile_manager import ProfileManager
from ..assistant.workspace_tools import get_workspace_tools
from ..config import PERSONAS, PROVIDERS
from ..modules.career import get_career_tools
from ..modules.outreach import get_outreach_tools
from ..modules.vision import get_and_clear_annotated_images, get_vision_tools
from ..tools.browser_tools import get_browser_tools
from ..tools.python_executor import get_and_clear_figure_buffer, python_interpreter
from ..tools.web_tools import get_web_tools
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class ThoughtStepTracer(BaseCallbackHandler):
    """Callback to capture the agent's live reasoning steps and tool executions."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: List[Dict[str, Any]] = []

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "tool")
        self.steps.append(
            {"type": "tool_start", "tool": tool_name, "input": input_str, "timestamp": time.strftime("%H:%M:%S")}
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.steps.append(
            {
                "type": "tool_end",
                "output": str(output)[:800] + ("..." if len(str(output)) > 800 else ""),
                "timestamp": time.strftime("%H:%M:%S"),
            }
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self.steps.append({"type": "tool_error", "error": str(error), "timestamp": time.strftime("%H:%M:%S")})


class JarvisOrchestrator:
    """
    Enterprise Orchestrator for Jarvis Super-Intelligence.
    Integrates Deep Agents multi-agent harness with specialized domain sub-agents.
    """

    def __init__(
        self,
        api_provider: str = "OpenRouter",
        api_key: str = "",
        model_name: str = "openai/gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        persona: str = "JARVIS Supreme",
        deep_research_mode: bool = False,
        document_tool: Optional[Any] = None,
        checkpointer: Optional[Any] = None,
        interrupt_on: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.api_provider = api_provider
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url or PROVIDERS.get(api_provider, {}).get("base_url")
        self.temperature = temperature
        self.persona = persona
        self.deep_research_mode = deep_research_mode
        self.document_tool = document_tool
        self.checkpointer = checkpointer
        self.interrupt_on = interrupt_on

        # Construct Agent Components
        self.llm = self._init_llm()
        self.tools = self._collect_tools()
        self.subagents = self._build_subagents()
        self.agent_graph = self._build_agent_graph()

    @property
    def agent_executor(self) -> Any:
        """Backwards-compatibility accessor for the compiled agent graph/executor."""
        return self.agent_graph

    def _init_llm(self) -> ChatOpenAI:
        """Initialize the ChatOpenAI client with custom provider base URL if applicable."""
        kwargs: Dict[str, Any] = {
            "model_name": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_retries": 2,
            "timeout": 60,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)

    def _collect_tools(self) -> List[Any]:
        """Aggregate primary orchestrator tools: Python REPL, Web Research, Vision, and Workspace Tools."""
        tools: List[Any] = []

        # 1. Python Code Execution Sandbox
        tools.append(python_interpreter)

        # 2. Deep Web Research Tools
        tools.extend(get_web_tools())

        # 3. Vision Intelligence Tools
        tools.extend(get_vision_tools())

        # 4. Workspace & Artifact Generation Tools
        tools.extend(get_workspace_tools())

        # 5. Career & ATS Optimization Tools
        tools.extend(get_career_tools())

        # 6. HR Outreach & Campaign Tools
        tools.extend(get_outreach_tools())

        # 7. Browser Navigation & Interaction Tools
        tools.extend(get_browser_tools())

        # 8. Universal Document RAG Tool (if documents uploaded)
        if self.document_tool:
            tools.append(self.document_tool)

        return tools

    def _build_subagents(self) -> List[Any]:
        """Assemble domain-specific sub-agents with isolated contexts."""
        if SubAgent is None:
            return []

        subagents: List[Any] = []

        # 1. Career & ATS Optimization Sub-Agent
        career_tools = get_career_tools()
        if career_tools:
            subagents.append(
                SubAgent(
                    name="career_specialist",
                    description=(
                        "Specialized in resume parsing, ATS scoring, JD matching, salary estimation, "
                        "and career guidance."
                    ),
                    tools=career_tools,
                    system_prompt=(
                        "You are the Career Intelligence Specialist for Jarvis. "
                        "Analyze resumes, calculate ATS compatibility scores against job descriptions, "
                        "and generate optimized career recommendations."
                    ),
                )
            )

        # 2. HR Outreach & Recruitment Campaign Sub-Agent
        outreach_tools = get_outreach_tools()
        if outreach_tools:
            subagents.append(
                SubAgent(
                    name="outreach_specialist",
                    description=(
                        "Specialized in candidate prospecting, cold email sequence crafting, "
                        "and HR recruitment outreach."
                    ),
                    tools=outreach_tools,
                    system_prompt=(
                        "You are the HR Outreach Specialist for Jarvis. "
                        "Parse candidate lead sheets, compose high-converting personalized email sequences, "
                        "and manage recruitment pipelines."
                    ),
                )
            )

        # 3. Vision & Image Analysis Sub-Agent
        vision_tools = get_vision_tools()
        if vision_tools:
            subagents.append(
                SubAgent(
                    name="vision_analyst",
                    description="Specialized in YOLOv8 visual object detection, OCR reading, and image analysis.",
                    tools=vision_tools,
                    system_prompt=(
                        "You are the Vision Intelligence Analyst for Jarvis. "
                        "Process uploaded images, detect objects with bounding boxes, extract text via OCR, "
                        "and provide visual intelligence."
                    ),
                )
            )

        # 4. Autonomous Web Navigation & Browser Interaction Sub-Agent
        browser_tools = get_browser_tools()
        if browser_tools:
            subagents.append(
                SubAgent(
                    name="browser_specialist",
                    description=(
                        "Specialized in autonomous web navigation, element clicking, form submission, "
                        "data scraping, and browser viewport interaction."
                    ),
                    tools=browser_tools,
                    system_prompt=(
                        "You are the Autonomous Web Navigation Specialist for Jarvis. "
                        "Navigate web applications, click interactive elements, fill forms, "
                        "extract structured tabular content, and inspect web page states."
                    ),
                )
            )

        # 5. Document RAG & Knowledge Retrieval Sub-Agent
        if self.document_tool:
            subagents.append(
                SubAgent(
                    name="document_researcher",
                    description="Specialized in deep semantic search and contextual question-answering over uploaded files.",
                    tools=[self.document_tool],
                    system_prompt=(
                        "You are the Document Research Specialist for Jarvis. "
                        "Search through uploaded PDFs and document chunks to extract factual answers with citations."
                    ),
                )
            )

        return subagents

    def _assemble_system_prompt(self) -> str:
        """Compose the full system prompt from personas, memory, and research modes."""
        persona_data = PERSONAS.get(self.persona, PERSONAS["JARVIS Supreme"])
        system_prompt = persona_data["prompt"]

        # Inject personal assistant identity and long-term memory
        system_prompt += ProfileManager.get_assistant_system_context()

        if self.deep_research_mode:
            system_prompt += (
                "\n\n[DEEP RESEARCH MODE ACTIVATED]:\n"
                "1. Break the user's query into 2-4 critical research sub-questions.\n"
                "2. Systematically execute multiple tool queries (cross-referencing documents, web search, wikipedia, or python analysis).\n"
                "3. Synthesize the findings into an executive-level report with structured headings, key takeaways, and explicit citations."
            )

        return system_prompt

    def _build_agent_graph(self) -> Any:
        """Construct the agent runtime (Deep Agents harness with fallback to classic AgentExecutor)."""
        system_prompt = self._assemble_system_prompt()

        # Primary: Deep Agents Graph
        if create_deep_agent is not None:
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.llm,
                    "tools": self.tools,
                    "system_prompt": system_prompt,
                    "subagents": self.subagents if self.subagents else None,
                }
                if self.checkpointer is not None:
                    kwargs["checkpointer"] = self.checkpointer
                if self.interrupt_on is not None:
                    kwargs["interrupt_on"] = self.interrupt_on
                return create_deep_agent(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to assemble DeepAgent graph, falling back to legacy executor: {e}")

        # Fallback: Classic Tool-Calling AgentExecutor
        if create_tool_calling_agent is not None and ChatPromptTemplate is not None:
            all_tools = list(self.tools)
            all_tools.extend(get_career_tools())
            all_tools.extend(get_outreach_tools())

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_tool_calling_agent(self.llm, all_tools, prompt)
            return AgentExecutor(
                agent=agent, tools=all_tools, verbose=True, handle_parsing_errors=True, max_iterations=10
            )

        raise RuntimeError("Neither create_deep_agent nor create_tool_calling_agent is available in the environment.")

    def run(self, user_input: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """
        Execute the agent pipeline on user input, returning output, steps, charts, and annotated images.
        Automatically applies sliding-window context compression to avoid token exhaustion.
        """
        tracer = ThoughtStepTracer()
        bounded_history = SessionManager.prune_context_window(chat_history)
        output_text = ""

        try:
            # Case 1: LangGraph / Deep Agents Compiled Graph
            if hasattr(self.agent_graph, "invoke") and not isinstance(self.agent_graph, AgentExecutor):
                messages: List[BaseMessage] = list(bounded_history)
                messages.append(HumanMessage(content=user_input))

                response = self.agent_graph.invoke(
                    {"messages": messages},
                    config={"callbacks": [tracer]},
                )

                # Extract final output from returned messages
                if isinstance(response, dict) and "messages" in response:
                    resp_messages = response["messages"]
                    for msg in reversed(resp_messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            if isinstance(msg.content, str):
                                output_text = msg.content
                            elif isinstance(msg.content, list):
                                output_text = "\n".join(
                                    block.get("text", "") if isinstance(block, dict) else str(block)
                                    for block in msg.content
                                )
                            break
                    if not output_text and resp_messages:
                        last_msg = resp_messages[-1]
                        output_text = getattr(last_msg, "content", str(last_msg))
                elif isinstance(response, str):
                    output_text = response
                elif hasattr(response, "content"):
                    output_text = str(response.content)
                else:
                    output_text = str(response)

            # Case 2: Classic LangChain AgentExecutor
            else:
                response = self.agent_graph.invoke(
                    {"input": user_input, "chat_history": bounded_history},
                    config={"callbacks": [tracer]},
                )
                output_text = response.get("output", "I processed your request, but generated an empty response.")

            if not output_text:
                output_text = "I processed your request, but generated an empty response."

        except Exception as e:
            logger.error(f"Agent Execution Error: {str(e)}", exc_info=True)
            output_text = f"An error occurred during agent processing: {str(e)}"
            tracer.steps.append({"type": "error", "error": str(e), "timestamp": time.strftime("%H:%M:%S")})

        # Collect any generated plots and annotated images
        figures = get_and_clear_figure_buffer()
        annotated_imgs = get_and_clear_annotated_images()

        return {
            "output": output_text,
            "steps": tracer.steps,
            "figures": figures,
            "annotated_images": annotated_imgs,
        }
