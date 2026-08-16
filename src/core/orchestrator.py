"""
Autonomous Multi-Modal Agent Orchestrator for Jarvis Super-Intelligence.
Coordinates tools (Document RAG, Vision, Python Sandbox, Web Research) with Thought-Step Tracing.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
try:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    from langchain.agents import create_tool_calling_agent, AgentExecutor

from ..config import PERSONAS, PROVIDERS
from ..tools.web_tools import get_web_tools
from ..tools.python_executor import python_interpreter, get_and_clear_figure_buffer
from ..modules.vision import get_vision_tools, get_and_clear_annotated_images
from ..assistant.workspace_tools import get_workspace_tools
from ..assistant.profile_manager import ProfileManager
from ..modules.career import get_career_tools
from ..modules.outreach import get_outreach_tools

logger = logging.getLogger(__name__)

class ThoughtStepTracer(BaseCallbackHandler):
    """Callback to capture the agent's live reasoning steps and tool executions."""
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "tool")
        self.steps.append({
            "type": "tool_start",
            "tool": tool_name,
            "input": input_str,
            "timestamp": time.strftime("%H:%M:%S")
        })

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.steps.append({
            "type": "tool_end",
            "output": str(output)[:800] + ("..." if len(str(output)) > 800 else ""),
            "timestamp": time.strftime("%H:%M:%S")
        })

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self.steps.append({
            "type": "tool_error",
            "error": str(error),
            "timestamp": time.strftime("%H:%M:%S")
        })

class JarvisOrchestrator:
    """Orchestrates Jarvis agent execution, tool integration, and thought tracking."""

    def __init__(
        self,
        api_provider: str = "OpenRouter",
        api_key: str = "",
        model_name: str = "openai/gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        persona: str = "JARVIS Supreme",
        deep_research_mode: bool = False,
        document_tool: Optional[Any] = None
    ):
        self.api_provider = api_provider
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url or PROVIDERS.get(api_provider, {}).get("base_url")
        self.temperature = temperature
        self.persona = persona
        self.deep_research_mode = deep_research_mode
        self.document_tool = document_tool

        # Construct Agent
        self.llm = self._init_llm()
        self.tools = self._collect_tools()
        self.agent_executor = self._build_executor()

    def _init_llm(self) -> ChatOpenAI:
        """Initialize the ChatOpenAI client with custom provider base URL if applicable."""
        kwargs: Dict[str, Any] = {
            "model_name": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_retries": 2,
            "timeout": 60
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)

    def _collect_tools(self) -> List[Any]:
        """Aggregate all available tools: web search, python REPL, vision, workspace files, and documents."""
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
        
        # 7. Universal Document RAG Tool (if documents uploaded)
        if self.document_tool:
            tools.append(self.document_tool)
            
        return tools

    def _build_executor(self) -> AgentExecutor:
        """Build the LangChain tool-calling AgentExecutor with personal assistant context."""
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

    def run(self, user_input: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """
        Execute the agent pipeline on user input, returning output, steps, charts, and annotated images.
        """
        tracer = ThoughtStepTracer()
        try:
            response = self.agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history},
                config={"callbacks": [tracer]}
            )
            output_text = response.get("output", "I processed your request, but generated an empty response.")
        except Exception as e:
            logger.error(f"Agent Execution Error: {str(e)}", exc_info=True)
            output_text = f"An error occurred during agent processing: {str(e)}"
            tracer.steps.append({
                "type": "error",
                "error": str(e),
                "timestamp": time.strftime("%H:%M:%S")
            })

        # Collect any generated plots and annotated images
        figures = get_and_clear_figure_buffer()
        annotated_imgs = get_and_clear_annotated_images()

        return {
            "output": output_text,
            "steps": tracer.steps,
            "figures": figures,
            "annotated_images": annotated_imgs
        }
