"""
Tests for JarvisOrchestrator tool collection, prompt binding, and agent construction.
"""

from src.core.orchestrator import JarvisOrchestrator


def test_orchestrator_initialization_defaults():
    """Verify default initialization gathers tools and builds agent executor."""
    orchestrator = JarvisOrchestrator(
        api_provider="OpenRouter", api_key="mock_key_for_testing", model_name="openai/gpt-4o", persona="JARVIS Supreme"
    )
    assert orchestrator.llm is not None
    assert len(orchestrator.tools) >= 15
    assert orchestrator.agent_executor is not None


def test_orchestrator_tool_aggregation_coverage():
    """Verify tools from all 6 subsystems are collected in orchestrator.tools."""
    orchestrator = JarvisOrchestrator(api_provider="OpenAI", api_key="mock_key_for_testing", model_name="gpt-4o")
    tool_names = [t.name for t in orchestrator.tools]

    # Verify presence of tools across all subsystems
    assert "python_interpreter" in tool_names
    assert "duckduckgo_search" in tool_names
    assert "wikipedia_lookup" in tool_names
    assert "read_webpage_content" in tool_names
    assert "analyze_uploaded_images" in tool_names
    assert "write_workspace_file" in tool_names
    assert "read_workspace_file" in tool_names
    assert "generate_excel_spreadsheet" in tool_names
    assert "generate_word_document" in tool_names
    assert "analyze_resume_and_ats" in tool_names
    assert "extract_candidate_skills" in tool_names
    assert "predict_career_salary_and_role" in tool_names
    assert "draft_personalized_outreach" in tool_names
    assert "generate_multi_stage_sequence" in tool_names
    assert "preview_campaign_batch" in tool_names
    assert "dispatch_email_campaign" in tool_names


def test_orchestrator_custom_document_tool_injection():
    """Verify custom document RAG retriever tool is included when provided."""
    from langchain_core.tools import tool

    @tool
    def mock_doc_retriever(query: str) -> str:
        """Mock document search tool."""
        return "Retrieved doc chunk"

    orchestrator = JarvisOrchestrator(
        api_provider="OpenRouter", api_key="mock_key_for_testing", document_tool=mock_doc_retriever
    )
    tool_names = [t.name for t in orchestrator.tools]
    assert "mock_doc_retriever" in tool_names
