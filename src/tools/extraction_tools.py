"""
Grounded Information Extraction & Visual Annotation Tools for J.A.R.V.I.S.
Powered by Google's LangExtract library.

Provides grounded entity extraction from unstructured documents, mapping every
extracted entity and attribute back to exact character spans in the source text,
and generating interactive HTML visualizer files within the workspace sandbox.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.tools import BaseTool, tool

from src.assistant.workspace_tools import _resolve_workspace_path

logger = logging.getLogger(__name__)

# Lazy detection of LangExtract
_LANGEXTRACT_AVAILABLE: Optional[bool] = None


def is_langextract_available() -> bool:
    """Check if the LangExtract library is installed and importable."""
    global _LANGEXTRACT_AVAILABLE
    if _LANGEXTRACT_AVAILABLE is None:
        try:
            import langextract as lx  # noqa: F401

            _LANGEXTRACT_AVAILABLE = True
        except ImportError:
            _LANGEXTRACT_AVAILABLE = False
    return _LANGEXTRACT_AVAILABLE


def extract_grounded_entities(
    text: str,
    prompt_description: str,
    examples: Optional[Sequence[Any]] = None,
    model_id: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    language_model_type: Optional[Any] = None,
    temperature: Optional[float] = None,
    max_char_buffer: int = 1000,
    use_schema_constraints: bool = True,
    additional_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract structured, grounded entities from text using LangExtract.

    Maps every extracted entity to its exact start and end character offsets in the source text.
    Returns a standardized dictionary containing extractions, character intervals, and metadata.
    """
    if not is_langextract_available():
        logger.warning("LangExtract is not installed or available in the environment.")
        return {
            "success": False,
            "error": "LangExtract library is unavailable.",
            "extractions": [],
            "text": text,
        }

    if not text or not text.strip():
        return {
            "success": True,
            "extractions": [],
            "text": text,
            "count": 0,
        }

    try:
        import langextract as lx

        kwargs: Dict[str, Any] = {
            "text_or_documents": text,
            "prompt_description": prompt_description,
            "model_id": model_id,
            "max_char_buffer": max_char_buffer,
            "use_schema_constraints": use_schema_constraints,
        }
        if examples is not None:
            kwargs["examples"] = examples
        if api_key is not None:
            kwargs["api_key"] = api_key
        if language_model_type is not None:
            kwargs["language_model_type"] = language_model_type
        if temperature is not None:
            kwargs["temperature"] = temperature
        if additional_context is not None:
            kwargs["additional_context"] = additional_context

        # Invoke LangExtract pipeline
        annotated_doc = lx.extract(**kwargs)

        # Normalize extractions
        extractions_list: List[Dict[str, Any]] = []
        raw_extractions = getattr(annotated_doc, "extractions", []) or []

        for item in raw_extractions:
            char_interval = getattr(item, "char_interval", None)
            interval_dict = None
            if char_interval is not None:
                interval_dict = {
                    "start_pos": getattr(char_interval, "start_pos", None),
                    "end_pos": getattr(char_interval, "end_pos", None),
                }

            alignment_status = getattr(item, "alignment_status", None)
            status_val = alignment_status.value if hasattr(alignment_status, "value") else str(alignment_status)

            extractions_list.append(
                {
                    "extraction_class": getattr(item, "extraction_class", "entity"),
                    "extraction_text": getattr(item, "extraction_text", ""),
                    "char_interval": interval_dict,
                    "attributes": getattr(item, "attributes", {}) or {},
                    "alignment_status": status_val,
                }
            )

        return {
            "success": True,
            "extractions": extractions_list,
            "count": len(extractions_list),
            "annotated_document": annotated_doc,
            "text": text,
        }
    except Exception as exc:
        logger.error(f"LangExtract execution failed: {exc}", exc_info=True)
        return {
            "success": False,
            "error": str(exc),
            "extractions": [],
            "text": text,
        }


def save_grounded_visualization(
    annotated_doc_or_results: Any,
    output_filename: str = "entity_grounding.html",
) -> str:
    """
    Generate an interactive HTML visualization of grounded extractions and persist it in workspace/.

    Returns the filename of the generated visualization inside the workspace sandbox.
    """
    if not is_langextract_available():
        raise RuntimeError("LangExtract is not available for HTML visualization generation.")

    import langextract as lx

    target_path = _resolve_workspace_path(output_filename)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve document object from extraction result dict if passed
    doc = annotated_doc_or_results
    if isinstance(annotated_doc_or_results, dict) and "annotated_document" in annotated_doc_or_results:
        doc = annotated_doc_or_results["annotated_document"]

    html_content = lx.visualize(doc)
    html_str = getattr(html_content, "data", html_content) if hasattr(html_content, "data") else str(html_content)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    logger.info(f"Saved LangExtract visualization to workspace: {target_path.name}")
    return target_path.name


@tool
def extract_structured_entities_tool(text: str, extraction_goal: str) -> str:
    """
    Extract structured facts, entities, and attributes from text with source grounding.
    Use this tool when you need to parse unstructured paragraphs, resumes, research papers,
    or reports into precise, verified entity attributes.
    """
    if not is_langextract_available():
        return json.dumps(
            {
                "status": "unavailable",
                "message": "LangExtract is not installed in this environment.",
                "entities": [],
            }
        )

    res = extract_grounded_entities(
        text=text,
        prompt_description=f"Extract key entities and attributes matching: {extraction_goal}",
    )
    if not res.get("success"):
        return json.dumps(
            {
                "status": "error",
                "message": res.get("error", "Extraction failed"),
                "entities": [],
            }
        )

    return json.dumps(
        {
            "status": "success",
            "count": res.get("count", 0),
            "entities": res.get("extractions", []),
        },
        indent=2,
    )


@tool
def visualize_extractions_tool(filename: str = "entity_grounding.html") -> str:
    """
    Check or confirm the location of the interactive visualizer HTML report inside the workspace.
    Use this tool to inform the user where their interactive extraction highlights are saved.
    """
    try:
        path = _resolve_workspace_path(filename)
        if path.exists():
            return f"Visualization is available in the workspace: '{filename}' ({path.stat().st_size} bytes)."
        return f"Visualization file '{filename}' has not been generated yet."
    except Exception as exc:
        return f"Error verifying visualization path: {exc}"


def get_extraction_tools() -> List[BaseTool]:
    """Retrieve the suite of LangExtract tools for registration in orchestrator and subagents."""
    return [
        extract_structured_entities_tool,
        visualize_extractions_tool,
    ]
