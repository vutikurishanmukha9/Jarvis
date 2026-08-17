"""
Universal Document Ingestion & Vector Retrieval Tool for Jarvis.
Supports PDF, Word (.docx), Excel (.xlsx), CSV, TXT, MD, JSON, and code files.
"""

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

try:
    from langchain_core.tools import BaseTool, create_retriever_tool
except ImportError:
    from langchain.tools.retriever import create_retriever_tool  # type: ignore[no-redef]
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def get_files_hash(uploaded_files: List[Any]) -> str:
    """Generate a combined SHA-256 hash of all uploaded files for caching."""
    hasher = hashlib.sha256()
    for file in uploaded_files:
        content = file.getvalue() if hasattr(file, "getvalue") else file.read()
        hasher.update(file.name.encode("utf-8"))
        hasher.update(content)
        if hasattr(file, "seek"):
            file.seek(0)
    return hasher.hexdigest()


def extract_text_from_file(file: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Extract readable text and metadata from a single uploaded file.
    Supports PDF, DOCX, CSV, XLSX, TXT, MD, JSON, PY.
    """
    file_name = file.name
    suffix = Path(file_name).suffix.lower()
    text = ""
    metadata = {"filename": file_name, "type": suffix, "size": len(file.getvalue())}

    try:
        content_bytes = file.getvalue()

        # 1. PDF
        if suffix == ".pdf":
            pdf_reader = PdfReader(io.BytesIO(content_bytes))
            num_pages = len(pdf_reader.pages)
            metadata["pages"] = num_pages
            page_texts = []
            for idx, page in enumerate(pdf_reader.pages):
                extracted = page.extract_text()
                if extracted:
                    page_texts.append(f"--- [Page {idx + 1} of {file_name}] ---\n{extracted}")
            text = "\n\n".join(page_texts)

        # 2. DOCX (Word)
        elif suffix == ".docx":
            try:
                import docx

                doc = docx.Document(io.BytesIO(content_bytes))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                metadata["paragraphs"] = len(paragraphs)
                text = f"=== Word Document: {file_name} ===\n" + "\n\n".join(paragraphs)
            except ImportError:
                text = f"[Warning: python-docx not installed, reading as raw text]\n{content_bytes.decode('utf-8', errors='ignore')}"

        # 3. CSV
        elif suffix == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(content_bytes))
                metadata["rows"] = len(df)
                metadata["columns"] = list(df.columns)
                summary_str = f"=== CSV Dataset: {file_name} ===\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                summary_str += f"Columns: {', '.join(df.columns)}\n\nFirst 25 Rows:\n{df.head(25).to_markdown(index=False)}\n\nStatistical Summary:\n{df.describe(include='all').to_string()}"
                text = summary_str
            except Exception:
                text = content_bytes.decode("utf-8", errors="ignore")

        # 4. Excel (XLSX, XLS)
        elif suffix in [".xlsx", ".xls"]:
            try:
                xls = pd.ExcelFile(io.BytesIO(content_bytes))
                metadata["sheets"] = xls.sheet_names
                sheets_text = [f"=== Excel Workbook: {file_name} ==="]
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    sheets_text.append(
                        f"--- Sheet: {sheet} ({df.shape[0]} rows x {df.shape[1]} cols) ---\n{df.head(20).to_markdown(index=False)}"
                    )
                text = "\n\n".join(sheets_text)
            except Exception as e:
                text = f"Error reading excel: {str(e)}"

        # 5. JSON
        elif suffix == ".json":
            data = json.loads(content_bytes.decode("utf-8", errors="ignore"))
            text = f"=== JSON Document: {file_name} ===\n" + json.dumps(data, indent=2)

        # 6. TXT, Markdown, Python, Code
        else:
            decoded = content_bytes.decode("utf-8", errors="ignore")
            text = f"=== Document: {file_name} ===\n{decoded}"

    except Exception as e:
        logger.error(f"Error extracting text from {file_name}: {str(e)}", exc_info=True)
        text = f"Error reading {file_name}: {str(e)}"

    return text, metadata


def process_documents_and_build_vector_store(
    uploaded_files: List[Any], api_provider: str = "OpenRouter", chunk_size: int = 1000, chunk_overlap: int = 150
) -> Tuple[Optional[FAISS], List[Dict[str, Any]], str]:
    """
    Process all uploaded document files, extract text, split into chunks,
    and construct a FAISS vector store.
    """
    if not uploaded_files:
        return None, [], ""

    all_texts = []
    file_summaries = []

    for file in uploaded_files:
        doc_text, meta = extract_text_from_file(file)
        if doc_text.strip():
            all_texts.append(doc_text)
            file_summaries.append(meta)

    if not all_texts:
        return None, file_summaries, "No readable text could be extracted from uploaded files."

    combined_text = "\n\n" + "=" * 50 + "\n\n".join(all_texts)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_text(combined_text)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    # Embeddings
    try:
        if api_provider == "OpenAI":
            embeddings: Any = OpenAIEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        vector_store = FAISS.from_texts(chunks, embeddings)
        return (
            vector_store,
            file_summaries,
            f"Successfully processed {len(uploaded_files)} file(s) into {len(chunks)} searchable vectors.",
        )
    except Exception as e:
        logger.error(f"Vector store creation error: {str(e)}", exc_info=True)
        return None, file_summaries, f"Embedding creation failed: {str(e)}"


def create_document_retriever_tool(vector_store: FAISS, top_k: int = 4) -> BaseTool:
    """Build a LangChain retriever tool from the FAISS vector store."""
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    return create_retriever_tool(
        retriever,
        "document_search",
        "Searches and retrieves the most relevant excerpts, tables, and sections from all uploaded files (PDF, Word, Excel, CSV, text).",
    )
