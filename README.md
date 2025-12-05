# Jarvis - PDF Question Answering Chatbot

An intelligent PDF chatbot that allows you to upload PDF documents and ask questions about their content. Supports **multiple LLM providers** including OpenAI, OpenRouter, and custom API endpoints.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Multi-Provider Support**: Choose from OpenAI, OpenRouter, or custom API endpoints
- **PDF Text Extraction**: Upload and process PDF documents automatically
- **Intelligent Q&A**: Ask natural language questions about your PDFs
- **Semantic Search**: Uses vector embeddings for accurate context retrieval
- **Multiple AI Models**: Access to GPT-3.5, GPT-4, Claude, and more via OpenRouter
- **Local Embeddings**: Uses HuggingFace embeddings for non-OpenAI providers (no extra API cost)
- **Configurable Settings**: Customize model parameters, chunk sizes, and more
- **Source Citations**: View the exact text chunks used to generate answers

## Supported Providers

| Provider | Models | Embeddings |
|----------|--------|------------|
| **OpenAI** | GPT-3.5 Turbo, GPT-4, GPT-4 Turbo | OpenAI Embeddings |
| **OpenRouter** | GPT-3.5, GPT-4, Claude, Gemini, and 100+ models | HuggingFace (Local) |
| **Custom** | Any OpenAI-compatible API | HuggingFace (Local) |

## Quick Start

### Prerequisites

- Python 3.8 or higher (Python 3.12 recommended)
- An API key from one of the supported providers:
  - [OpenAI](https://platform.openai.com/api-keys)
  - [OpenRouter](https://openrouter.ai/keys) (recommended - access to all models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Jarvis.git
   cd Jarvis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. **Select Provider**
   - Choose your API provider (OpenAI, OpenRouter, or Custom)

2. **Enter API Key**
   - Paste your API key in the input field

3. **Choose Model**
   - Select from available models for your provider
   - Use "Custom model name" checkbox for any model not listed

4. **Upload PDF**
   - Click "Browse files" and select a PDF document
   - Wait for processing (first run downloads the embedding model)

5. **Ask Questions**
   - Type your question in the text input
   - Press Enter to get an AI-generated answer
   - Expand "View source text chunks" to see citations

## Configuration Options

### Model Settings
- **Temperature** (0.0 - 1.0): Controls response creativity
- **Max Tokens** (100 - 2000): Maximum response length

### Advanced Settings
- **Chunk Size**: Characters per text chunk (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 150)
- **Number of Chunks**: Relevant chunks for context (default: 4)

## Using OpenRouter

[OpenRouter](https://openrouter.ai) provides access to 100+ AI models through a single API. Benefits:

- **One API key for all models** - Access GPT-4, Claude, Llama, Gemini, and more
- **Usage-based pricing** - Pay only for what you use
- **No rate limits** - Higher throughput than individual providers
- **Model fallbacks** - Automatic fallback if a model is unavailable

### OpenRouter Model Examples
```
openai/gpt-3.5-turbo
openai/gpt-4-turbo
anthropic/claude-3-sonnet
anthropic/claude-3-opus
google/gemini-pro
meta-llama/llama-3-70b-instruct
```

## Technical Details

### How It Works

1. **PDF Processing**: Extracts text from uploaded PDFs using PyPDF2
2. **Text Chunking**: Splits text into overlapping chunks for better context
3. **Embeddings**: 
   - OpenAI provider: Uses OpenAI's embedding API
   - Other providers: Uses local HuggingFace embeddings (all-MiniLM-L6-v2)
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **Answer Generation**: Retrieves relevant chunks and generates answers using the selected LLM

### Dependencies

- `streamlit` - Web application framework
- `PyPDF2` - PDF text extraction
- `langchain` - LLM orchestration
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community integrations (FAISS, HuggingFace)
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Local embeddings

## Troubleshooting

### "Invalid API key" Error
- Verify your API key is correct
- For OpenRouter, ensure you've selected "OpenRouter" as the provider

### "No text could be extracted"
- The PDF might be image-based (scanned)
- Try a different PDF with selectable text

### Slow First PDF Processing
- First run downloads the HuggingFace embedding model (~90MB)
- Subsequent runs are much faster

### Import Errors
```bash
pip install -r requirements.txt
```

## Project Structure

```
Jarvis/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── LICENSE             # MIT License
├── .gitignore          # Git ignore rules
├── .streamlit/         # Streamlit configuration
│   └── secrets.toml.example
└── logs/               # Application logs
```

## Security

- API keys are entered directly in the UI and not stored permanently
- Never commit API keys to version control
- The `.streamlit/secrets.toml` file is in `.gitignore`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with Streamlit, LangChain, and ❤️**
