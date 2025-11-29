# 📄 Jarvis - PDF Question Answering Chatbot

An intelligent PDF chatbot powered by OpenAI and LangChain that allows you to upload PDF documents and ask questions about their content.

## ✨ Features

- **PDF Text Extraction**: Upload and process PDF documents
- **Intelligent Q&A**: Ask natural language questions about your PDFs
- **Semantic Search**: Uses vector embeddings for accurate context retrieval
- **Multiple AI Models**: Support for GPT-3.5 Turbo, GPT-4, and GPT-4 Turbo
- **Configurable Settings**: Customize model parameters, chunk sizes, and more
- **Source Citations**: View the exact text chunks used to generate answers
- **Secure API Key Management**: Multiple options for API key configuration

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Jarvis
   ```

2. **Create a virtual environment** (recommended)
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

### Configuration

You have three options for configuring your OpenAI API key:

#### Option 1: Streamlit Secrets (Recommended for Production)

1. Create the secrets directory:
   ```bash
   mkdir .streamlit
   ```

2. Copy the example secrets file:
   ```bash
   copy .streamlit\secrets.toml.example .streamlit\secrets.toml
   ```

3. Edit `.streamlit/secrets.toml` and add your API key:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-api-key-here"
   ```

#### Option 2: Environment Variable

Set the environment variable in your terminal:

```bash
# Windows
set OPENAI_API_KEY=sk-your-actual-api-key-here

# macOS/Linux
export OPENAI_API_KEY=sk-your-actual-api-key-here
```

#### Option 3: UI Input (Development/Testing)

Simply enter your API key in the sidebar when you run the application.

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage

1. **Enter API Key** (if not using secrets.toml)
   - Enter your OpenAI API key in the sidebar

2. **Upload PDF**
   - Click "Browse files" in the sidebar
   - Select a PDF document from your computer

3. **Wait for Processing**
   - The app will extract and process the text
   - You'll see a success message when ready

4. **Ask Questions**
   - Type your question in the text input
   - Press Enter to get an AI-generated answer
   - View source chunks to see where the answer came from

5. **Customize Settings** (Optional)
   - Select different AI models (GPT-3.5, GPT-4)
   - Adjust temperature for creativity
   - Configure chunk sizes and retrieval parameters

## 🎛️ Advanced Settings

### Model Selection
- **GPT-3.5 Turbo**: Fast and cost-effective
- **GPT-4**: More accurate and nuanced responses
- **GPT-4 Turbo**: Latest model with improved performance

### Temperature
- **0.0**: Deterministic, focused answers
- **0.5**: Balanced creativity and accuracy
- **1.0**: More creative and varied responses

### Chunk Configuration
- **Chunk Size**: Number of characters per text chunk (default: 1000)
- **Chunk Overlap**: Overlap between chunks for context continuity (default: 150)
- **Number of Chunks**: How many relevant chunks to retrieve (default: 4)

## 🛠️ Troubleshooting

### "No text could be extracted from the PDF"
- **Cause**: The PDF might be image-based or corrupted
- **Solution**: Try using a different PDF or use OCR to convert image-based PDFs to text

### "Invalid API key"
- **Cause**: Incorrect or expired OpenAI API key
- **Solution**: Verify your API key at https://platform.openai.com/api-keys

### "Rate limit exceeded"
- **Cause**: Too many requests to OpenAI API
- **Solution**: Wait a few minutes and try again, or upgrade your OpenAI plan

### Import Errors
- **Cause**: Missing dependencies
- **Solution**: Run `pip install -r requirements.txt` again

### "Module not found" errors
- **Cause**: Virtual environment not activated
- **Solution**: Activate your virtual environment before running the app

## 📦 Dependencies

- **streamlit**: Web application framework
- **PyPDF2**: PDF text extraction
- **langchain**: LLM orchestration framework
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-community**: Community integrations
- **openai**: OpenAI API client
- **faiss-cpu**: Vector similarity search
- **python-dotenv**: Environment variable management

## 🔒 Security

- Never commit your `.streamlit/secrets.toml` file (it's in `.gitignore`)
- Never share your OpenAI API key publicly
- Use environment variables or secrets management in production
- Regularly rotate your API keys

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💡 Tips

- **Better Questions**: Ask specific questions for better answers
- **Context Matters**: The AI can only answer based on the PDF content
- **Chunk Size**: Larger chunks provide more context but may be less precise
- **Model Selection**: Use GPT-4 for complex documents requiring deep understanding

## 🐛 Known Limitations

- Cannot process image-based PDFs without OCR
- Limited to text content (doesn't analyze images, charts, or tables)
- Answer quality depends on PDF text extraction quality
- Large PDFs may take longer to process

## 📧 Support

For issues and questions, please open an issue on the GitHub repository.

---

**Built with ❤️ using Streamlit, LangChain, and OpenAI**
