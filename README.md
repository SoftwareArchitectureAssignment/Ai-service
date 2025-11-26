# RAG PDF Chatbot - Optimized File Processing

A FastAPI-based chatbot that uses RAG (Retrieval-Augmented Generation) to answer questions from PDF documents with optimized file handling.


## 🎯 Quick Start

```powershell
# 1. Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
Copy-Item .env.example .env
# Edit .env with your MongoDB URI and Google API Key

# 3. Run
uvicorn app.main:app --reload --port 8003



### 1. Fork and Clone the Repository
```bash
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Set Up Environment Variables
Create a `.env` file in the project root with the following variables:
```
MONGODB_URI=your_mongodb_connection_string
DB_NAME=pdf_chatbot
COLLECTION_NAME=pdf_files
API_KEY=your_google_ai_api_key
MODEL_NAME=Google AI
```

3. **Deploy Manually**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run locally to test
   uvicorn app.main:app --reload --port 8003
   ```

## 🔧 Local Development

1. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Development Server**:
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

## 🤖 RAG Pipeline
This application uses:
- Google's Generative AI for embeddings and chat
- FAISS for vector similarity search (stored under `data/faiss_index/` by default)
- MongoDB for document storage
- FastAPI for the web server


