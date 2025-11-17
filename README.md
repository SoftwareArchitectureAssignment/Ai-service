# RAG PDF Chatbot

A FastAPI-based chatbot that uses RAG (Retrieval-Augmented Generation) to answer questions from PDF documents.

## 🚀 Deployment on Render

### Prerequisites
- GitHub account
- Render account (sign up at [render.com](https://render.com/))
- MongoDB Atlas database (get a free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register))
- Google AI API key (get it from [Google AI Studio](https://makersuite.google.com/))

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

### 3. Deploy to Render
1. Push your code to a GitHub repository
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" and select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - **Name**: `rag-pdf-chatbot` (or your preferred name)
   - **Region**: Choose the one closest to your users
   - **Branch**: `main` (or your preferred branch)
   - **Build Command**: `chmod +x render-build.sh && ./render-build.sh`
   - **Start Command**: `uvicorn app.main:app --host=0.0.0.0 --port=$PORT`
6. Add environment variables from your `.env` file
7. Under "Advanced", set:
   - **Build Timeout**: 1800 seconds (30 minutes)
8. Click "Create Web Service"

### 4. Troubleshooting Deployment
If your deployment fails or times out:
1. **Check Logs**: Go to your service on Render and check the logs for specific errors
2. **Common Issues**:
   - **Timeout during build**: The free tier has limited resources. Try deploying during off-peak hours.
   - **Missing dependencies**: Ensure all dependencies are in `requirements.txt`
   - **Environment variables**: Double-check all required environment variables are set
3. **Deploy Manually**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run locally to test
   uvicorn app.main:app --reload
   ```

### 5. Access Your Application
Once deployed, you'll get a URL like `https://rag-pdf-chatbot.onrender.com`

## 📚 API Endpoints

- `POST /upload-pdfs`: Upload PDF files
- `POST /chat`: Send chat messages
- `GET /conversations?user_id=...`: Get conversation history (with pagination `limit`, `page`)
- `GET /pdf-files`: List uploaded PDF files
- `DELETE /pdf-files/{file_id}`: Delete a PDF file
- `GET /health`: Health check endpoint

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

## 📝 Notes
- FAISS index path configurable via env `FAISS_INDEX_DIR` (default `data/faiss_index`)
- The first request after deployment may take longer as Render spins up the instance
- Free tier has limitations on CPU and memory - consider upgrading for production use
- All PDF processing happens in memory - large files may cause timeouts

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
