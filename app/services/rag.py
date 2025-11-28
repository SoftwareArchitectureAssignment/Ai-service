import os
import logging
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from app.core.config import settings
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def get_text_chunks(text: str, model_name: str):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, model_name: str, api_key: str | None = None, file_id: str | None = None, url_hash: str | None = None):
    
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
    index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
    
    docs_with_metadata = []
    for chunk in text_chunks:
        doc = Document(
            page_content=chunk,
            metadata={
                "file_id": file_id or "unknown",
                "url_hash": url_hash or "unknown",
                "created_at": str(datetime.now())
            }
        )
        docs_with_metadata.append(doc)
    
    if os.path.exists(index_path):
        try:
            existing_store = FAISS.load_local(
                settings.FAISS_INDEX_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            # Add new documents to existing store
            existing_store.add_documents(docs_with_metadata)
            existing_store.save_local(settings.FAISS_INDEX_DIR)
            return existing_store
        except Exception as e:
            logger.error(f"Error loading existing index, creating new one: {e}", exc_info=True)
    
    # Create new vector store with documents
    vector_store = FAISS.from_documents(docs_with_metadata, embedding=embeddings)
    vector_store.save_local(settings.FAISS_INDEX_DIR)
    return vector_store


def load_vector_store(api_key: str | None = None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    
    index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return None
    
    try:
        return FAISS.load_local(
            settings.FAISS_INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def delete_vectors_by_file_id(file_id: str, file_id_AI_service: str, api_key: str | None = None):
   
    try:
        
        key = api_key or settings.API_KEY
        
        if not key:
            raise HTTPException(status_code=400, detail="API_KEY not provided and not found in settings")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=key,
        )
        index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
        if not os.path.exists(index_path):
            logger.info(f"No FAISS index found for deletion of file {file_id}")
            return True
        
        try:
            vector_store = FAISS.load_local(
                settings.FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}", exc_info=True)
            return False
        
        docs_to_keep = []
        docs_deleted_count = 0
        
        if hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
            print("Iterating over documents in FAISS index for deletion")

            for doc_id, doc in vector_store.docstore._dict.items():
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                if str(metadata.get('file_id')) == str(file_id_AI_service):
                    docs_deleted_count += 1
                    logger.debug(f"Marking document {doc_id} for deletion (file_id: {file_id})")
                    continue
                
                # Otherwise, keep this document
                docs_to_keep.append(doc)
        
        logger.info(f"Found {docs_deleted_count} documents to delete for file {file_id}")
        
        # Handle different cases
        if docs_deleted_count == 0:
            logger.info(f"No embeddings found for file {file_id}")
            return True
        if len(docs_to_keep) == 0:
            # All documents were deleted, remove the entire index
            logger.info(f"All documents deleted, removing FAISS index files")
            index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
            pkl_path = os.path.join(settings.FAISS_INDEX_DIR, "index.pkl")
            docstore_path = os.path.join(settings.FAISS_INDEX_DIR, "docstore.pkl")
            
            for path in [index_path, pkl_path, docstore_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.debug(f"Deleted index file: {path}")
                    except Exception as e:
                        logger.warning(f"Could not delete index file {path}: {e}")
            
            return True
        
        logger.info(f"Rebuilding FAISS index with {len(docs_to_keep)} remaining documents")
        try:
            new_vector_store = FAISS.from_documents(
                docs_to_keep,
                embeddings
            )
            new_vector_store.save_local(settings.FAISS_INDEX_DIR)
            logger.info(f"Successfully rebuilt FAISS index, deleted {docs_deleted_count} embeddings for file {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"Error deleting vectors for file {file_id}: {e}", exc_info=True)
        return False


def get_conversational_chain(model_name: str, api_key: str | None = None):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in context, say: "answer is not available in the context".\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """

    if model_name == "Google AI":
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3,
            google_api_key=api_key,
        )
    else:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3,
            google_api_key=api_key,
        )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain
