import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings


def get_text_chunks(text: str, model_name: str):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, model_name: str, api_key: str | None = None):
    """
    Create or update FAISS vector store with new text chunks.
    If an existing index exists, it will be loaded and updated.
    """
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
    index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
    
    # Check if existing index exists
    if os.path.exists(index_path):
        try:
            # Load existing index and add new chunks
            existing_store = FAISS.load_local(
                settings.FAISS_INDEX_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            # Add new texts to existing store
            existing_store.add_texts(text_chunks)
            existing_store.save_local(settings.FAISS_INDEX_DIR)
            return existing_store
        except Exception as e:
            print(f"Error loading existing index, creating new one: {e}")
    
    # Create new vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(settings.FAISS_INDEX_DIR)
    return vector_store


def load_vector_store(api_key: str | None = None):
    """
    Load existing FAISS vector store.
    Returns None if no index exists.
    """
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


def delete_vectors_by_file_id(file_id: str, api_key: str | None = None):
    """
    Delete vectors associated with a specific file from FAISS index.
    Rebuilds the index without the deleted file's vectors.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        
        index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
        if not os.path.exists(index_path):
            return False
        
        # Load the existing index
        vector_store = FAISS.load_local(
            settings.FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Filter out vectors for the deleted file
        # FAISS stores metadata in docstore, we need to rebuild without this file's vectors
        remaining_docs = []
        for doc_id, doc in vector_store.docstore._dict.items():
            # Check if document metadata contains the file_id
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            if metadata.get('file_id') != file_id:
                remaining_docs.append(doc)
        
        if len(remaining_docs) == 0:
            # If no documents left, delete the index
            if os.path.exists(index_path):
                os.remove(index_path)
            # Also remove the index.pkl file if it exists
            pkl_path = os.path.join(settings.FAISS_INDEX_DIR, "index.pkl")
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
            return True
        
        # Rebuild vector store without the deleted file's vectors
        if remaining_docs:
            new_vector_store = FAISS.from_documents(
                remaining_docs,
                embeddings
            )
            new_vector_store.save_local(settings.FAISS_INDEX_DIR)
            return True
        
        return False
    except Exception as e:
        print(f"Error deleting vectors for file {file_id}: {e}")
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
