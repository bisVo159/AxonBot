from pinecone import Pinecone, ServerlessSpec, Metric
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY, EMBED_MODEL, PINECONE_INDEX_NAME


pc=Pinecone(api_key=PINECONE_API_KEY)
embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL)

try:
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric=Metric.COSINE,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
except Exception as e:
    print(f"Index creation failed: {e}. Trying to recreate...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric=Metric.COSINE,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index=pc.Index(PINECONE_INDEX_NAME)
    
vector_store = PineconeVectorStore(index, embedding)

def get_retriever():   
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5,"fetch_k": 20}
    )

def add_document_to_vectorstore(text_content: str):
    if not text_content:
        raise ValueError("Document content cannot be empty.")
    
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print(f"No existing namespace to clear: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    documents=text_splitter.create_documents([text_content])
    print(f"Splitting document into {len(documents)} chunks for indexing...")
    
    vector_store.add_documents(documents)

    print("successfully added documents into vectorstore")
