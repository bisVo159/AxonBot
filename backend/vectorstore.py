from pinecone import Pinecone, ServerlessSpec, Metric
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY, EMBED_MODEL


pc=Pinecone(api_key=PINECONE_API_KEY)
embeddings=HuggingFaceEmbeddings(model_name=EMBED_MODEL)

index_name = "langgraph-rag-index"

def get_retriever():
    if not pc.has_index(index_name):
        pc.create_index(
        name=index_name,
        dimension=768,
        metric=Metric.COSINE,
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
        
    vector_store = PineconeVectorStore(index=index_name, embedding=embeddings)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5}
    )

def add_documents_to_vectorstore(text_content: str):
    if not text_content:
        raise ValueError("Document content cannot be empty.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    documents=text_splitter.create_documents([text_content])
    print(f"Splitting document into {len(documents)} chunks for indexing...")
    

    vector_store = PineconeVectorStore(index=index_name, embedding=embeddings)
    vector_store.add_documents(documents)

    print("successfully added documents into vectorstore")
