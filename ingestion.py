import os

from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# embeddings = OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDINGS_MODEL"))
embeddings = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBEDDINGS_MODEL"))

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} records to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=os.environ.get("VDB_INDEX_NAME"), namespace=os.environ.get("VDB_NAMESPACE"))
    print("*** Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
