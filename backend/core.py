import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama


def run_llm(query: str, chat_history: List[Dict[str, Any]]):
    # embeddings = OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDINGS_MODEL"))
    embeddings = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBEDDINGS_MODEL"))

    docsearch = PineconeVectorStore(index_name=os.environ.get("VDB_INDEX_NAME"), namespace=os.environ.get("VDB_NAMESPACE"), embedding=embeddings)

    # chat = ChatOpenAI(verbose=True, temperature=0)
    chat = ChatOllama(model=os.environ.get("OLLAMA_MODEL"), verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain,
    )

    result = qa.invoke({"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }

    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["result"])
