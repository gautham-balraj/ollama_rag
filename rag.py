import sys
import os
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import argparse
import getpass
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain_community.embeddings import CohereEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec
import time
from langchain.vectorstores import Pinecone as Pinecone_lang
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

os.environ["COHERE_API_KEY"] = #cohere api key


def index_creation(name):
    serverless = True
    PINECONE_API_KEY = # pinecone api key 
    pc = Pinecone(api_key=PINECONE_API_KEY)
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    spec = ServerlessSpec(cloud="aws", region="us-west-2")
    index_name = name

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in existing_indexes:
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=384,  # dimensionality of minilm
            metric="cosine",
            spec=spec,
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        # connect to index
        index = pc.Index(index_name)
        time.sleep(1)


def ollama(model: str):
    try:
        llm = Ollama(
            model=model,
            # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.9,
        )
        print(f"ollama -- {model} loaded")
        return llm
    except Exception as e:
        print("ollma failed to load the model")


def main():
    parser = argparse.ArgumentParser(
        description="read the url and model for the ollama"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://example.com",
        required=True,
        help="The URL to filter out.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        required=True,
        help="The model to use for the ollama",
    )
    args = parser.parse_args()
    url = args.url

    # text extracttio and embedding

    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=160,
    )
    chunks = text_splitter.split_documents(data)
    print("Number if chunks from the provided URL:", len(chunks))
    # gpt4all_embeddings = GPT4AllEmbeddings()
    index_name = "semantic-search-fast"
    index_creation("semantic-search-fast")
    cohere_embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    # api_key = getpass.getpass(prompt="Please enter your API key: ")
    docsearch = Pinecone_lang.from_documents(
        chunks, cohere_embeddings, index_name=index_name
    )

    # using chroma db
    # vectorstore = Chroma.from_documents(documents=chunks,
    #                                     embedding=GPT4AllEmbeddings(),
    #                                     persist_directory="./chroma_db")
    # retriever = vectorstore.as_retriever()

    # using pinecone serverless
    retreiver = docsearch.as_retriever()
    return retreiver


def final(query, retreiver):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ollama("mistral")
    chain = (
        RunnableParallel({"context": retreiver, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    res = chain.invoke(query)
    return res


if __name__ == "__main__":
    retreiver = main()
    print("pinecone index created and vectors uploaded")
    while True:
        query = input("Enter the question or to exit enter bye: ")
        if query == "bye":
            break
        print("generting response...")
        response = final(query, retreiver)
        print(response)
