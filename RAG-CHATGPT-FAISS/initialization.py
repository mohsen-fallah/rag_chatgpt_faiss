import os
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
# from langchain_openai import OpenAIEmbeddings


def data_loader(file_path):
    loader = CSVLoader(file_path)
    data = loader.load()
    return data


def data_loader_fromweb():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    data = text_splitter.split_documents(docs)
    return data


def hf_embedding(model_name):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    # import os

    # Print the current working directory
    # print("Current working directory:", os.getcwd())
    # if os.path.exists(model_name):
    #     print("*************")
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embedding


def create_vectordb(data, embedding, path_to_save):
    vectordb = FAISS.from_documents(data, embedding)
    vectordb.save_local(path_to_save)
    return vectordb


# def create_vectordb_from_scratch(data_path, embedding_name, vector_db_path):
#     data = data_loader(data_path)
#     embedding = hf_embedding(embedding_name)
#     vectordb = create_vectordb(data, embedding, vector_db_path)
#     return vectordb


def create_vectordb_from_scratchweb(embedding_name, vector_db_path):
    data = data_loader_fromweb()
    # embedding = hf_embedding(embedding_name)
    # openai_api_key = "mykey"
    # embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = create_vectordb(data, embedding_name, vector_db_path)
    return vectordb


def load_vectordb(vectordb_path, data_path, embedding, allow_dangerous_deserialization=False):
    if os.path.exists(vectordb_path):
        vectordb = FAISS.load_local(vectordb_path, embedding,
            allow_dangerous_deserialization=allow_dangerous_deserialization)
    else:
        vectordb = create_vectordb_from_scratchweb(embedding, vectordb_path)
    return vectordb


