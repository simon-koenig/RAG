# Components for RAG pipeline
import csv
import pprint
import re

import marqo
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from langchain.text_splitter import (
    CharacterTextSplitter,  # need to install langchain
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer

# Reranker Endpoint
RERANKER_ENDPOINT = "http://10.103.251.104:8883/rerank"
query_threshold = 0.5
num_ref = 3


# Helper functions
def escape_markdown(text):
    """
    Escapes special characters in Markdown text.

    Args:
        text (str): The input text.

    Returns:
        str: The escaped text.
    """
    MD_SPECIAL_CHARS = "\`*_{}[]()#+-.!"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\" + char)
    text = text.strip()
    return text


def token_estimate(text):  # OpenAI suggest a token consists of 3/4 words
    return len(re.findall(r"\w+", text)) * 4 / 3


def chunkText(text, method="recursive", chunk_size=512, chunk_overlap=128):
    """
    Splits the input text into chunks.

    Args:
        text (str): The input text.
        method (str, optional): The method used for chunking. Defaults to "recursive".
        chunk_size (int, optional): The size of each chunk in characters. Defaults to 512.
        chunk_overlap (int, optional): The overlap between chunks in characters. Defaults to 128.

    Returns:
        list: A list of chunks.
    """
    if method == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", ".", "!", "?"],
        )
        splitted_text = splitter.split_text(text)

    elif method == "sentence":
        # We estimate a sentence to have 50 characters on average

        splitter = NLTKTextSplitter(".")
        splitted_text = splitter.split_text(text)

    elif method == "fixed_size":
        splitter = CharacterTextSplitter(
            separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splitted_text = splitter.split_text(text)

    return [chunk for chunk in splitted_text]


def get_reranked_docs(query, docs, reranker_endpoint=RERANKER_ENDPOINT):
    """
    Reranks the documents based on the query.

    Args:
        query (str): The query.
        docs (list): A list of documents.
        reranker_endpoint (str, optional): The reranker endpoint. Defaults to RERANKER_ENDPOINT.

    Returns:
        list: A list of reranked documents.
    """
    data = {
        "query": query,
        "docs": docs,
        "limit": len(docs),
    }

    response = requests.post(reranker_endpoint, json=data)
    reranked_docs = response.json()
    return reranked_docs


def clean_sentence(sentence):
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace("\t", " ")
    sentence = sentence.replace("\r", " ")
    sentence = sentence.strip()
    return sentence
