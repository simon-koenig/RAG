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


def write_context_relevance_to_csv(filename, scores, evaluator):
    """
    Writes context relevance scores to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        scores (dict): A dictionary of query-context relevance scores.
        evaluator (str): The name of the evaluator.

    Returns:
        None
    """
    # Open the CSV file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the title row
        title = f"Query-Context-Relevance-{evaluator}"
        writer.writerow([title])

        # Assuming all contexts arrays have the same length
        num_contexts = len(next(iter(scores.values())))
        # Write the header row
        header = ["Query"] + [f"Context_{i}_Score" for i in range(num_contexts)]
        writer.writerow(header)

        # Write the data rows
        for query, contexts_scores in scores.items():
            row = [
                query
            ] + contexts_scores.tolist()  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


def write_faithfulness_to_csv(filename, scores, evaluator):
    """
    Writes faithfulness scores to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        scores (dict): A dictionary of answer-context faithfulness scores.
        evaluator (str): The name of the evaluator.

    Returns:
        None
    """
    # Open the CSV file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the title row
        title = f"Answer-Context-Faithfulness-{evaluator}"
        writer.writerow([title])

        # Assuming all contexts arrays have the same length
        num_contexts = len(next(iter(scores.values())))
        # Write the header row
        header = ["Answer"] + [f"Context_{i}_Score" for i in range(num_contexts)]
        writer.writerow(header)

        # Write the data rows
        for llm_answer, contexts_scores in scores.items():
            row = [
                llm_answer
            ] + contexts_scores.tolist()  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


def write_answer_relevance_to_csv(filename, scores, evaluator):
    """
    Writes answer relevance scores to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        scores (dict): A dictionary of query-answer relevance scores.
        evaluator (str): The name of the evaluator.

    Returns:
        None
    """
    # Open the CSV file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the title row
        title = f"Query-Answer-Relevance-{evaluator}"
        writer.writerow([title])

        # Write the header row
        header = ["Query", "Answer_Score"]
        writer.writerow(header)

        # Write the data rows
        for query, llm_answer_score in scores.items():
            row = [query, llm_answer_score]  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


def write_correctness_to_csv(filename, scores, evaluator):
    """
    Writes correctness scores to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        scores (dict): A dictionary of ground-truth answer correctness scores.
        evaluator (str): The name of the evaluator.

    Returns:
        None
    """
    # Open the CSV file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the title row
        title = f"Ground-Truth-Answer-Correctness-{evaluator}"
        writer.writerow([title])

        # Write the header row
        header = ["Ground_Truth", "Answer_Score"]
        writer.writerow(header)

        # Write the data rows
        for answer, llm_answer_score in scores.items():
            row = [answer, llm_answer_score]  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


class VectorStore:
    """
    A class representing a vector store for indexing and retrieving documents.

    Attributes:
        mq (marqo.Client): The Marqo client for interacting with the vector store.
        indexName (str): The name of the current index.
        model (str): The model used for generating sentence embeddings.

    Methods:
        __init__(self, url): Initializes a VectorStore object with the specified Marqo client URL.
        connectIndex(self, indexName): Connects to the specified index.
        createIndex(self, indexName, settings): Creates a new index with the specified name and settings.
        deleteIndex(self, indexName): Deletes the index with the specified name.
        indexDocument(self, document): Indexes a single document.
        indexDocuments(self, documents, maxDocs=1000000): Indexes multiple documents.
        retrieveDocuments(self, query, k, rerank=False, prepost_context=False): Retrieves documents based on a query.
        getBackground(self, query, num_ref, lang, rerank=False, prepost_context=False): Retrieves background information based on a query.
    """

    def __init__(self, url):
        self.mq = marqo.Client(url=url)
        self.indexName = None
        self.model = (
            "flax-sentence-embeddings/all_datasets_v4_mpnet-base"  # Default model
        )

    def connectIndex(self, indexName):
        """
        Connects to the specified index.

        Args:
            indexName (str): The name of the index to connect to.

        Returns:
            None
        """
        # Check if the index exists and connect to it
        indexName = indexName.lower()
        if indexName in [
            key.get("indexName") for key in self.mq.get_indexes()["results"]
        ]:
            self.indexName = indexName
            print(f"Index connected: {indexName} ")
        else:
            print(
                f"Index not found: {indexName}. Beware the index name must be lower case."
            )

    def createIndex(self, indexName, settings):
        """
        Creates a new index with the specified name and settings.

        Args:
            indexName (str): The name of the index to create.
            settings (dict): The settings for the new index. It should have the following keys:
            - 'split_method': The method used for splitting text.
            - 'distance_metric': The distance metric used for similarity calculations.
            - 'model': The model used for generating sentence embeddings.

        Returns:
            None
        """
        # Create a new index with name indexName
        indexName = indexName.lower()
        current_indexes = [d["indexName"] for d in self.mq.get_indexes()["results"]]
        if indexName in current_indexes:
            print(f"Index already exists: {indexName} ")
            # Set indexName as the current index
            print(f"Defaulting to index connection. Index connected: {indexName} ")
            self.indexName = indexName
            return
        try:
            self.split_method = settings["split_method"]
            self.distance_metric = settings["distance_metric"]
            self.model = settings["model"]
        except:
            print(
                f"Settings could not be parsed to create a new index with name: {indexName}"
            )

        try:
            index_settings = {
                "model": self.model,
                "normalizeEmbeddings": True,
                "textPreprocessing": {
                    "splitLength": 2,
                    "splitOverlap": 0,
                    "splitMethod": self.split_method,
                },
                "annParameters": {
                    "spaceType": self.distance_metric,
                    # Tinker with this. Try increasing efConstruction along with m for better recall
                    # https://www.pinecone.io/learn/series/faiss/hnsw/
                    "parameters": {"efConstruction": 512, "m": 16},
                },
            }
            print("Indexname: ", indexName)
            self.mq.create_index(indexName, settings_dict=index_settings)
            # mq.create_index(indexName,  model='flax-sentence-embeddings/all_datasets_v4_mpnet-base')
            print(f"New index created: {indexName}  ")
            # Set indexName as the current index
            self.indexName = indexName
        except:
            print(f"Failed to created new index: {indexName} with settings: {settings}")

    def deleteIndex(self, indexName):
        """
        Deletes an index by indexName.

        Args:
            indexName (str): The name of the index to be deleted.

        Returns:
            None
        """
        try:
            # now remove the marqo index
            self.mq.delete_index(indexName)
            print(f"Successfully deleted Index: {indexName}")
        except:
            print("Unable to delete: " + indexName)

    def indexDocument(self, document):
        """
        Indexes a document in the search engine.

        Args:
            document (dict): A dictionary representing the document to be indexed. It should have the following keys:
                - 'chunk_id': The ID of the document chunk.
                - 'text': The main text of the document.
                - 'pre_context': The pre-context of the document.
                - 'post_context': The post-context of the document.

        Returns:
            None
        """
        chunk_id = document["chunk_id"]
        main_text = document["text"]
        pre_context = document["pre_context"]
        post_context = document["post_context"]

        try:
            self.mq.index(self.indexName).add_documents(
                [
                    {
                        "text": main_text,
                        "chunk": chunk_id,
                        "pre_context": pre_context,
                        "post_context": post_context,
                    }
                ],
                tensor_fields=["text"],
            )

        except:
            print(
                f"Ingest error for passage with id: {chunk_id},"
                "Documents has to be a dict with keys 'id' and 'text'"
            )

    def indexDocuments(self, documents, maxDocs=1000000):
        """
        Indexes a list of documents.

        Args:
            documents (list): A list of dictionaries representing the documents to be indexed.
                Each dictionary should have the following keys: 'id', 'text'
            maxDocs (int, optional): The maximum number of documents to index. Defaults to 1000000.

        Raises:
            Exception: If there is an error in indexing the corpus.

        Returns:
            None
        """
        n_docs = min(len(documents), maxDocs)

        try:
            for i in range(n_docs):
                chunk_id = documents[i]["id"]
                main_text = documents[i]["text"]
                print(f" Current i = {i}")
                # Exceptions are needed the first two and last two documents
                if i == 0:
                    pre_context = ""
                    post_context = documents[i + 1]["text"] + documents[i + 2]["text"]
                elif i == 1:
                    pre_context = documents[i - 1]["text"]
                    post_context = documents[i + 1]["text"] + documents[i + 2]["text"]
                elif i == n_docs - 1:
                    pre_context = documents[i - 1]["text"] + documents[i - 2]["text"]
                    post_context = ""
                elif i == n_docs - 2:
                    pre_context = documents[i - 1]["text"] + documents[i - 2]["text"]
                    post_context = documents[i + 1]["text"]
                else:
                    pre_context = documents[i - 1]["text"] + documents[i - 2]["text"]
                    post_context = documents[i + 1]["text"] + documents[i + 2]["text"]

                document = {
                    "text": main_text,
                    "chunk_id": chunk_id,
                    "pre_context": pre_context,
                    "post_context": post_context,
                }
                print(f"Indexing document: {document}")
                print(f" Successfully indexed document number: {i}")
                self.indexDocument(document)
        except:
            print(
                f"Error in indexing corpus for index: {self.indexName}, "
                " documents has to be a list of dictionaries with keys 'id' and 'text'"
            )

    def retrieveDocuments(self, query, k, rerank=False, prepost_context=False):
        """
        Retrieve top k documents from indexName based on the given query.

        Args:
            query (str): The query string.
            k (int): The number of documents to retrieve.
            rerank (bool, optional): Whether to perform reranking. Defaults to False.
            prepost_context (bool, optional): Whether to include pre and post context. Defaults to False.

        Returns:
            tuple: A tuple containing two lists of the same length - contexts and ids.
                - contexts (list): The retrieved document texts.
                - ids (list): The IDs of the retrieved documents.
        """
        response = self.mq.index(self.indexName).search(
            q=query,  # Query string
            limit=k,  # Number of documents to retrieve
            # attributes_to_retrieve=["text", "chunk"],  # Attributes to retrieve
        )

        contexts = [response["hits"][i]["text"] for i in range(len(response["hits"]))]
        ids = [response["hits"][i]["chunk"] for i in range(len(response["hits"]))]

        return contexts, ids

    def getBackground(self, query, num_ref, lang, rerank=False, prepost_context=False):
        """
        Retrieves background information and contexts based on a given query. Filterstring under developement

        Args:
            query (str): The query string.
            num_ref (int): The number of documents to retrieve.
            lang (str): The language to filter the documents.
            rerank (bool, optional): Whether to rerank the search results. Defaults to False.
            prepost_context (bool, optional): Whether to include pre and post context in the background. Defaults to False.

        Returns:
            tuple: A tuple containing the background string, a list of contexts, and a list of context IDs.
        """

        # Retrieve top k documents from indexName based on query
        # Modify filter string to include language
        # filterstring = f"lang:{lang}"  # Add language to filter string
        # print(f"Filterstring: {filterstring}")

        # Semantic Search
        response_sem = self.mq.index(self.indexName).search(
            q=query,  # Query string
            limit=num_ref,  # Number of documents to retrieve
            # attributes_to_retrieve=["text", "chunk"],
            # filter_string=filterstring,  # Filter in db, e.g. for lang  # Attributes to retrieve, explicit is faster
        )

        # Lexial Search
        response_lex = self.mq.index(self.indexName).search(
            q=query,  # Query string
            limit=num_ref,  # Number of documents to retrieve
            # attributes_to_retrieve=["text", "chunk"],  # Attributes to retrieve, explicit is faster
            # filter_string=filterstring,  # Filter in db, e.g. for lang
            search_method="LEXICAL",
        )

        # Construct contexts and background
        background = ""
        contexts = []
        context_ids = []
        full_results = []
        plain_text_results = []

        # If rerank is false, combine semantic and lexical search results
        if rerank is False:
            # Combine semantic and lexical search results
            for response in [response_sem, response_lex]:
                # Construct Background
                for i in range(len(response["hits"])):
                    # Get current context
                    text = response["hits"][i]["text"]
                    # title = response["hits"][i]["title"]
                    # link_url = response["hits"][i]["link_url"]
                    score = response["hits"][i]["_score"]
                    # Augment context with title, link and score
                    context = f"Text: {text} "
                    contexts.append(context)
                    # Get current context id
                    Id = response["hits"][i]["chunk"]
                    context_ids.append(Id)

                    ## pprint.pprint(contexts)
                    # If pre post context is true, add pre and post context to background
                    if prepost_context:
                        pre_context = response["hits"][i]["pre_context"]
                        post_context = response["hits"][i]["post_context"]
                        background += (
                            pre_context + " " + context + " " + post_context + " "
                        )
                    else:  # Else just add context
                        background += context + " "

        # If rerank is true, rerank the results
        if rerank is True:
            print(f" Semantic Search Results: {response_sem}")
            print(f" Lexical Search Results: {response_lex}")
            for response in response_sem["hits"]:
                full_results.append(response)
                plain_text_results.append(response["text"])
            for response in response_lex["hits"]:
                full_results.append(response)
                plain_text_results.append(response["text"])

            headers = {
                "Content-Type": "application/json",
            }
            data = {
                "query": query,
                "raw_results": plain_text_results,
                "num_results": num_ref * 2,  # To get all results
            }

            # Get response from reranker
            response = requests.post(RERANKER_ENDPOINT, headers=headers, json=data)
            # Safe guard
            print(f"Response: {response}")
            if not response.status_code == 200:
                print("Reranker failed")
                raise ("Reranker failed")
                return
            # Unpack json response
            reranked_results = response.json()
            # Iterate over reranked results
            for reranked_res in reranked_results:
                current_index = reranked_res["result_index"]
                print(f"Current Index: {current_index}")
                # Get current highest ranked context
                hit = full_results[current_index]
                text = hit["text"]
                # title = hit["title"]
                # link_url = hit["link_url"]
                score = hit["_score"]
                # Augment context with title, link and score
                context = f" Text: {text} "
                contexts.append(context)
                # Get current context id
                Id = hit["chunk"]
                context_ids.append(Id)

                ## pprint.pprint(contexts)
                # If pre post context is true, add pre and post context to background
                if prepost_context:
                    pre_context = hit["pre_context"]
                    post_context = hit["post_context"]
                    background += pre_context + " " + context + " " + post_context + " "
                else:  # Else just add context
                    background += context + " "

        return background, contexts, context_ids

    def getIndexes(self):
        """
        Retrieves the indexes from the message queue.

        Returns:
            A list of indexes.
        """
        return self.mq.get_indexes()["results"]

    def getIndexStats(self):
        """
        Retrieves the statistics of the specified index.

        Returns:
            dict: A dictionary containing the statistics of the index.
        """
        return self.mq.index(self.indexName).get_stats()

    def getIndexSettings(self):
        """
        Retrieves the settings of the specified index.

        Returns:
            dict: A dictionary containing the settings of the index.
        """
        return self.mq.index(self.indexName).get_settings()

    def emptyIndex(self):
        """
        Delete all documents in the index.

        This method deletes all documents in the specified index by iterating through the documents
        and deleting them in batches of 500. It prints the progress and the number of documents deleted.

        Returns:
            None
        """
        print(f"Deleting all documents in index: {self.indexName}")
        currentDocs = self.mq.index(self.indexName).search(q="", limit=500)
        delete_count = len(currentDocs["hits"])
        nAllDocs = self.mq.index(self.indexName).get_stats()["numberOfDocuments"]
        while len(currentDocs["hits"]) > 0:
            for doc in currentDocs["hits"]:
                self.mq.index(self.indexName).delete_documents([doc["_id"]])
            currentDocs = self.mq.index(self.indexName).search(q="", limit=500)
            print(f'Len of current docs {len(currentDocs["hits"])}')
            print(
                f"Deleted : {delete_count} documents of {nAllDocs} documents in index: {self.indexName} "
            )
            delete_count += len(currentDocs["hits"])
        # Check if done
        if len(currentDocs["hits"]) == 0:
            print(f"Done! Deleted all documents in index: {self.indexName}")
        else:
            print(
                f"Failed to delete all documents in index: {self.indexName}. \n"
                f" Documents left: {len(currentDocs['hits'])}"
            )


class RagPipe:
    # RAG pipeline
    def __init__(self):
        self.PROMPT_EN = (
            "You are a helpful assisstant. Context information is given in the following text."
            "Use only information from the context to answer the question."
            "If you are uncertain, you must say so. Give reasoning on your answer by only"
            "refering to the given context."
        )
        self.PROMPT_DE = (
            "Sie sind ein hilfreicher Assistent. Antworte auf Deutsch. Kontextinformationen sind im folgenden Text enthalten."
            "Verwenden Sie ausschließlich Informationen aus dem Kontext, um die Frage zu beantworten."
            "Wenn Sie sich unsicher sind, müssen Sie das sagen. Begründen Sie Ihre Antwort, indem Sie nur"
            "indem Sie sich auf den gegebenen Kontext beziehen."
        )
        # Default to english
        self.PROMPT = self.PROMPT_EN

    def connectVectorStore(self, vectorStore):
        self.DB = vectorStore

    def connectLLM(self, LLM_URL, LLM_NAME):
        self.LLM_URL = LLM_URL
        self.LLM_NAME = LLM_NAME
        print(f" Language model URL: {LLM_URL}")
        print(f" Language model connected: {LLM_NAME}")

    def setCostumPrompt(self, userPrompt):
        self.PROMPT = userPrompt
        print(f"Prompt set: {userPrompt}")

    def sendToLLM(
        self,
        messages,
        model_temp=0.0,
        answer_size=100,
        presence_pen=0.0,
        repeat_pen=0.0,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer N/A ",
        }
        data = {
            "model": self.LLM_NAME,
            "messages": messages,
            # "temperature": model_temp,
            # "max_tokens": answer_size,
            # "presence_penalty": presence_pen,
            # "repeat_penalty": repeat_pen,
        }
        endpoint = self.LLM_URL + "/chat/completions"
        print("Sending query to OpenAI endpoint: " + endpoint)
        report = requests.post(endpoint, headers=headers, json=data).json()
        print("Received response...")
        if "choices" in report:
            if len(report["choices"]) > 0:  # Always take the first choice.
                result = report["choices"][0]["message"]["content"]
            else:
                result = "No result generated!"
        else:
            result = report
        return result

    def evalSendToLLM(
        self,
        messages,
        model_temp=0.0,
        answer_size=1,
        presence_pen=0.0,
        repeat_pen=0.0,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer N/A ",
        }
        data = {
            "model": self.LLM_NAME,
            "messages": messages,
            "temperature": model_temp,
            "max_tokens": answer_size,
            "presence_penalty": presence_pen,
            "repeat_penalty": repeat_pen,
        }
        endpoint = self.LLM_URL + "/chat/completions"
        print("Sending query to OpenAI endpoint: " + endpoint)
        report = requests.post(endpoint, headers=headers, json=data).json()
        print("Received response...")
        if "choices" in report:
            if len(report["choices"]) > 0:  # Always take the first choice.
                result = report["choices"][0]["message"]["content"]
            else:
                result = "No result generated!"
        else:
            result = report
        return result

    def answerQuery(self, query, rerank=False, prepost_context=False, lang="EN"):
        # Retrieve top k documents from indexName based on query

        # Update filter string with language for index search
        background, contexts, contexts_ids = self.DB.getBackground(
            query,
            num_ref=num_ref,
            lang=lang,
            rerank=rerank,
            prepost_context=prepost_context,
        )

        # Update language prompt for LLM
        if lang == "DE":
            self.PROMPT = self.PROMPT_DE
        if lang == "EN":
            self.PROMPT = self.PROMPT_EN

        # Tell llm again to obey instructions
        enforce_query = (
            "Given this context information and not prior knowledge, answer the following user query"
            + query
        )
        messages = [
            {"role": "user", "content": self.PROMPT},
            {"role": "assistant", "content": background},
            {"role": "user", "content": enforce_query},
        ]

        result = self.sendToLLM(messages)
        return result, contexts, contexts_ids

    def evaluate_context_relevance(
        self,
        queries,
        contexts=None,
        contexts_ids=None,
        goldPassages=None,
        evaluator="sem_similarity",
    ):
        # Type checking
        if not isinstance(queries, list):
            raise TypeError(
                f"Queries must be of type list, but got {type(queries).__name__}."
            )

        # check if gold passages available
        if goldPassages is None:
            scores = {}
            # If no contexts are provided, retrieve top 3 documents from index based on query
            # for each document.

            if (
                contexts is None
            ):  # Extend context to list of nones to match queries length
                contexts = [None] * len(queries)

            # Loop over all queries with their respective contexts
            for query, context in zip(queries, contexts):
                if context is None:
                    # Retrieve top 3 documents from index based on query
                    context, ids = self.DB.retrieveDocuments(query, 3)

                measurements = []
                # Loop over all contexts for a query
                for single_context in context:
                    print(f"Context: {single_context}")

                    # Evaluate context relevance based on chosen evaluator
                    if evaluator == "sem_similarity":
                        measure = self.semantic_similarity(single_context, query)
                    elif evaluator == "llm_judge":
                        measure = self.llm_binary_context_relevance(
                            single_context, query
                        )
                    # Convert measure to float and append to list
                    measurements.append(round(float(measure), 3))
                # Compute mean context relevance over all contexts per query
                scores[query] = np.array(measurements)

            return scores

        if goldPassages is not None:
            pass  # Implement evaluation with goldPassages
            print("Evaluation with goldPassages not implemented yet")

    def llm_binary_context_relevance(self, context, query):
        messages = [
            {
                "role": "system",
                "content": "Given the following context and query,"
                " Give a binary rating, either 0 or 1."
                " Respond with 0 if an answer to the query cannot be derived from the given context. "
                "Respond with 0 if an answer to the query can be derived from the given context.  "
                'Strictly respond with  either  "0" or "1"'
                'The output must strictly and only be a single integer "0" or "1" and no additional text.',
            },
            {"role": "user", "content": f"Context: {context} ; Query: {query}"},
        ]

        result = self.evalSendToLLM(messages)
        return result

    def evaluate_faithfulness(
        self, answers, contexts, evaluator="sem_similarity", contexts_ids=None
    ):
        # Type checking
        if not isinstance(answers, list):
            raise TypeError(
                f"Answers must be of type list, but got {type(answers).__name__}."
            )
        scores = {}
        for answer, context in zip(answers, contexts):
            measurements = []
            for single_context in context:
                # Insert here evaluation measure of retrieved context
                print(f"Context: {single_context}")  #
                if evaluator == "sem_similarity":
                    measure = self.semantic_similarity(single_context, answer)
                elif evaluator == "llm_judge":
                    measure = self.llm_binary_faithfulness(single_context, answer)

                measurements.append(round(float(measure), 3))
            # Compute mean faithfulness over all contexts per answer
            scores[answer] = np.array(measurements)  # Insert evaluation measure here

        return scores

    def llm_binary_faithfulness(self, context, answer):
        messages = [
            {
                "role": "system",
                "content": "Given the following context and answer,"
                " Give a binary rating, either 0 or 1."
                " Respond wiht 0 if the answer is not sufficiently grounded in the context. "
                " Respond wiht 1 if the answer is sufficiently grounded in the context. "
                ' Strictly respond with  either  "0" or "1"'
                'The output must strictly and only be a single integer "0" or "1" and no additional text.',
            },
            {"role": "user", "content": f"Context: {context} ; Answer: {answer}"},
        ]

        result = self.evalSendToLLM(messages)
        return result

    def evaluate_answer_relevance(self, queries, answers, evaluator="sem_similarity"):
        # Type checking
        if not isinstance(answers, list):
            raise TypeError(
                f"Answers must be of type list, but got {type(answers).__name__}."
            )

        if not isinstance(queries, list):
            raise TypeError(
                f"Queries must be of type list, but got {type(queries).__name__}."
            )

        scores = {}
        for answer, query in zip(answers, queries):
            print(f"Answer: {answer}")
            print(f"Query: {query}")
            # Evaluate context relevance based on chosen evaluator
            if evaluator == "sem_similarity":
                measure = self.semantic_similarity(answer, query)
            elif evaluator == "llm_judge":
                measure = self.llm_binary_answer_relevance(answer, query)
            # Convert measure to float and append to list
            scores[query] = round(float(measure), 3)
        return scores

    def llm_binary_answer_relevance(self, answer, query):
        messages = [
            {
                "role": "system",
                "content": "Given the following query and answer,"
                "Analyse the question and answer without consulting prior knowledge."
                " Determine if the answer is relevant to the question."
                " Give a binary rating, either 0 or 1."
                " Consider whether the answer addresses all parts of question asked."
                " Respond with 0 if the answer does not address the question"
                " Respond with 1 if the answer addresses to the question"
                ' Strictly respond with  either  "0" or "1"'
                'The output must strictly and only be a single integer "0" or "1" and no additional text.',
            },
            {"role": "user", "content": f"Query: {query} ; Answer: {answer}"},
        ]
        result = self.evalSendToLLM(messages)
        return result

    def evaluate_correctness(self, answers, ground_truths, evaluator="sem_similarity"):
        # Type checking
        if not isinstance(answers, list):
            raise TypeError(
                f"Answers must be of type list, but got {type(answers).__name__}."
            )

        if not isinstance(ground_truths, list):
            raise TypeError(
                f"Queries must be of type list, but got {type(ground_truths).__name__}."
            )

        scores = {}
        for answer, ground_truth in zip(answers, ground_truths):
            print(f"Answer: {answer}")
            print(f"Ground truth: {ground_truth}")
            if evaluator == "sem_similarity":
                measure = self.semantic_similarity(answer, ground_truth)
            elif evaluator == "llm_judge":
                measure = self.llm_binary_correctness(answer, ground_truth)

            scores[answer] = round(float(measure), 3)
        return scores

    def llm_binary_correctness(self, answer, ground_truth):
        messages = [
            {
                "role": "system",
                "content": "Given the following answer and ground truth,"
                "Analyse the question and answer without consulting prior knowledge."
                " Determine if the answer is correct based on the ground truth."
                " Give a binary rating, either 0 or 1."
                " Consider whether the ground truth matches the answer in meaning."
                " Respond with 0 if the answer is incorrect based on the ground truth."
                " Respond with 1 if the answer is correct based on the ground truth."
                ' Strictly respond with  either  "0" or "1"'
                'The output must strictly and only be a single integer "0" or "1" and no additional text.',
            },
            {
                "role": "user",
                "content": f"Answer: {answer} ; GroundTruth: {ground_truth}",
            },
        ]
        result = self.evalSendToLLM(messages)
        return result

    def semantic_similarity(self, sentence1, sentence2):
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # cheap model for dev
        # all-mpnet-base-v2 , more performant model, but slower
        sentence1_vec = model.encode([sentence1])

        sentence2_vec = model.encode([sentence2])

        similarity_score = model.similarity(
            sentence1_vec, sentence2_vec
        )  # Default is cosine simi
        print(f"\n Similarity Score = {similarity_score} ")

        return similarity_score

    def run(
        self,
        questions,
        ground_truths=None,
        corpus_list=None,
        newIngest=False,
        rerank=False,
        prepost_context=False,
        lang="EN",
        maxDocs=1000000,
        maxQueries=1000000,
    ):
        # Run RAG pipeline for questions, ground_truhts and corpus_list
        if newIngest:
            self.DB.emptyIndex()
            print("Index emptied")
            print("Start indexing documents. Please wait. ")
            self.DB.indexDocuments(documents=corpus_list, maxDocs=maxDocs)
            print(f" You are using index: {self.DB.indexName}")
            print(f"Done! Index Stats:  {self.DB.getIndexStats()}")

        elif not newIngest or not corpus_list:
            print("Using already indexed documents.")
            print(f" You are using index: {self.DB.indexName}")
            print(f"Index Stats:  {self.DB.getIndexStats()}")

        # Create a list of dictionaries with keys: question, answer, contexts, context_ids, ground_truth

        print("Start answering queries. Please wait. ")

        # Create list of list of rag elements. Every rag element is a dictionary
        # containing the question, answer, contexts, context_ids and ground_truth
        self.rag_elements = []
        if ground_truths is None:
            print("No ground truths given!")
            ground_truths = [None] * len(questions)

        for question, ground_truth in zip(
            questions[:maxQueries],
            ground_truths[:maxQueries],
        ):
            self.rag_elements.append(
                {
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "contexts_ids": [],
                    "ground_truth": ground_truth,
                }
            )

        # Iterate over the rag elements and get the answer from the LLM model and the contexts from the Vector DB
        for rag_element in self.rag_elements:
            print(f"Current Question: {rag_element['question']}")
            llmanswer, contexts, contexts_ids = self.answerQuery(
                rag_element["question"], rerank, prepost_context, lang
            )  # Get answer from LLM model
            rag_element["answer"] = llmanswer
            rag_element["contexts"] = contexts
            rag_element["contexts_ids"] = contexts_ids

    def eval(self, method=None, queries=None, evaluator="sem_similarity"):
        # Select evalaution method to run
        if method is None or method not in [
            "context_relevance",
            "faithfulness",
            "answer_relevance",
            "correctness",
            "all",
        ]:
            print("No evaluation method selected")
            return
        # Print choices
        print(f"Running evaluation for method: {method}")
        print(f"Using evaluator: {evaluator}")

        if method == "context_relevance":
            if queries is not None:
                # Bypass rag pipeline run and just evaluate context relevance
                # betweeen queries and contexts
                scores = self.evaluate_context_relevance(queries, contexts=None)
            else:
                contexts = [element["contexts"] for element in self.rag_elements]
                queries = [element["question"] for element in self.rag_elements]
                contexts_ids = [
                    element["contexts_ids"] for element in self.rag_elements
                ]
                scores = self.evaluate_context_relevance(
                    queries, contexts, contexts_ids=contexts_ids, evaluator=evaluator
                )
            return scores

        if method == "faithfulness":
            answers = [element["answer"] for element in self.rag_elements]
            contexts = [element["contexts"] for element in self.rag_elements]
            contexts_ids = [element["contexts_ids"] for element in self.rag_elements]
            scores = self.evaluate_faithfulness(
                answers,
                contexts,
                contexts_ids=contexts_ids,
                evaluator=evaluator,
            )
            return scores

        if method == "answer_relevance":
            queries = [element["question"] for element in self.rag_elements]
            answers = [element["answer"] for element in self.rag_elements]
            scores = self.evaluate_answer_relevance(queries, answers, evaluator)
            return scores

        if method == "correctness":
            answers = [element["answer"] for element in self.rag_elements]
            ground_truths = [element["ground_truth"] for element in self.rag_elements]
            scores = self.evaluate_correctness(answers, ground_truths, evaluator)
            return scores

        if method == "all":
            queries = [element["question"] for element in self.rag_elements]
            contexts = [element["contexts"] for element in self.rag_elements]
            contexts_ids = [element["context_ids"] for element in self.rag_elements]

            answers = [element["answer"] for element in self.rag_elements]
            ground_truths = [element["ground_truth"] for element in self.rag_elements]

            cr_scores = self.evaluate_context_relevance(
                queries, contexts, evaluator=evaluator, contexts_ids=contexts_ids
            )

            f_scores = self.evaluate_faithfulness(
                answers, contexts, evaluator=evaluator, contexts_ids=contexts_ids
            )
            ar_scores = self.evaluate_answer_relevance(queries, answers, evaluator)
            c_scores = self.evaluate_correctness(answers, ground_truths, evaluator)

            return {
                "context_relevance": cr_scores,
                "faithfulness": f_scores,
                "answer_relevance": ar_scores,
                "correctness": c_scores,
            }

    def eval_context_goldPassages(self, goldPassages):
        # Evaluate context relevance with goldPassages
        contexts = [element["contexts"] for element in self.rag_elements]
        queries = [element["question"] for element in self.rag_elements]
        contexts_ids = [element["contexts_ids"] for element in self.rag_elements]

        matches = []
        for query, context_ids, goldPs in zip(queries, contexts_ids, goldPassages):
            print(f"Context_ids: {context_ids}")
            print(f"goldPasssageContexts: {goldPs}")

            # Count number of matches in context_ids and goldPs
            # Beware, that the number of elements in goldPs per query varies.
            set_goldPs = set(goldPs)
            number_matches = sum(1 for element in context_ids if element in set_goldPs)
            print(f"Query: {query}")
            print(f"Number of matches: {number_matches}")
            matches.append(number_matches)


class DatasetHelpers:
    # Helper functions for datasets
    def __init__(self, chunking_params=None):
        self.chunking_params = chunking_params

    def loadFromDocuments(self, documents):
        # Load from a corpus of pdf documents
        pass

    def loadSQUAD(self):
        # Load SQUAD dataset
        pass

    def loadTriviaQA(self):
        # Load TriviaQA dataset
        pass

    def loadHotpotQA(self):
        # Load HotpotQA dataset
        pass

    def loadNaturalQuestions(self):
        # Load NaturalQuestions dataset
        pass

    def loadMiniWiki(self):
        # Load MiniWiki dataset
        print("Loading MiniWiki dataset")
        corpus = load_dataset("rag-datasets/mini_wikipedia", "text-corpus")["passages"]
        # Create a list of dictionaries with keys: passage, id
        corpus_list = []
        for passage, iD in zip(corpus["passage"], corpus["id"]):
            corpus_list.append({"text": passage, "id": iD})

        QA = load_dataset("rag-datasets/mini_wikipedia", "question-answer")["test"]

        queries = QA["question"]
        ground_truths = QA["answer"]

        return corpus_list, queries, ground_truths

    def loadMiniBioasq(self):
        # Load MiniBioasq dataset
        print("Loading MiniBioasq dataset")
        corpus = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")["train"]
        # Create a list of dictionaries with keys: passage, id
        corpus_list = []
        for passage, iD in zip(corpus["passage"], corpus["id"]):
            corpus_list.append({"text": passage, "id": iD})

        QA = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")[
            "train"
        ]

        queries = QA["question"]
        ground_truths = QA["answer"]
        goldPassages = QA["relevant_passage_ids"]

        return corpus_list, queries, ground_truths, goldPassages

    def loadQM(self):
        # Load QM dataset
        print("Loading AIT QM dataset")

        corpus_list = None
        queries = None
        ground_truths = None

        # Skip the first row
        df = pd.read_excel("./data/100_questions.xlsx", skiprows=0, usecols=[1])
        queries = df.iloc[:, 0].tolist()

        return corpus_list, queries, ground_truths
