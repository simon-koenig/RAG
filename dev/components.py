# Components for RAG pipeline
import pprint
import re

import marqo
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from lingua import Language, LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer


# Helper functions
def escape_markdown(text):
    MD_SPECIAL_CHARS = "\`*_{}[]()#+-.!"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\" + char)
    text = text.strip()
    return text


def get_language_detector():
    languages = [Language.ENGLISH, Language.GERMAN]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    return detector


class VectorStore:
    def __init__(self, url):
        self.mq = marqo.Client(url=url)
        self.indexName = None
        self.model = (
            "flax-sentence-embeddings/all_datasets_v4_mpnet-base"  # Default model
        )

    def connectIndex(self, indexName):
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
        # Create a new index with name indexName
        indexName = indexName.lower()
        if indexName in self.mq.get_indexes()["results"]:
            print(f"Index already exists: {indexName} ")
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
        # Delete index by indexName
        try:
            # now remove the marqo index
            self.mq.delete_index(indexName)

            print(f" Sucessfuylly deleted Index: {indexName}")
        except:
            print("Unable to delete: " + indexName)

    def indexDocument(self, document):
        # Documents has to be a dict with keys 'id' and 'text'
        ID = document["id"]
        text = document["text"]

        try:
            self.mq.index(self.indexName).add_documents(
                [
                    {
                        "text": text,
                        "tokens": self.token_estimate(text),
                        "chunk": ID,
                    }
                ],
                # Arguments in tensor_fields will have vectors generated for them. For best recall and performance, minimise the number of arguments in tensor_fields.
                tensor_fields=["text"],
            )

        except:
            print(
                f"Ingest error for passage with id: {ID},"
                "Documents has to be a dict with keys 'id' and 'text'"
            )

    def indexDocuments(self, documents, maxDocs=1000000):
        # Index a single document. Which has the form
        # document = {"id": "1", "text": "Some Text"}
        try:
            for i in range(min(len(documents), maxDocs)):
                self.indexDocument(documents[i])
        except:
            print(
                f"Error in indexing corpus for index: {self.indexName}, "
                " documents has to be a list of dictionaries with keys 'id' and 'text'"
            )

    def retrieveDocuments(self, query, k, rerank=False, prepost_context=False):
        # Retrieve top k documents from indexName based on query
        response = self.mq.index(self.indexName).search(
            q=query,  # Query string
            limit=k,  # Number of documents to retrieve
            # attributes_to_retrieve=["text", "chunk"],  # Attributes to retrieve
        )

        contexts = [response["hits"][i]["text"] for i in range(len(response["hits"]))]
        ids = [response["hits"][i]["chunk"] for i in range(len(response["hits"]))]

        return contexts, ids

    def getBackground(self, query, k, lang, rerank=False, prepost_context=False):
        # Retrieve top k documents from indexName based on query
        # Modify filter string to include language
        filterstring = f"lang:{lang}"  # Add language to filter string

        # Semantic Search
        response_sem = self.mq.index(
            self.indexName
        ).search(
            q=query,  # Query string
            limit=k,  # Number of documents to retrieve
            # attributes_to_retrieve=["text", "chunk"],
            filter_string=filterstring,  # Filter in db, e.g. for lang  # Attributes to retrieve, explicit is faster
        )

        # Lexial Search
        response_lex = self.mq.index(self.indexName).search(
            q=query,  # Query string
            limit=k,  # Number of documents to retrieve
            # attributes_to_retrieve=["text", "chunk"],  # Attributes to retrieve, explicit is faster
            filter_string=filterstring,  # Filter in db, e.g. for lang
            search_method="LEXICAL",
        )

        # Construct contexts and background
        background = ""
        contexts = []
        context_ids = []
        # Combine semantic and lexical search results
        for response in [response_sem, response_lex]:
            # Construct Background

            for i in range(len(response["hits"])):
                # Get current context
                text = response["hits"][i]["text"]
                title = response["hits"][i]["title"]
                link_url = response["hits"][i]["link_url"]
                score = response["hits"][i]["_score"]
                # Augment context with title, link and score
                context = f" {text}, source: {title},  ({link_url}) (Score: {score}) "
                contexts.append(context)
                # Get current context id
                Id = response["hits"][i]["chunk"]
                context_ids.append(Id)

                ## pprint.pprint(contexts)
                # If pre post context is true, add pre and post context to background
                if prepost_context:
                    pre_context = response["hits"][i]["pre_context"]
                    post_context = response["hits"][i]["post_context"]
                    background += pre_context + " " + context + " " + post_context + " "
                else:  # Else just add context
                    background += context + " "

        return background, contexts, context_ids

    def getIndexes(self):
        return self.mq.get_indexes()["results"]

    def getIndexStats(self):
        return self.mq.index(self.indexName).get_stats()

    def getIndexSettings(self):
        return self.mq.index(self.indexName).get_settings()

    def emptyIndex(self):
        # Delete all Documents in the index
        print(f"Deleting all documents in index: {self.indexName}")
        allDocs = self.mq.index(self.indexName).search(q="", limit=1000)
        print(f"Documents in index: {len(allDocs['hits'])}")
        while len(allDocs["hits"]) > 0:
            for doc in allDocs["hits"]:
                self.mq.index(self.indexName).delete_documents([doc["_id"]])
            allDocs = self.mq.index(self.indexName).search(q="", limit=1000)
        # Check if done
        if len(allDocs["hits"]) == 0:
            print(f"Done: Deleted all documents in index: {self.indexName}")
        else:
            print(
                f"Failed to delete all documents in index: {self.indexName}. \n"
                f" Documents left: {len(allDocs['hits'])}"
            )

    def token_estimate(self, text):  # OpenAI suggest a token consists of 3/4 words
        return len(re.findall(r"\w+", text)) * 4 / 3


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

    def answerQuery(self, query, rerank=False, prepost_context=False):
        # Retrieve top k documents from indexName based on query
        languge_detector = get_language_detector()
        language = languge_detector.detect_language_of(query)
        lang = language.iso_code_639_1.name

        # Update filter string with language for index search
        background, contexts, ids = self.DB.getBackground(
            query, k=3, lang=lang, rerank=False, prepost_context=False
        )

        # Update language prompt for LLM
        if lang == "DE":
            self.PROMPT = self.PROMPT_DE
        if lang == "EN":
            self.PROMPT = self.PROMPT_EN

        messages = [
            {"role": "user", "content": self.PROMPT},
            {"role": "assistant", "content": background},
            {"role": "user", "content": query},
        ]

        result = self.sendToLLM(messages)
        return result, contexts, ids

    def evaluate_context_relevance(
        self, queries, contexts=None, ids=None, goldPassages=None
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

            for query, context in zip(queries, contexts):
                if context is None:
                    context, ids = self.DB.retrieveDocuments(query, 3)

                measurements = []
                for single_context in context:
                    # Insert here evaluation measure of retrieved context
                    # print(f"ID: {single_id}")
                    print(f"Context: {single_context}")
                    measure = self.llm_binary_context_relevance(single_context, query)
                    measurements.append(int(measure))

                scores[query] = np.mean(np.array(measurements))

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

    def evaluate_faithfulness(self, answers, contexts):
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
                print(f"Context: {single_context}")
                measure = self.llm_binary_faithfullness(single_context, answer)
                measurements.append(int(measure))
            # Compute mean faithfullness over all contexts per answer
            scores[answer] = np.mean(
                np.array(measurements)
            )  # Insert evaluation measure here

        return scores

    def llm_binary_faithfullness(self, context, answer):
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

    def evaluate_answer_relevance(self, queries, answers):
        # Type checking
        if not isinstance(answers, list):
            raise TypeError(
                f"Answers must be of type list, but got {type(answers).__name__}."
            )

        if not isinstance(queries, list):
            raise TypeError(
                f"Queries must be of type list, but got {type(queries).__name__}."
            )

        scores = []
        for answer, query in zip(answers, queries):
            print(f"Answer: {answer}")
            print(f"Query: {query}")
            measure = self.llm_binary_answer_relevance(answer, query)
            scores.append(int(measure))
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

    def evaluate_correctness(self, answers, ground_truths):
        # Type checking
        if not isinstance(answers, list):
            raise TypeError(
                f"Answers must be of type list, but got {type(answers).__name__}."
            )

        if not isinstance(ground_truths, list):
            raise TypeError(
                f"Queries must be of type list, but got {type(ground_truths).__name__}."
            )

        scores = []
        for answer, ground_truth in zip(answers, ground_truths):
            print(f"Answer: {answer}")
            print(f"Ground truth: {ground_truth}")
            measure = self.semantic_similarity(answer, ground_truth)
            scores.append(round(float(measure), 3))
        return scores

    def semantic_similarity(self, sentence1, sentence2):
        model = SentenceTransformer("all-mpnet-base-v2")

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
                    "context_ids": [],
                    "ground_truth": ground_truth,
                }
            )

        # Iterate over the rag elements and get the answer from the LLM model and the contexts from the Vector DB
        for rag_element in self.rag_elements:
            print(f"Current Question: {rag_element['question']}")
            llmanswer, contexts, context_ids = self.answerQuery(
                rag_element["question"], rerank, prepost_context
            )  # Get answer from LLM model
            rag_element["answer"] = llmanswer
            rag_element["contexts"] = contexts
            rag_element["context_ids"] = context_ids

    def eval(self, method=None, queries=None):
        # Select evalaution method to run
        if method is None:
            print("No evaluation method selected")
            return

        if method == "context_relevance":
            if queries is not None:
                # Bypass rag pipeline run and just evaluate context relevance
                # betweeen queries and contexts
                scores = self.evaluate_context_relevance(queries, contexts=None)
            else:
                contexts = [element["contexts"] for element in self.rag_elements]
                scores = self.evaluate_context_relevance(queries, contexts)
            return scores

        if method == "faithfulness":
            answers = [element["answer"] for element in self.rag_elements]
            contexts = [element["contexts"] for element in self.rag_elements]
            scores = self.evaluate_faithfulness(answers, contexts)
            return scores

        if method == "answer_relevance":
            queries = [element["question"] for element in self.rag_elements]
            answers = [element["answer"] for element in self.rag_elements]
            scores = self.evaluate_answer_relevance(queries, answers)
            return scores

        if method == "correctness":
            answers = [element["answer"] for element in self.rag_elements]
            ground_truths = [element["ground_truth"] for element in self.rag_elements]
            scores = self.evaluate_correctness(answers, ground_truths)
            return scores

        if method == "all":
            queries = [element["question"] for element in self.rag_elements]
            contexts = [element["contexts"] for element in self.rag_elements]
            answers = [element["answer"] for element in self.rag_elements]
            ground_truths = [element["ground_truth"] for element in self.rag_elements]

            cr_scores = self.evaluate_context_relevance(queries, contexts)
            f_scores = self.evaluate_faithfulness(answers, contexts)
            ar_scores = self.evaluate_answer_relevance(queries, answers)
            c_scores = self.evaluate_correctness(answers, ground_truths)

            return {
                "context_relevance": cr_scores,
                "faithfulness": f_scores,
                "answer_relevance": ar_scores,
                "correctness": c_scores,
            }


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

    def loadQM(self):
        # Load QM dataset
        print("Loading AIT QM dataset")

        corpus_list = None
        queries = None
        ground_truths = None

        # Skip the first row
        df = pd.read_excel("./data/100_questions.xlsx", skiprows=1, usecols=[1])
        queries = df.iloc[:, 0].tolist()

        return corpus_list, queries, ground_truths
