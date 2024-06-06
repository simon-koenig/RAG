# Components for RAG pipeline
import pprint
import re

import marqo
import requests
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, url):
        self.mq = marqo.Client(url=url)
        self.indexName = None

    def connectIndex(self, indexName):
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
                        "id": ID,
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

    def retrieveDocuments(self, query, k):
        # Retrieve top k documents from indexName based on query
        response = self.mq.index(self.indexName).search(
            q=query,  # Query string
            limit=k,  # Number of documents to retrieve
            attributes_to_retrieve=["text", "id"],  # Attributes to retrieve
        )

        contexts = [response["hits"][i]["text"] for i in range(len(response["hits"]))]
        ids = [response["hits"][i]["id"] for i in range(len(response["hits"]))]

        return contexts, ids

    def getBackground(self, query, k):
        # Retrieve top k documents from indexName based on query
        response = self.mq.index(self.indexName).search(
            q=query,  # Query string
            limit=k,  # Number of documents to retrieve
            attributes_to_retrieve=["text", "id"],  # Attributes to retrieve
        )

        # Get retrieved text.
        contexts = [response["hits"][i]["text"] for i in range(len(response["hits"]))]
        pprint.pprint(contexts)
        background = ""
        for text in contexts:
            background += text + " "
        return background

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
        self.PROMPT = " Dummy prompt"

    def connectVectorStore(self, vectorStore):
        self.DB = vectorStore

    def connectLLM(self, LLM_URL, LLM_NAME):
        self.LLM_URL = LLM_URL
        self.LLM_NAME = LLM_NAME
        print(f" Language model URL: {LLM_URL}")
        print(f" Language model connected: {LLM_NAME}")

    def setCostumPrompt(self, query, userPrompt):
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

    def answerQuery(self, query):
        # Retrieve top k documents from indexName based on query

        background = self.DB.getBackground(query, 3)
        messages = [
            {"role": "user", "content": self.PROMPT},
            {"role": "assistant", "content": background},
            {"role": "user", "content": query},
        ]

        result = self.sendToLLM(messages)
        return result

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
                    context, ids = self.retrieveDocuments(query, 3)

                measurements = []
                for single_context in context:
                    # Insert here evaluation measure of retrieved context
                    # print(f"ID: {single_id}")
                    print(f"Context: {single_context}")
                    measure = self.llm_binary_context_relevance(single_context, query)
                    measurements.append(measure)

                scores[query] = measurements  # Insert evaluation measure here

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
                " Respond wiht 0 if the context is not sufficient for answering the query. "
                " Respond with 1 the context is sufficient for answering the query. "
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
                measurements.append(measure)

            scores[answer] = measurements  # Insert evaluation measure here

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
            scores.append(measure)
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
            scores.append(measure)
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


class RagElement:
    def __init__(self):
        self.query = None
        self.answer = None
        self.ground_truth = None
        self.contexts = None
        self.contexts_ids = None
        self.goldPassages = None
        self.goldPassages_ids = None

    def setQuery(self, query):
        self.query = query

    def getQuery(self):
        return self.query

    def setAnswer(self, answer):
        self.answer = answer

    def getAnswer(self):
        return self.answer

    def setGroundTruth(self, ground_truth):
        self.ground_truth = ground_truth

    def getGroundTruth(self):
        return self.ground_truth

    def setContexts(self, contexts):
        self.contexts = contexts

    def getContexts(self):
        return self.contexts

    def setContextsIds(self, contexts_ids):
        self.contexts_ids = contexts_ids

    def getContextsIds(self):
        return self.contexts_ids

    def setGoldPassages(self, goldPassages):
        self.goldPassages = goldPassages

    def getGoldPassages(self):
        return self.goldPassages

    def setGoldPassagesIds(self, goldPassages_ids):
        self.goldPassages_ids = goldPassages_ids

    def getGoldPassagesIds(self):
        return self.goldPassages_ids

    def getRagElement(self):
        return {
            "query": self.query,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "contexts": self.contexts,
            "contexts_ids": self.contexts_ids,
            "goldPassages": self.goldPassages,
            "goldPassages_ids": self.goldPassages_ids,
        }

    def setRagElement(self, ragElement):
        self.query = ragElement["query"]
        self.answer = ragElement["answer"]
        self.ground_truth = ragElement["ground_truth"]
        self.contexts = ragElement["contexts"]
        self.contexts_ids = ragElement["contexts_ids"]
        self.goldPassages = ragElement["goldPassages"]
        self.goldPassages_ids = ragElement["goldPassages_ids"]
        return self

    def resetRagElement(self):
        self.query = None
        self.answer = None
        self.ground_truth = None
        self.contexts = None
        self.contexts_ids = None
        self.goldPassages = None
        self.goldPassages_ids = None
        return self
