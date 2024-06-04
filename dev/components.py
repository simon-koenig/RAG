# Components for RAG pipeline
import pprint
import re

import marqo
import requests


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
        return self.mq.index(self.indexName).search(
            q=query,  # Query string
            limit=k,  # Number of documents to retrieve
            attributes_to_retrieve=["text", "id"],  # Attributes to retrieve
        )

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

    def answerQuery(self, query):
        # Retrieve top k documents from indexName based on query

        background = self.DB.getBackground(query, 3)
        messages = [
            {"role": "user", "content": self.PROMPT},
            {"role": "assistant", "content": background},
            {"role": "user", "content": query},
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer N/A ",
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
        print("Response: \n")
        pprint.pprint(result)
        return result
