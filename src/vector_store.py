# Vector Store for RAG pipeline

import marqo
import requests
from utils import chunkText

# Reranker Endpoint
RERANKER_ENDPOINT = "http://10.103.251.104:8883/rerank"
query_threshold = 0.5
num_ref = 3


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

    def createIndex(self, indexName, settings=None):
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
        if settings:
            self.split_method = settings["split_method"]
            self.distance_metric = settings["distance_metric"]
            self.model = settings["model"]
        else:
            print(f"No settings provided for index: {indexName}. Use default settings.")

        try:
            if settings:
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
            else:
                self.mq.create_index(indexName)
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
                        "chunk_id": chunk_id,
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

    def indexDocumentBunch(self, documentBunch):
        """
        Indexes a list of documents in the search engine.

        Args:
            documentBunch (list): A list of dictionaries, with each dict representing the document to be indexed.
            It should have the following keys:
                - 'chunk_id': The ID of the document chunk.
                - 'text': The main text of the document.
                - 'pre_context': The pre-context of the document.
                - 'post_context': The post-context of the document.

        Returns:
            None
        """

        # Make bunches of bunchSize documents to be indexed max at once
        bunchSize = 128
        bunches = [
            documentBunch[i : i + bunchSize]
            for i in range(0, len(documentBunch), bunchSize)
        ]
        # Iteratore over bunches of documents and index them simultaneously

        for i, bunch in enumerate(bunches):
            try:
                # Index the bunch of documents
                self.mq.index(self.indexName).add_documents(
                    bunch,
                    tensor_fields=["text"],
                )
                print(f"Indexed {len(bunch)} documents in bunch {i}")

            except:
                print(
                    "Ingest error."
                    "DocumentBunch has to be a list of dicts with at least the"
                    "keys: 'chunk_id', 'text',"
                    "'pre_context', and 'post_context'"
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
        all_documents_chunked = []  # List to store all dictionaries of chunked documents
        # Iterate over all documents and get the document id and text of full document
        for n in range(n_docs):
            document_id = documents[n]["id"]
            main_text = documents[n]["text"]
            # print(f"Indexing document: {document_id}")
            # print(f"Full Document Text: {main_text}")
            # Chunk the document into smaller chunks
            chunks = chunkText(
                text=main_text,
                method="recursive",
                chunk_size=256,
                chunk_overlap=0,
            )

            # Iterate over all chunks and add pre and post context to them
            lenChunks = len(chunks)
            # print(f"Number of chunks: {lenChunks}")
            for i, chunk in enumerate(chunks):
                # print(f" Current i = {i}")
                pre_context = ""
                post_context = ""
                # Exceptions are needed the first two and last two documents
                if i > 1:
                    pre_context += chunks[i - 2] + "  "

                if i > 0:
                    pre_context += chunks[i - 1] + " "

                if i < lenChunks - 1:
                    post_context += chunks[i + 1] + " "

                if i < lenChunks - 2:
                    post_context += chunks[i + 2] + " "

                document = {
                    "text": chunk,
                    "chunk_id": document_id,
                    "pre_context": pre_context,
                    "post_context": post_context,
                }
                # print(f"Indexing document: {document}")
                all_documents_chunked.append(document)

        # Pass the chunked documents to the indexDocumentBunch method
        self.indexDocumentBunch(all_documents_chunked)
        print(f" Successfully chunked  {n+1} documents!")

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
            # attributes_to_retrieve=["text", "chunk_id"],  # Attributes to retrieve
        )

        contexts = [response["hits"][i]["text"] for i in range(len(response["hits"]))]
        ids = [response["hits"][i]["chunk_id"] for i in range(len(response["hits"]))]

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
                    Id = response["hits"][i]["chunk_id"]
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
        currentDocs = self.mq.index(self.indexName).search(q="", limit=1000)
        delete_count = len(currentDocs["hits"])
        nAllDocs = self.mq.index(self.indexName).get_stats()["numberOfDocuments"]
        while len(currentDocs["hits"]) > 0:
            for doc in currentDocs["hits"]:
                self.mq.index(self.indexName).delete_documents([doc["_id"]])
            currentDocs = self.mq.index(self.indexName).search(q="", limit=1000)
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
