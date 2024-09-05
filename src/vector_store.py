# Vector Store for RAG pipeline

from pprint import pprint

import marqo
import requests
from utils import chunkText

# Reranker Endpoint
RERANKER_ENDPOINT = "http://10.103.251.104:8883/rerank"
query_threshold = 0.5


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
                print(f"Indexed {len(bunch)} chunks in bunch {i}")

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

                ## Remove special characters
                pre_context = pre_context.replace("\n", " ")
                post_context = post_context.replace("\n", " ")
                chunk = chunk.replace("\n", " ")

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

        contexts = [hit["text"] for hit in response["hits"]]
        ids = [hit["chunk_id"] for hit in response["hits"]]

        return contexts, ids

    def getBackground(
        self,
        query,
        num_ref,
        lang,
        rerank=False,  # [False, True, "rrf"]
        prepost_context=False,
        background_reversed=False,
        query_expansion=False,
        LLM_URL=False,
        LLM_NAME=False,
    ):
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

        if query_expansion is False:
            # Semantic Search
            response_sem = self.mq.index(self.indexName).search(
                q=query,  # Query string
                limit=num_ref,  # Number of documents to retrieve
                # attributes_to_retrieve=["text", "chunk"],
                # filter_string=filterstring,  # Filter in db, e.g. for lang  # Attributes to retrieve, explicit is faster
            )["hits"]

            # Lexial Search
            response_lex = self.mq.index(self.indexName).search(
                q=query,  # Query string
                limit=num_ref,  # Number of documents to retrieve
                # attributes_to_retrieve=["text", "chunk"],  # Attributes to retrieve, explicit is faster
                # filter_string=filterstring,  # Filter in db, e.g. for lang
                search_method="LEXICAL",
            )["hits"]

        # If context expansion is needed
        if query_expansion is not False:
            print("Query expansion is needed.")
            # Send data to the LLM for context expansion to retrieve more relevant contexts
            # Query expansion demands LLM_URL and LLM_NAME, otherwise not possible
            if (LLM_URL or LLM_NAME) is False:
                raise Exception(
                    "LLM_URL and LLM_NAME must be provided for query expansion."
                )
                return
            # Request headers
            headers = {
                "Content-Type": "application/json",
            }
            # Request payload
            messages = [
                {
                    "role": "system",
                    "content": "You an information system that helps process user questions. Provide information such that a vector database retrieval system can find the most relevant documents.",
                },
                {
                    "role": "user",
                    "content": f"Expand the following query:\n\n{query}\n\n to {query_expansion} relevant queries which are close in meaning but use different wording."
                    "Here are some examples:"
                    '1. Query: "What is the capital of France?"'
                    'Example expansion: ["Big cities in France", "Capitals in Europe", What is the capital of France?, "What is the capital of France?"]'
                    '2. Query: "Drugs for cancer treatment?"'
                    'Example expansions: ["Cure against cancer", "Medications for cancer", "Drugs for cancer treatment?"]'
                    '3. "What positions are there in a football team?"'
                    'Example expansions: [Football team roles, "Positions in soccer", "What positions are there in a football team?"]'
                    "Use the same list structure in your answer as in the examples above. The last query in the list should be the original query."
                    "Structure your response as a list of strings, where each string is a query. The answer should be just this list and nothing else.",
                },
            ]
            data = {
                "model": LLM_NAME,
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.7,
            }

            # Send the request
            endpoint = f"{LLM_URL}/chat/completions"
            response = requests.post(endpoint, headers=headers, json=data)
            # Safe guard
            if not response.status_code == 200:
                print(f"LLM failed with status code: {response.status_code}")
                raise ("LLM failed")
                return

            # Parse the response
            response_data = response.json()

            # Extract the queries from the response
            print("Response data:")
            pprint(response_data)
            if "choices" in response_data:
                extended_queries = (
                    response_data["choices"][0]["message"]["content"]
                    .strip("[]")
                    .replace('"', "")
                    .split(", ")
                )

            # Print or use the list of queries
            print(extended_queries)
            print(f" Type of extended queries: {type(extended_queries)}")

            # Merge search results for all queries in the extended_queries list
            response_sem = []
            response_lex = []
            # Now we have the extended queries, we can search for them in the index
            for query in extended_queries:
                # Semantic Search
                response_sem_temp = self.mq.index(self.indexName).search(
                    q=query,  # Query string
                    limit=num_ref,  # Number of documents to retrieve
                    # attributes_to_retrieve=["text", "chunk"],
                    # filter_string=filterstring,  # Filter in db, e.g. for lang  # Attributes to retrieve, explicit is faster
                )["hits"]

                # Lexial Search
                response_lex_temp = self.mq.index(self.indexName).search(
                    q=query,  # Query string
                    limit=num_ref,  # Number of documents to retrieve
                    # attributes_to_retrieve=["text", "chunk"],  # Attributes to retrieve, explicit is faster
                    # filter_string=filterstring,  # Filter in db, e.g. for lang
                    search_method="LEXICAL",
                )["hits"]

                # Append to the response lists
                response_sem.extend(response_sem_temp)
                response_lex.extend(response_lex_temp)

        ##
        ## Construct contexts and background
        ##

        background = ""
        contexts = []
        context_ids = []
        full_results = []
        plain_text_results = []

        # Get initial search results
        # print(f" Semantic Search Results: {response_sem}")
        # print(f" Lexical Search Results: {response_lex}")
        # Fusing the lists alternately
        full_results = [
            item for pair in zip(response_lex, response_sem) for item in pair
        ]

        # Adding the remaining elements (if results are not the same length)
        full_results.extend(
            response_lex[len(response_sem) :] or response_sem[len(response_lex) :]
        )
        # Get plain text results for semantic reranker later on
        plain_text_results = [item["text"] for item in full_results]

        # If rerank is false, combine semantic and lexical search results
        if rerank is False:
            # Combine semantic and lexical search results
            for hit in full_results:
                # If needed, augment context with title, link and score
                text = hit["text"]
                # title = hit["title"]
                # link_url = hit["link_url"]
                score = hit["_score"]
                ##
                ## Context Expansion
                ##
                if prepost_context:
                    pre_context = hit["pre_context"]
                    post_context = hit["post_context"]
                    context = f" Context: {pre_context} {text} {post_context} "  # Context expansion
                else:
                    context = f" Context: {text} "
                # Append to context list
                contexts.append(context)
                # Get current context id and append to context_ids
                Id = hit["chunk_id"]
                context_ids.append(Id)

                ##
                ## Context Expansion
                ##

                if prepost_context:
                    pre_context = hit["pre_context"]
                    post_context = hit["post_context"]
                    context = f" Context: {pre_context} {text} {post_context} "  # Context expansion
                else:
                    context = f" Context: {text} "

                contexts.append(context)
                # Get current context id and append to context_ids
                Id = hit["chunk_id"]
                context_ids.append(Id)

        if rerank is True:
            # Rerank the results using the semantic reranker
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
                print(f"Reranked index: {current_index}")
                # Get current highest ranked context
                hit = full_results[current_index]
                # If needed, augment context with title, link and score
                text = hit["text"]
                # title = hit["title"]
                # link_url = hit["link_url"]
                score = hit["_score"]

                ##
                ## Context Expansion
                ##

                if prepost_context:
                    pre_context = hit["pre_context"]
                    post_context = hit["post_context"]
                    context = f" Context: {pre_context} {text} {post_context} "  # Context expansion
                else:
                    context = f" Context: {text} "

                contexts.append(context)
                # Get current context id and append to context_ids
                Id = hit["chunk_id"]
                context_ids.append(Id)

        if rerank == "rrf":
            # Rerank the results using RRF

            tensor_results = [
                {"rank": i + 1, "id": response["chunk_id"]}
                for i, response in enumerate(response_sem)
            ]
            lex_results = [
                {"rank": i + 1, "id": response["chunk_id"]}
                for i, response in enumerate(response_lex)
            ]
            for i in range(len(tensor_results)):
                pprint(tensor_results[i])
                pprint(lex_results[i])

            # Step 1: Init empty set
            unique_ids = set()

            # Step 2: Combine the keys to get unique keys
            for dic in tensor_results + lex_results:
                unique_ids.add(dic["id"])
            # Step 3: Create a new dictionary with all values set to 0
            rrf_scores = {key: 0 for key in unique_ids}

            pprint(rrf_scores)
            # Step 4: Iterate over the results and add the scores
            for result in lex_results + tensor_results:
                ID = result["id"]
                rank = result["rank"]
                rrf_scores[ID] += 1 / (rank + 60)

            pprint(rrf_scores)

            # Step 5: sort the search results by the rrf scores
            sorted_rrf_scores = sorted(
                rrf_scores.items(), key=lambda x: x[1], reverse=True
            )
            pprint(sorted_rrf_scores)
            # Get sorted ids
            sorted_ids = [id_ for id_, score in sorted_rrf_scores]
            pprint(sorted_ids)

            # Step 1: Create a mapping from ID to full_results
            id_to_dict = {d["chunk_id"]: d for d in full_results}

            # Step 2: Sort the list of dictionaries according to sorted_ids
            sorted_hits = [id_to_dict[id_] for id_ in sorted_ids]

            # Now, sorted_hits contains dictionaries ordered according to sorted_ids
            pprint(sorted_hits)

            # Build the background
            # Get current context
            for hit in sorted_hits:
                text = hit["text"]
                # If needed, Augment context with title, link and score
                # title = hit["title"]
                # link_url = hit["link_url"]
                score = hit["_score"]
                ##
                ## Context Expansion
                ##
                if prepost_context:
                    pre_context = hit["pre_context"]
                    post_context = hit["post_context"]
                    context = f" Context: {pre_context} {text} {post_context} "  # Context expansion
                else:
                    context = f" Context: {text} "
                # Append to context list
                contexts.append(context)
                # Get current context id and append to context_ids
                Id = hit["chunk_id"]
                context_ids.append(Id)

        # Add context to background
        # Use only the top 5 contexts for background
        # If descending order is needed, set reverse to False
        if background_reversed is False:
            background = " ".join(contexts[:5])

        # If ascending order is needed, set reverse to True
        if background_reversed is True:
            background = " ".join(reversed(contexts[:5]))

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
