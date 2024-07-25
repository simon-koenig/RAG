# Main object for RAG pipeline

from pprint import pprint

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Reranker Endpoint
RERANKER_ENDPOINT = "http://10.103.251.104:8883/rerank"
query_threshold = 0.5
num_ref = 3


class RagPipe:
    """
    RAGPipe is a class that represents a pipeline for the RAG (Retrieval-Augmented Generation) model.
    It provides methods for connecting to a vector store, connecting to a language model, setting a custom prompt,
    sending queries to the language model, answering user queries, evaluating context relevance, faithfulness, answer relevance
    and correctness.

    Attributes:
        PROMPT_EN (str): The default English prompt for the language model.
        PROMPT_DE (str): The default German prompt for the language model.
        PROMPT (str): The current prompt for the language model.

    Methods:
        connectVectorStore(vectorStore): Connects the pipeline to a vector store.
        connectLLM(LLM_URL, LLM_NAME): Connects the pipeline to a language model.
        setCostumPrompt(userPrompt): Sets a custom prompt for the language model.
        sendToLLM(messages, model_temp, answer_size, presence_pen, repeat_pen): Sends messages to the language model and retrieves a response.
        evalSendToLLM(messages, model_temp, answer_size, presence_pen, repeat_pen): Sends messages to the language model and evaluates the response.
        answerQuery(query, rerank, prepost_context, lang): Answers a user query based on the given context.
        evaluate_context_relevance(queries, contexts, contexts_ids, goldPassages, evaluator): Evaluates the relevance of contexts for a given set of queries.
        llm_binary_context_relevance(context, query): Evaluates the binary relevance of a context for a given query.
        evaluate_faithfulness(answers, contexts, evaluator, contexts_ids): Evaluates the faithfulness of answers given a set of contexts.
        llm_binary_faithfulness(context, answer): Evaluates the binary faithfulness of an answer given a context.
        evaluate_answer_relevance(queries, answers, evaluator): Evaluates the relevance of answers for a given set of queries.
    """

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
        """
        Connects the vector store to the current instance.

        Args:
            vectorStore: The vector store to connect.

        Returns:
            None
        """
        self.DB = vectorStore

    def connectLLM(self, LLM_URL, LLM_NAME):
        """
        Connects to a Language Model (LLM) using the provided URL and name.

        Args:
            LLM_URL (str): The URL of the Language Model.
            LLM_NAME (str): The name of the Language Model.

        Returns:
            None

        """
        self.LLM_URL = LLM_URL
        self.LLM_NAME = LLM_NAME
        print(f" Language model URL: {LLM_URL}")
        print(f" Language model connected: {LLM_NAME}")

    def setCostumPrompt(self, userPrompt):
        """
        Sets the custom prompt for the chatbot.

        Parameters:
        - userPrompt (str): The custom prompt provided by the user.

        Returns:
        None
        """
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
        """
        Sends a query to the OpenAI endpoint for generating a response using the LLM model.

        Args:
            messages (list): A list of messages to be used as input for generating the response.
            model_temp (float, optional): The temperature parameter for controlling the randomness of the generated response. Defaults to 0.0.
            answer_size (int, optional): The maximum number of tokens in the generated response. Defaults to 100.
            presence_pen (float, optional): The presence penalty parameter for encouraging or discouraging the model to talk about specific topics. Defaults to 0.0.
            repeat_pen (float, optional): The repetition penalty parameter for discouraging the model from repeating the same phrases. Defaults to 0.0.

        Returns:
            str: The generated response from the LLM model.
        """
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
        pprint(data)
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
        """
        Sends a query to the OpenAI endpoint for language model evaluation.

        Args:
            messages (list): A list of messages to send to the language model.
            model_temp (float, optional): The temperature value for model sampling. Defaults to 0.0.
            answer_size (int, optional): The maximum number of tokens in the generated response. Defaults to 1.
            presence_pen (float, optional): The presence penalty value. Defaults to 0.0.
            repeat_pen (float, optional): The repeat penalty value. Defaults to 0.0.

        Returns:
            str: The generated response from the language model. Which is either 0 or 1.
        """
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
        """
        Answers a user query based on the given parameters.

        Args:
            query (str): The user query to be answered.
            rerank (bool, optional): Whether to rerank the documents based on the query. Defaults to False.
            prepost_context (bool, optional): Whether to include pre/post context in the answer. Defaults to False.
            lang (str, optional): The language of the query. Defaults to "EN".

        Returns:
            tuple: A tuple containing the result of the query, the contexts, and the context IDs.
        """
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
            {"role": "system", "content": self.PROMPT},
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

    def eval(self, method=None, only_queries=None, evaluator="sem_similarity"):
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
            if only_queries is not None:
                # Bypass rag pipeline run and just evaluate context relevance
                # betweeen queries and contexts
                scores = self.evaluate_context_relevance(only_queries, contexts=None)
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

        return matches
