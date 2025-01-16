# Main object for RAG pipeline

import concurrent.futures

import requests
from tqdm import tqdm
from utils import clean_sentence


class RagPipe:
    def __init__(self):
        self.PROMPT_EN = (
            "You are a helpful assisstant. Context information is given in the following text. "
            "Use the information from the context instead of pretrained knowledge to answer the question."
        )
        self.PROMPT_DE = (
            "Sie sind ein hilfreicher Assistent. Antworten Sie auf Deutsch. Kontextinformationen sind im folgenden Text enthalten. "
            "Verwenden Sie ausschließlich Informationen aus dem Kontext, um die Frage zu beantworten. "
            'Sagen Sie nichts wie "Basierend auf dem gegebenen Kontext", geben Sie einfach die Antwort. '
            "Wenn Sie sich unsicher sind, müssen Sie das sagen. Begründen Sie Ihre Antwort, indem Sie nur "
            "indem Sie sich auf den gegebenen Kontext beziehen. "
            "Die Antwort sollte explizit und erläuternd sein. "
            "Außer es ist eine Ja- oder Nein-Frage, dann antworten Sie mit Ja oder Nein."
        )
        # Default to english
        self.PROMPT = self.PROMPT_EN

        # Default configurations
        self.query_expansion = False
        self.rerank = False
        self.prepost_context = False
        self.background_reversed = False
        self.lang = "EN"
        self.search_ref_lex = 2
        self.search_ref_sem = 2
        self.num_ref_lim = 2
        self.model_temp = 0.1
        self.answer_token_num = 50

    def setConfigs(
        self,
        query_expansion=False,
        rerank=True,
        prepost_context=False,
        background_reversed=False,
        lang="EN",
        search_ref_lex=2,
        search_ref_sem=2,
        num_ref_lim=4,
        model_temp=0.1,
        answer_token_num=50,
    ):
        """
        Sets the configurations for the pipeline.

        Args:
            query_expansion (int): The number of query expansions.
            rerank (bool): Whether to rerank the documents based on the query.
            prepost_context (bool): Whether to include pre/post context in the answer.
            background_reversed (bool): Whether to reverse the background.
            lang (str): The language of the query.
            search_ref_lex (int): The number of lexical references to search for.
            search_ref_sem_ (int): The number of semantic references to search for.
            num_ref_lim (int): The number of references to limit.

        Returns:
            None
        """
        self.query_expansion = query_expansion
        self.rerank = rerank
        self.prepost_context = prepost_context
        self.background_reversed = background_reversed
        self.lang = lang
        self.search_ref_lex = search_ref_lex
        self.search_ref_sem = search_ref_sem
        self.num_ref_lim = num_ref_lim
        self.model_temp = model_temp
        self.answer_token_num = answer_token_num

    def getConfigs(self):
        """
        Returns the configurations for the pipeline.

        Returns:
            dict: A dictionary containing the configurations for the pipeline.
        """
        return {
            "query_expansion": self.query_expansion,
            "rerank": self.rerank,
            "prepost_context": self.prepost_context,
            "background_reversed": self.background_reversed,
            "lang": self.lang,
            "search_ref_lex": self.search_ref_lex,
            "search_ref_sem": self.search_ref_sem,
            "num_ref_lim": self.num_ref_lim,
            "model_temp": self.model_temp,
            "answer_token_num": self.answer_token_num,
        }

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
            "temperature": self.model_temp,
            "max_tokens": self.answer_token_num,
            # "presence_penalty": presence_pen,
            # "repeat_penalty": repeat_pen,
        }
        endpoint = self.LLM_URL + "/chat/completions"
        # print("Sending query to OpenAI endpoint: " + endpoint)

        # pprint(data)
        # Have neested try block to handle connection errors
        try:
            report = requests.post(endpoint, headers=headers, json=data)
            # pprint(report)
            if report.status_code != 200:
                print(f"Response not as expected. Status code: {report.status_code}")
            report = report.json()
        except Exception as e:
            print(f" Error: {e} ")
            try:
                report = requests.post(endpoint, headers=headers, json=data)
                if report.status_code != 200:
                    print(
                        f"Response not as expected. Status code: {report.status_code}"
                    )
                # pprint(report)
                report = report.json()
            except Exception as e:
                print(f"Error in second try: {e}")

        # print("Received response...")
        if "choices" in report:
            if len(report["choices"]) > 0:  # Always take the first choice.
                result = report["choices"][0]["message"]["content"]
            else:
                result = "No result generated!"
        else:
            result = report

        return result

    def answerQuery(
        self,
        query,
    ):
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
        # print("Waiting for background!")

        try:
            background, contexts, contexts_ids = self.DB.getBackground(
                query,
                search_ref_lex=self.search_ref_lex,
                search_ref_sem=self.search_ref_sem,
                num_ref_lim=self.num_ref_lim,
                lang=self.lang,
                rerank=self.rerank,
                prepost_context=self.prepost_context,
                background_reversed=self.background_reversed,
                query_expansion=self.query_expansion,
                LLM_URL=self.LLM_URL,
                LLM_NAME=self.LLM_NAME,
            )
        except Exception as e:
            print(f"Error: {e}")
            print("Trying again!")
            background, contexts, contexts_ids = self.DB.getBackground(
                query,
                search_ref_lex=self.search_ref_lex,
                search_ref_sem=self.search_ref_sem,
                num_ref_lim=self.num_ref_lim,
                lang=self.lang,
                rerank=self.rerank,
                prepost_context=self.prepost_context,
                background_reversed=self.background_reversed,
                query_expansion=self.query_expansion,
                LLM_URL=self.LLM_URL,
                LLM_NAME=self.LLM_NAME,
            )
        if not background:
            # print("No background received!")
            raise ValueError("No background received!")
        # Update language prompt for LLM
        if self.lang == "DE":
            self.PROMPT = self.PROMPT_DE
        if self.lang == "EN":
            self.PROMPT = self.PROMPT_EN

        # Tell llm again to obey instructions
        enforce_query = (
            "The answer has to mention explicit details, be explainatory, be short and give a brief reasoning. The answer has to refer to the query."
            "If the answer is not provided in the context. You must say that you dont know the answer."
            'Do not include sentences like "According to the given context" or "Based on the context".'
            "Using only information from the provided context and not prior knowledge, answer the following user query: "
            + query
        )
        messages = [
            {"role": "system", "content": self.PROMPT},
            {"role": "assistant", "content": background},
            {"role": "user", "content": enforce_query},
        ]

        result = self.sendToLLM(messages)
        # Check difference background and context
        # pprint(contexts)
        # pprint(background)
        return result, contexts, contexts_ids

    def run(
        self,
        questions,
        ground_truths=None,
        goldPassagesIds=None,
        corpus_list=None,
        newIngest=False,
        maxDocs=1000000,
        nThreads=1,
    ):
        # Run RAG pipeline for questions, ground_truhts and corpus_list
        if newIngest:
            self.DB.emptyIndex()
            print("Index emptied")
            print("Start indexing documents. Please wait. ")
            # Ask user if they are certain they want to ingest new documents
            if input("Are you sure you want to ingest new documents? (y/n): ") == "y":
                self.DB.indexDocuments(documents=corpus_list, maxDocs=maxDocs)
                print(f" You are using index: {self.DB.indexName}")
                print(f"Done! Index Stats:  {self.DB.getIndexStats()}")
            else:
                print("No new documents ingested.")
        elif not (newIngest and corpus_list):
            print("Using already indexed documents.")
            print(f" You are using index: {self.DB.indexName}")
            print(f"Index Stats:  {self.DB.getIndexStats()}")

        # Create a list of dictionaries with keys: question, answer, contexts, context_ids, ground_truth

        # print("Start answering queries. Please wait. ")

        self.rag_elements = []
        if ground_truths is None:
            # print("No ground truths given!")
            ground_truths = [None] * len(questions)

        # If goldPassagesIds are given and ground truths the same length as questions
        # if goldPassagesIds is None, create a list of None values
        if len(questions) > len(ground_truths):
            ground_truths = ground_truths + [None] * (
                len(questions) - len(ground_truths)
            )
        elif len(ground_truths) > len(questions):
            ground_truths = ground_truths[: len(questions)]

        if goldPassagesIds is None:
            # print("No goldPassages given!")
            goldPassagesIds = [None] * len(questions)

        if len(questions) > len(goldPassagesIds):
            goldPassagesIds = goldPassagesIds + [None] * (
                len(questions) - len(goldPassagesIds)
            )
        elif len(goldPassagesIds) > len(questions):
            goldPassagesIds = goldPassagesIds[: len(questions)]

        # Check if questions, ground_truths and goldPassages have the same length
        if not len(questions) == len(ground_truths) == len(goldPassagesIds):
            raise ValueError(
                "Questions, ground_truths and goldPassages must have the same length."
            )

        # Create list of list of rag elements. Every rag element is a dictionary
        # containing the question, answer, contexts, context_ids and ground_truth
        for question, ground_truth, goldPassages in zip(
            questions,
            ground_truths,
            goldPassagesIds,
        ):
            self.rag_elements.append(
                {
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "contexts_ids": [],
                    "ground_truth": ground_truth,
                    "goldPassages": goldPassages,
                }
            )

        # Iterate over the rag elements and get the answer from the LLM model and the contexts from the Vector DB
        # size = len(self.rag_elements)

        # Define helper function to process rag elements in parallel
        def process_rag_element(rag_element):
            # Get answer from RAG system
            query = clean_sentence(rag_element["question"])
            try:
                llmanswer, contexts, contexts_ids = self.answerQuery(
                    query=query,
                )
            except Exception as e:
                print(f"Error: {e}")
                print(f"Currrent question: {rag_element['question']}")
                print("Could not answer question. Skipping to next question.")

            # Clean llmanswer
            llmanswer = clean_sentence(llmanswer)

            # Clean contexts
            contexts = [clean_sentence(context) for context in contexts]

            rag_element["answer"] = llmanswer
            rag_element["contexts"] = contexts
            rag_element["contexts_ids"] = contexts_ids

            return rag_element

        # Process rag elements in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=nThreads) as executor:
            results = list(
                tqdm(
                    executor.map(process_rag_element, self.rag_elements),
                    total=len(self.rag_elements),
                )
            )
        self.rag_elements = results
