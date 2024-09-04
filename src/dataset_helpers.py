# Dataset Preprations for RAG pipeline
import pandas as pd
from datasets import load_dataset


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

        # Clean queries and ground truths and store them in lists
        queries = []
        ground_truths = []
        for query, ground_truth in zip(QA["question"], QA["answer"]):
            query = query.replace("\n", " ")
            query = query.replace("\t", " ")
            queries.append(query)
            ground_truth = ground_truth.replace("\n", " ")
            ground_truth = ground_truth.replace("\t", " ")
            ground_truths.append(ground_truth)

        return corpus_list, queries, ground_truths

    def loadMiniBiosqa(self):
        # Load MiniBioasq dataset
        print("Loading MiniBioasq dataset")
        corpus = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")["test"]
        # Create a list of dictionaries with keys: passage, id
        corpus_list = []
        for passage, iD in zip(corpus["passage"], corpus["id"]):
            corpus_list.append({"text": passage, "id": iD})

        # Load QA dataset
        QA = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")[
            "train"
        ]

        # Clean queries and ground truths and store them in lists
        queries = []
        ground_truths = []
        for query, ground_truth in zip(QA["question"], QA["answer"]):
            query = query.replace("\n", " ")
            query = query.replace("\t", " ")
            queries.append(query)
            ground_truth = ground_truth.replace("\n", " ")
            ground_truth = ground_truth.replace("\t", " ")
            ground_truths.append(ground_truth)

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
