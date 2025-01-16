# Dataset Preprations for RAG pipeline
import pandas as pd
from datasets import load_dataset


class DatasetHelpers:
    """
    Helper functions for loading and processing various datasets for the RAG pipeline.
    """

    def __init__(self, chunking_params: dict = None):
        """
        Initialize the DatasetHelpers class.

        Args:
            chunking_params (dict, optional): Parameters for chunking the dataset. Defaults to None.
        """
        self.chunking_params = chunking_params

    def loadFromDocuments(self, documents: list) -> None:
        """
        Load from a corpus of PDF documents.

        Args:
            documents (list): List of document paths.
        """
        pass

    def loadSQUAD(self) -> None:
        """
        Load the SQUAD dataset.
        """
        pass

    def loadTriviaQA(self) -> None:
        """
        Load the TriviaQA dataset.
        """
        pass

    def loadHotpotQA(self) -> None:
        """
        Load the HotpotQA dataset.
        """
        pass

    def loadNaturalQuestions(self) -> None:
        """
        Load the NaturalQuestions dataset.
        """
        pass

    def loadMiniWiki(self) -> tuple:
        """
        Load the MiniWiki dataset.

        Returns:
            tuple: A tuple containing the corpus list, queries, ground truths, and None.
        """
        print("Loading MiniWiki dataset")
        corpus = load_dataset("rag-datasets/mini_wikipedia", "text-corpus")["passages"]
        corpus_list = [
            {"text": passage, "id": iD}
            for passage, iD in zip(corpus["passage"], corpus["id"])
        ]

        QA = load_dataset("rag-datasets/mini_wikipedia", "question-answer")["test"]
        queries = [
            query.replace("\n", " ").replace("\t", " ") for query in QA["question"]
        ]
        ground_truths = [
            ground_truth.replace("\n", " ").replace("\t", " ")
            for ground_truth in QA["answer"]
        ]

        return corpus_list, queries, ground_truths, None

    def loadMiniBiosqa(self) -> tuple:
        """
        Load the MiniBioasq dataset.

        Returns:
            tuple: A tuple containing the corpus list, queries, ground truths, and gold passages.
        """
        print("Loading MiniBioasq dataset")
        corpus = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")["test"]
        corpus_list = [
            {"text": passage, "id": iD}
            for passage, iD in zip(corpus["passage"], corpus["id"])
        ]

        QA = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")[
            "train"
        ]
        queries = [
            query.replace("\n", " ").replace("\t", " ") for query in QA["question"]
        ]
        ground_truths = [
            ground_truth.replace("\n", " ").replace("\t", " ")
            for ground_truth in QA["answer"]
        ]
        goldPassages = QA["relevant_passage_ids"]

        return corpus_list, queries, ground_truths, goldPassages

    def loadQM(self) -> tuple:
        """
        Load the QM dataset.

        Returns:
            tuple: A tuple containing the corpus list, queries, and ground truths.
        """
        print("Loading AIT QM dataset")

        corpus_list = None
        ground_truths = None

        df = pd.read_excel("./data/100_questions.xlsx", skiprows=0, usecols=[1])
        queries = df.iloc[:, 0].tolist()

        return corpus_list, queries, ground_truths
