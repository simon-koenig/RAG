# Imports
import sys

sys.path.append("./")
import re
from collections import Counter

import numpy as np
import requests
from config import LLM_URL
from tqdm import tqdm

# Specify LLM Judge name
LLM_NAME = "gemma2:latest"


def evaluate(
    rag_elements: list[dict],
    select: str = None,
    given_queries: list = None,
    evaluator: str = "ROUGE-1",
) -> dict:
    """
    Evaluate the given RAG elements based on the selected evaluation metric.

    Args:
        rag_elements (list): List of RAG elements containing contexts, questions, answers, etc.
        select (str, optional): The evaluation metric to use. Options are "context_relevance",
                                "faithfulness", "answer_relevance", "correctness", "all". Defaults to None.
        given_queries (list, optional): List of queries to evaluate context relevance. Defaults to None.
        evaluator (str, optional): The evaluator to use. Options are "sem-similarity", "llm-judge", "ROUGE-1". Defaults to "ROUGE-1".

    Returns:
        dict: A dictionary containing the evaluation scores for the selected metric.
    """
    scorer = None  # Optionally repace with sentence transformer
    # Select evaluation metric to run
    if select is None or select not in [
        "context_relevance",
        "faithfulness",
        "answer_relevance",
        "correctness",
        "all",
    ]:
        print("No evaluation metric selected")
        return {}

    if select == "context_relevance":
        if given_queries is not None:
            # Bypass RAG pipeline run and just evaluate context relevance
            # between given_queries and contexts
            cr_scores = evaluate_context_relevance(
                given_queries, contexts=None, evaluator=evaluator, scorer=scorer
            )
        else:
            contexts = [element["contexts"] for element in rag_elements]
            queries = [element["question"] for element in rag_elements]
            contexts_ids = [element["contexts_ids"] for element in rag_elements]
            cr_scores = evaluate_context_relevance(
                queries,
                contexts,
                contexts_ids=contexts_ids,
                evaluator=evaluator,
                scorer=scorer,
            )
        return {
            "context_relevance": cr_scores,
        }

    if select == "faithfulness":
        answers = [element["answer"] for element in rag_elements]
        contexts = [element["contexts"] for element in rag_elements]
        contexts_ids = [element["contexts_ids"] for element in rag_elements]
        f_scores = evaluate_faithfulness(
            answers,
            contexts,
            contexts_ids=contexts_ids,
            evaluator=evaluator,
            scorer=scorer,
        )
        return {
            "faithfulness": f_scores,
        }

    if select == "answer_relevance":
        queries = [element["question"] for element in rag_elements]
        answers = [element["answer"] for element in rag_elements]
        ar_scores = evaluate_answer_relevance(queries, answers, evaluator, scorer)
        return {"answer_relevance": ar_scores}

    if select == "correctness":
        answers = [element["answer"] for element in rag_elements]
        ground_truths = [element["ground_truth"] for element in rag_elements]
        c_scores = evaluate_correctness(answers, ground_truths, evaluator, scorer)
        return {"correctness": c_scores}

    if select == "all":
        queries = [element["question"] for element in rag_elements]
        contexts = [element["contexts"] for element in rag_elements]
        contexts_ids = [element["contexts_ids"] for element in rag_elements]

        answers = [element["answer"] for element in rag_elements]
        ground_truths = [element["ground_truth"] for element in rag_elements]

        cr_scores = evaluate_context_relevance(
            queries,
            contexts,
            evaluator=evaluator,
            contexts_ids=contexts_ids,
            scorer=scorer,
        )

        f_scores = evaluate_faithfulness(
            answers,
            contexts,
            evaluator=evaluator,
            contexts_ids=contexts_ids,
            scorer=scorer,
        )
        ar_scores = evaluate_answer_relevance(
            queries,
            answers,
            evaluator=evaluator,
            scorer=scorer,
        )
        c_scores = evaluate_correctness(
            answers,
            ground_truths,
            evaluator=evaluator,
            scorer=scorer,
        )

        return {
            "context_relevance": cr_scores,
            "faithfulness": f_scores,
            "answer_relevance": ar_scores,
            "correctness": c_scores,
        }


def evaluate_context_relevance(
    queries: list[str],
    contexts: list[list[str]] = None,
    contexts_ids: list[list[int]] = None,
    evaluator: str = "sem-similarity",
    vectorDB=None,
    scorer=None,
) -> np.ndarray:
    """
    Evaluate the context relevance of given queries and contexts.

    Args:
        queries (list[str]): List of queries to evaluate context relevance.
        contexts (list[list[str]], optional): List of contexts corresponding to each query. Defaults to None.
        contexts_ids (list[list[int]], optional): List of context IDs corresponding to each query. Defaults to None.
        evaluator (str, optional): The evaluator to use. Options are "sem-similarity", "llm-judge", "ROUGE-1". Defaults to "sem-similarity".
        vectorDB (optional): The vector database object to retrieve documents from index. Defaults to None.
        scorer (optional): The scorer object for semantic similarity evaluation. Defaults to None.

    Returns:
        np.ndarray: A 2D array containing the evaluation scores for context relevance.
    """
    # Type checking
    if not isinstance(queries, list):
        raise TypeError(
            f"Queries must be of type list, but got {type(queries).__name__}."
        )

    # Initialize scores array
    max_context_length = max(
        len(context) for context in contexts if context is not None
    )
    scores = np.zeros((len(queries), max_context_length), dtype=float)

    # Evaluate context relevance without pipeline run beforehand
    # In this case Please provide a vectorDB object to retrieve documents from index.
    # If no contexts are provided, retrieve top 3 documents from index based on query
    # for each document.

    if contexts is None:  # Extend context to list of nones to match queries length
        contexts = [None] * len(queries)

    # Loop over all queries with their respective contexts
    for i, (query, context) in tqdm(
        enumerate(zip(queries, contexts)), total=len(queries)
    ):
        if context is None:
            # Retrieve top 3 documents from index based on query
            context, ids = vectorDB.retrieveDocuments(query, 3)

        measurements = []
        # Loop over all contexts for a query
        for single_context in context:
            # Evaluate context relevance based on chosen evaluator
            if evaluator == "sem-similarity":
                measure = semantic_similarity(single_context, query, scorer)
            elif evaluator == "llm-judge":
                measure = llm_context_relevance(single_context, query)
            elif evaluator == "ROUGE-1":
                measure = ROUGE(single_context, query)
            # Convert measure to float and append to list
            measurements.append(round(float(measure), 3))

        # Extend measurements with None to match max_context_length if needed
        measurements.extend([None] * (max_context_length - len(measurements)))

        scores[i] = np.array(measurements)

    return scores


def llm_context_relevance(context: str, query: str) -> str:
    """
    Evaluates the relevance of a given context to a query using a language model.

    Args:
        context (str): The context to be evaluated.
        query (str): The query to which the context's relevance is to be evaluated.

    Returns:
        int: A relevance rating from 1 to 5, where:
            - 1 indicates the context is not relevant to the query at all.
            - 2 indicates the context is slightly relevant to the query.
            - 3 indicates the context is moderately relevant to the query.
            - 4 indicates the context is mostly relevant to the query.
            - 5 indicates the context is completely relevant to the query.
    """
    messages = [
        {
            "role": "user",
            "content": (
                " Given the following context and query, give a rating from 1 to 5."
                " Respond with 1 if the context is not relevant to the query at all."
                " Respond with 2 if the context is slightly relevant to the query."
                " Respond with 3 if the context is moderately relevant to the query."
                " Respond with 4 if the context is mostly relevant to the query."
                " Respond with 5 if the context is completely relevant to the query."
                ' Your response must strictly and only be a single integer from "1" to "5" and no additional text.'
                " Adhere to the examples:"
                ' If the context is "The pandemic, spanning the whole globe started in 2019." and the query is "What year did the corona pandemic start?", your response should be "5".'
                ' If the context is "The sky is blue" and the query is "What color is the sky?", your response should be "5".'
                ' If the context is "The sky is blue" and the query is "What is the weather like?", your response should be "4".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "What happens to water at high temperatures?", your response should be "4".'
                ' If the context is "The sky is blue" and the query is "What color is the sky usually?", your response should be "3".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "At what temperature does water usually boil?", your response should be "3".'
                ' If the context is "The sky is blue" and the query is "What color is the ocean?", your response should be "2".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "What is the boiling point of water in Fahrenheit?", your response should be "2".'
                ' If none of the nouns in the query are present in the context, the context is not relevant and your response should be "1".'
                ' If the context is "The sky is blue" and the query is "What color is the grass?", your response should be "1".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the query is '
                ' "What year did the corona pandemic start?", your response should be "1". '
                f'Here are the Context: "{context}" and the Query: "{query}".'
            ),
        },
    ]

    result = evalSendToLLM(messages)
    return result


def evaluate_faithfulness(
    answers: list[str],
    contexts: list[list[str]],
    evaluator: str = "ROUGE-1",
    contexts_ids: list[list[int]] = None,
    scorer=None,
) -> np.ndarray:
    """
    Evaluate the faithfulness of given answers with respect to their contexts.

    Args:
        answers (list[str]): List of answers to evaluate.
        contexts (list[list[str]]): List of contexts corresponding to each answer.
        evaluator (str, optional): The evaluator to use. Options are "sem-similarity", "llm-judge", "ROUGE-1". Defaults to "ROUGE-1".
        contexts_ids (list[list[int]], optional): List of context IDs corresponding to each answer. Defaults to None.
        scorer (optional): The scorer object for semantic similarity evaluation. Defaults to None.

    Returns:
        np.ndarray: A 2D array containing the evaluation scores for faithfulness.
    """
    # Type checking
    if not isinstance(answers, list):
        raise TypeError(
            f"Answers must be of type list, but got {type(answers).__name__}."
        )
    max_context_length = max(
        len(context) for context in contexts if context is not None
    )
    # Initialize scores array
    scores = np.zeros((len(answers), max_context_length), dtype=float)
    for i, (answer, context) in tqdm(
        enumerate(zip(answers, contexts)), total=len(answers)
    ):
        measurements = []
        for single_context in context:
            # Insert here evaluation measure of retrieved context
            if evaluator == "sem-similarity":
                measure = semantic_similarity(answer, single_context, scorer)
            elif evaluator == "llm-judge":
                measure = llm_faithfulness(answer, single_context)
            elif evaluator == "ROUGE-1":
                measure = ROUGE(answer, single_context)

            measurements.append(round(float(measure), 3))

        # Extend measurements with None to match max_context_length if needed
        measurements.extend([None] * (max_context_length - len(measurements)))
        scores[i] = np.array(measurements)

    return scores


def llm_faithfulness(context: str, answer: str) -> str:
    """
    Evaluates the faithfulness of a given answer with respect to a context using a language model.

    Args:
        context (str): The context to be evaluated.
        answer (str): The answer to which the context's faithfulness is to be evaluated.

    Returns:
        str: A faithfulness rating from 1 to 5, where:
            - 1 indicates the answer is not sufficiently grounded in the context at all.
            - 2 indicates the answer is slightly grounded in the context.
            - 3 indicates the answer is moderately grounded in the context.
            - 4 indicates the answer is mostly grounded in the context.
            - 5 indicates the answer is completely grounded in the context.
    """
    messages = [
        {
            "role": "user",
            "content": (
                " Given the following context and answer, give a rating from 1 to 5."
                " Respond with 1 if the answer is not sufficiently grounded in the context at all."
                " Respond with 2 if the answer is slightly grounded in the context."
                " Respond with 3 if the answer is moderately grounded in the context."
                " Respond with 4 if the answer is mostly grounded in the context."
                " Respond with 5 if the answer is completely grounded in the context."
                ' Your response must strictly and only be a single integer from "1" to "5" and no additional text.'
                " Adhere to the examples:"
                ' If the context is "The sky is blue" and the answer is "The sky is blue", your response should be "5".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water boils at 100 degrees Celsius", your response should be "5".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water boils at around 100 degrees Celsius", your response should be "4".'
                ' If the context is "The pandemic, spanning the whole globe started in 2019." and the answer is "The pandemic started in late 2019.", your response should be "4".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water boils at a high temperature", your response should be "3".'
                ' If the context is "The sky is blue" and the answer is "The sky is somewhat blue", your response should be "3".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the answer is "The pandemic was a significant global event.", your response should be "2".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the answer is "The pandemic caused many deaths.", your response should be "2".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water freezes at 0 degrees Celsius", your response should be "1".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the answer is "The pandemic started in 2019.", your response should be "1".'
                ' If none of the nouns in the answer are present in the context, the answer is not grounded and your response should be "1".'
                ' If the answer is "I do not know" or "there is no mention of {...} in the given context", the answer is not grounded and your response should be "1".'
                f'Here are the Context: "{context}" and the Answer: "{answer}".'
            ),
        },
    ]

    result = evalSendToLLM(messages)
    return result


def evaluate_answer_relevance(
    queries: list[str], answers: list[str], evaluator: str = "ROUGE-1", scorer=None
) -> np.ndarray:
    """
    Evaluate the relevance of given answers with respect to their queries.

    Args:
        queries (list[str]): List of queries to evaluate answer relevance.
        answers (list[str]): List of answers to evaluate.
        evaluator (str, optional): The evaluator to use. Options are "sem-similarity", "llm-judge", "ROUGE-1". Defaults to "ROUGE-1".
        scorer (optional): The scorer object for semantic similarity evaluation. Defaults to None.

    Returns:
        np.ndarray: An array containing the evaluation scores for answer relevance.
    """
    # Type checking
    if not isinstance(answers, list):
        raise TypeError(
            f"Answers must be of type list, but got {type(answers).__name__}."
        )

    if not isinstance(queries, list):
        raise TypeError(
            f"Queries must be of type list, but got {type(queries).__name__}."
        )

    # Initialize scores array
    scores = np.zeros(len(answers), dtype=float)
    for i, (answer, query) in tqdm(
        enumerate(zip(answers, queries)), total=len(answers)
    ):
        # Evaluate answer relevance based on chosen evaluator
        if evaluator == "sem-similarity":
            measure = semantic_similarity(answer, query, scorer=scorer)
        elif evaluator == "llm-judge":
            measure = llm_answer_relevance(answer, query)
        elif evaluator == "ROUGE-1":
            measure = ROUGE(answer, query)
        # Convert measure to float and append to list
        scores[i] = round(float(measure), 3)
    return scores


def llm_answer_relevance(answer: str, query: str) -> str:
    """
    Evaluates the relevance of an answer with respect to a query using a language model.

    Args:
        answer (str): The answer to be evaluated.
        query (str): The query to which the answer's relevance is being evaluated.

    Returns:
        int: A relevance rating from 1 to 5, where:
            - 1 indicates the answer is not relevant to the query at all.
            - 2 indicates the answer is slightly relevant to the query.
            - 3 indicates the answer is moderately relevant to the query.
            - 4 indicates the answer is mostly relevant to the query.
            - 5 indicates the answer is highly relevant to the query.
    """
    messages = [
        {
            "role": "user",
            "content": (
                'You are a rating judge on the relevance of a given "answer" with respect to a "query."'
                ' Given the following "query" and "answer", give a rating from 1 to 5.'
                " Your rating system is as follows:"
                ' Respond with 1 if the "answer" is not relevant to the "query" at all.'
                ' Respond with 2 if the "answer" is slightly relevant to the "query".'
                ' Respond with 3 if the "answer" is moderately relevant to the "query".'
                ' Respond with 4 if the "answer" is mostly relevant to the "query".'
                ' Respond with 5 if the "answer" is highly relevant to the "query".'
                ' Your response must strictly and only be a single integer from "1" to "5" and no additional text.'
                " Adhere to the examples:"
                ' If the "query" is "At what temperature does water boil?" and the "answer" is "Water boils at 100 degrees Celsius", your response should be "5".'
                ' If the "query" is "What is the capital of Germany?" and the "answer" is "Berlin is a major city in Germany.", your response should be "4".'
                ' If the "query" is "How many moons does the Earth have?" and the "answer" is "The Earth has at least one moon", your response should be "3".'
                ' If the "query" is "What is the capital of Germany?" and the "answer" is "Berlin is known for its history.", your response should be "2".'
                ' If the "query" is "What year did the corona pandemic start?" and the "answer" is "The pandemic, was a global event and lead to many deaths.", your response should be "1".'
                ' If the "query" is "What is the capital of Germany?" and the "answer" is "Germany is a country in Europe.", your response should be "1".'
                ' If none of the nouns in the "answer" are present in the "query", the "answer" is not relevant, your response should be "1".'
                ' If the "answer" is "I do not know" or "there is no mention of {...} in the given context", your response should be "1".'
                f'Here are the "Query": "{query}" and the "Answer": "{answer}".'
            ),
        },
    ]
    result = evalSendToLLM(messages)
    return result


def evaluate_correctness(
    answers: list,
    ground_truths: list,
    evaluator: str = "ROUGE-1",
    scorer: object = None,
) -> np.ndarray:
    """
    Evaluates the correctness of given answers against ground truths using the specified evaluator.

    Args:
        answers (list): A list of answers to be evaluated.
        ground_truths (list): A list of ground truth values to compare against.
        evaluator (str, optional): The method to use for evaluation. Options are "sem-similarity", "llm-judge", and "ROUGE-1". Defaults to "ROUGE-1".
        scorer (object, optional): An optional scorer object to be used with the "sem-similarity" evaluator. Defaults to None.

    Raises:
        TypeError: If `answers` or `ground_truths` are not of type list.

    Returns:
        np.ndarray: An array of scores representing the correctness of each answer.
    """

    # Type checking
    if not isinstance(answers, list):
        raise TypeError(
            f"Answers must be of type list, but got {type(answers).__name__}."
        )

    if not isinstance(ground_truths, list):
        raise TypeError(
            f"Queries must be of type list, but got {type(ground_truths).__name__}."
        )

    # Initialize scores array
    scores = np.zeros(len(answers), dtype=float)
    for i, (answer, ground_truth) in tqdm(
        enumerate(zip(answers, ground_truths)), total=len(answers)
    ):
        # print(f"Answer: {answer}")
        # print(f"Ground truth: {ground_truth}")
        if evaluator == "sem-similarity":
            measure = semantic_similarity(answer, ground_truth, scorer=scorer)
        elif evaluator == "llm-judge":
            measure = llm_correctness(answer, ground_truth)
        elif evaluator == "ROUGE-1":
            measure = ROUGE(answer, ground_truth)
        scores[i] = round(float(measure), 3)
    # print(f" Scores: {scores}")
    return scores


def llm_correctness(answer: str, ground_truth: str) -> str:
    """
    Evaluates the correctness of an answer with respect to a ground truth using a language model.

    Args:
        answer (str): The answer to be evaluated.
        ground_truth (str): The ground truth to which the answer's correctness is being evaluated.

    Returns:
        str: A correctness rating from 1 to 5, where:
            - 1 indicates the answer is not correct based on the ground truth.
            - 2 indicates the answer is slightly correct based on the ground truth.
            - 3 indicates the answer is moderately correct based on the ground truth.
            - 4 indicates the answer is mostly correct based on the ground truth.
            - 5 indicates the answer is completely correct based on the ground truth.
    """
    messages = [
        {
            "role": "user",
            "content": (
                'You are a rating judge for the correctness of a given "answer" with respect to a "ground-truth."'
                " Your rating system is as follows:"
                ' Respond with "1" if the "answer" is not correct based on the "ground-truth".'
                ' Respond with "2" if the "answer" is slightly correct based on the "ground-truth".'
                ' Respond with "3" if the "answer" is moderately correct based on the "ground-truth".'
                ' Respond with "4" if the "answer" is mostly correct based on the "ground-truth".'
                ' Respond with "5" if the "answer" is completely correct based on the "ground-truth".'
                " Adhere to the examples:"
                ' If the "answer" is "yes" and the "ground-truth" is "yes", your response should be "5".'
                ' If the "answer" is "The sky is clear and blueish" and the "ground-truth" is "The sky is blue", your response should be "4".'
                ' If the "answer" is "The sky is somewhat blue" and the "ground-truth" is "The sky is blue", your response should be "3".'
                ' If the "answer" is "The sky is cloudy and the "ground-truth" is "The sky is blue", your response should be "2".'
                ' If none of the nouns in the "answer" are present in the "ground-truth", the "answer" is not correct. Thus your response should be "1".'
                ' If the "answer" is "yes" and the "ground-truth" is "no", your response should be "1".'
                ' If the "answer" is "no" and the "ground-truth" is "yes", your response should be "1".'
                ' If the "answer" includes "there is no mention of {...} in the given context" and the "ground-truth" is "yes" or "no", your response should be "1".'
                ' If the "answer" includes "I do not know", your response should be "1".'
                ' Given the following "answer" and "ground-truth", give a rating from 1 to 5.'
                ' Your response must strictly and only be a single integer from "1" to "5" and no additional text.'
                f'Here are the "Answer": "{answer}" and the "ground-truth": "{ground_truth}".'
            ),
        },
    ]

    result = evalSendToLLM(messages)

    return result


# Function to calculate BERTScore semantic similarity, scorer has to be a sentence transformer model
def semantic_similarity(candidate: str, reference: str, scorer=None) -> float:
    """
    Calculate the semantic similarity between a candidate and a reference text using BERTScore.

    Args:
        candidate (str): The candidate text to be evaluated.
        reference (str): The reference text to compare against.
        scorer (optional): The scorer object for BERTScore evaluation. Defaults to None.

    Returns:
        float: The mean recall score from BERTScore.
    """
    # BERTScore calculation
    P, R, F1 = scorer.score([candidate.lower()], [reference.lower()])
    return R.mean()


# Function to calculate ROUGE-N (ROUGE-1, ROUGE-2, etc.)
def ROUGE(candidate: str, reference: str, n: int = 1) -> float:
    """
    Calculate ROUGE-N score between candidate and reference texts.

    Args:
        candidate (str): The candidate text to be evaluated.
        reference (str): The reference text to compare against.
        n (int, optional): The n-gram length to use for ROUGE calculation. Defaults to 1.

    Returns:
        float: The recall score for the ROUGE-N metric.
    """

    # Helper function to tokenize sentences into words
    def tokenize(sentence: str) -> list[str]:
        return re.findall(r"\w+", sentence.lower())

    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    ref_ngrams = [
        " ".join(reference_tokens[i : i + n])
        for i in range(len(reference_tokens) - n + 1)
    ]
    cand_ngrams = [
        " ".join(candidate_tokens[i : i + n])
        for i in range(len(candidate_tokens) - n + 1)
    ]

    ref_count = Counter(ref_ngrams)
    cand_count = Counter(cand_ngrams)

    overlap = sum((cand_count & ref_count).values())
    total_ref = sum(ref_count.values())

    recall = overlap / total_ref if total_ref > 0 else 0

    return recall


def evalSendToLLM(
    messages: list[dict],
    LLM_NAME: str = LLM_NAME,
    LLM_URL: str = LLM_URL,
    model_temp: float = 0.0,
    answer_size: int = 1,
    presence_pen: float = 0.0,
    repeat_pen: float = 0.0,
) -> str:
    """
    Sends a query to the OpenAI endpoint for language model evaluation.

    Args:
        messages (list[dict]): A list of messages to send to the language model.
        LLM_NAME (str, optional): The name of the language model to use. Defaults to LLM_NAME.
        LLM_URL (str, optional): The URL of the language model endpoint. Defaults to LLM_URL.
        model_temp (float, optional): The temperature value for model sampling. Defaults to 0.0.
        answer_size (int, optional): The maximum number of tokens in the generated response. Defaults to 1.
        presence_pen (float, optional): The presence penalty value. Defaults to 0.0.
        repeat_pen (float, optional): The repeat penalty value. Defaults to 0.0.

    Returns:
        str: The generated response from the language model, which is either 1, 2, 3, 4, or 5.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer N/A",
    }
    data = {
        "model": LLM_NAME,
        "messages": messages,
        "temperature": model_temp,
        "max_tokens": answer_size,
        "presence_penalty": presence_pen,
        "repeat_penalty": repeat_pen,
    }
    endpoint = LLM_URL + "/chat/completions"
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code != 200:
        print(f"LLM failed with status code: {response.status_code}")
        raise Exception("LLM failed")
    report = response.json()
    if "choices" in report and len(report["choices"]) > 0:
        result = report["choices"][0]["message"]["content"]
    else:
        result = "No result generated!"
    return result


def eval_context_goldPassages(
    rag_elements: list[dict], goldPassages: list[list]
) -> list[list]:
    """
    Evaluate context relevance with gold passages.

    Args:
        rag_elements (list): A list of dictionaries, where each dictionary contains:
            - "question" (str): The query question.
            - "contexts_ids" (list): A list of context IDs.
        goldPassages (list): A list of lists, where each inner list contains the gold passage IDs for the corresponding query.

    Returns:
        list: A list of lists, where each inner list contains:
            - The number of matches between context IDs and gold passage IDs.
            - The number of gold passage IDs for the corresponding query.
    """
    # Evaluate context relevance with goldPassages
    # contexts = [element["contexts"] for element in rag_elements]
    queries = [element["question"] for element in rag_elements]
    contexts_ids = [element["contexts_ids"] for element in rag_elements]

    matches = []
    for query, context_ids, goldPs in zip(queries, contexts_ids, goldPassages):
        print(f"Context_ids: {context_ids}")
        print(f"goldPasssageContexts: {goldPs}")

        # Count number of matches in context_ids and goldPs
        # Beware, that the number of elements in goldPs per query varies.
        set_goldPs = set(goldPs)
        set_context_ids = set(context_ids)
        number_matches = sum(1 for element in set_context_ids if element in set_goldPs)
        print(f"Query: {query}")
        print(f"Number of matches: {number_matches}")
        print(f"Number of goldPs: {len(goldPs)}")
        matches.append([number_matches, len(goldPs)])

    return matches
