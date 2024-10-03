# Imports
import re
from collections import Counter

import numpy as np
import requests
from bert_score import BERTScorer
from csv_helpers import read_pipe_results_from_csv, write_eval_results_to_csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Constants
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3.1:latest"


def evaluate(rag_elements, method=None, given_queries=None, evaluator="sem_similarity"):
    if evaluator == "sem_similarity":
        scorer = BERTScorer(model_type="roberta-base")
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
    # print(f"Running evaluation for method: {method}")
    # print(f"Using evaluator: {evaluator}")

    if method == "context_relevance":
        if given_queries is not None:
            # Bypass rag pipeline run and just evaluate context relevance
            # betweeen given_queries and contexts
            scores = evaluate_context_relevance(
                given_queries, contexts=None, evaluator=evaluator, scorer=scorer
            )
        else:
            contexts = [element["contexts"] for element in rag_elements]
            queries = [element["question"] for element in rag_elements]
            contexts_ids = [element["contexts_ids"] for element in rag_elements]
            scores = evaluate_context_relevance(
                queries,
                contexts,
                contexts_ids=contexts_ids,
                evaluator=evaluator,
                scorer=scorer,
            )
        return scores

    if method == "faithfulness":
        answers = [element["answer"] for element in rag_elements]
        contexts = [element["contexts"] for element in rag_elements]
        contexts_ids = [element["contexts_ids"] for element in rag_elements]
        scores = evaluate_faithfulness(
            answers,
            contexts,
            contexts_ids=contexts_ids,
            evaluator=evaluator,
            scorer=scorer,
        )
        return scores

    if method == "answer_relevance":
        queries = [element["question"] for element in rag_elements]
        answers = [element["answer"] for element in rag_elements]
        scores = evaluate_answer_relevance(queries, answers, evaluator)
        return scores

    if method == "correctness":
        answers = [element["answer"] for element in rag_elements]
        ground_truths = [element["ground_truth"] for element in rag_elements]
        scores = evaluate_correctness(answers, ground_truths, evaluator, scorer)
        return scores

    if method == "all":
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
        ar_scores = evaluate_answer_relevance(queries, answers, evaluator, scorer)
        c_scores = evaluate_correctness(answers, ground_truths, evaluator, scorer)

        return {
            "context_relevance": cr_scores,
            "faithfulness": f_scores,
            "answer_relevance": ar_scores,
            "correctness": c_scores,
        }


def evaluate_context_relevance(
    queries,
    contexts=None,
    contexts_ids=None,
    goldPassages=None,
    evaluator="sem_similarity",
    vectorDB=None,
    scorer=None,
):
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
    # scores = []
    ##
    ## Evaluate context relevance without pipeline run beforehand
    ## In this case Please provide a vectorDB object to retrieve documents from index.
    ##
    ## If no contexts are provided, retrieve top 3 documents from index based on query
    ## for each document.

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
            # print(f"Context: {single_context}")

            # Evaluate context relevance based on chosen evaluator
            if evaluator == "sem_similarity":
                measure = semantic_similarity(single_context, query, scorer)
            elif evaluator == "llm_judge":
                measure = llm_binary_context_relevance(single_context, query)
            elif evaluator == "ROUGE-2":
                measure = ROUGE(single_context, query)
            # Convert measure to float and append to list
            # print(f"Measure: {measure}")
            measurements.append(round(float(measure), 3))
        # Compute mean context relevance over all contexts per query

        # measurements = np.array(measurements)
        # print(f"Measurements: {measurements}")

        # Extend measurements with None to match max_context_length if needed
        measurements.extend([None] * (max_context_length - len(measurements)))

        scores[i] = np.array(measurements)

        # scores.append(measurements)

    return scores


def llm_binary_context_relevance(context, query):
    messages = [
        {
            "role": "user",
            "content": (
                "Given the following context and query,"
                " Give a rating from 1 to 5."
                " Respond with 1 if the context is not relevant to the query at all."
                " Respond with 2 if the context is slightly relevant to the query."
                " Respond with 3 if the context is moderately relevant to the query."
                " Respond with 4 if the context is mostly relevant to the query."
                " Respond with 5 if the context is completely relevant to the query."
                ' Your response must strictly and only be a single integer from "1" to "5" and no additional text.'
                " Some Examples:"
                ' If none of the nouns in the query are present in the context, the context is not relevant and your response should be "1".'
                ' If the context is "The sky is blue" and the query is "What color is the grass?", your response should be "1".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "At what temperature does water freeze?", your response should be "1".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the query is "What year did the corona pandemic start?", your response should be "1".'
                ' If the context is "The pandemic, spanning the whole globe started in 2019." and the query is "What year did the corona pandemic start?", your response should be "5".'
                ' If the context is "The sky is blue" and the query is "What color is the sky?", your response should be "5".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "At what temperature does water boil?", your response should be "5".'
                ' If the context is "The sky is blue" and the query is "What is the weather like?", your response should be "4".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "What happens to water at high temperatures?", your response should be "4".'
                ' If the context is "The sky is blue" and the query is "What color is the sky usually?", your response should be "3".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "At what temperature does water usually boil?", your response should be "3".'
                ' If the context is "The sky is blue" and the query is "What color is the ocean?", your response should be "2".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "What is the boiling point of water in Fahrenheit?", your response should be "2".'
                f'Here are the Context: "{context}" and the Query: "{query}".'
            ),
        },
    ]

    result = evalSendToLLM(messages)
    return result


def evaluate_faithfulness(
    answers, contexts, evaluator="sem_similarity", contexts_ids=None, scorer=None
):
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
            # print(f"Context: {single_context}")  #
            if evaluator == "sem_similarity":
                measure = semantic_similarity(answer, single_context, scorer)
            elif evaluator == "llm_judge":
                measure = llm_binary_faithfulness(answer, single_context)
            elif evaluator == "ROUGE-2":
                measure = ROUGE(answer, single_context)

            measurements.append(round(float(measure), 3))

        # Extend measurements with None to match max_context_length if needed
        measurements.extend([None] * (max_context_length - len(measurements)))
        scores[i] = np.array(measurements)

    return scores


def llm_binary_faithfulness(context, answer):
    messages = [
        {
            "role": "user",
            "content": (
                "Given the following context and answer,"
                " Give a rating from 1 to 5."
                " Respond with 1 if the answer is not sufficiently grounded in the context at all."
                " Respond with 2 if the answer is slightly grounded in the context."
                " Respond with 3 if the answer is moderately grounded in the context."
                " Respond with 4 if the answer is mostly grounded in the context."
                " Respond with 5 if the answer is completely grounded in the context."
                ' Your response must strictly and only be a single integer from "1" to "5" and no additional text.'
                " Some Examples:"
                ' If none of the nouns in the answer are present in the context, the answer is not grounded and your response should be "1".'
                ' If the answer is "I do not know" or "there is no mention of {...} in the given context", the answer is not grounded and your response should be "1".'
                ' If the context is "The sky is blue" and the answer is "The sky is blue", your response should be "5".'
                ' If the context is "The sky is blue" and the answer is "The grass is green", your response should be "1".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water boils at 100 degrees Celsius", your response should be "5".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water freezes at 0 degrees Celsius", your response should be "1".'
                ' If the context is "The pandemic, spanning the whole globe started in 2019." and the answer is "The pandemic started in 2019.", your response should be "5".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the answer is "The pandemic started in 2019.", your response should be "1".'
                ' If the context is "The sky is blue" and the answer is "The sky is somewhat blue", your response should be "3".'
                ' If the context is "The sky is blue" and the answer is "The sky is clear and blue", your response should be "4".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water boils at around 100 degrees Celsius", your response should be "4".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water boils at a high temperature", your response should be "3".'
                ' If the context is "The pandemic, spanning the whole globe started in 2019." and the answer is "The pandemic started in late 2019.", your response should be "4".'
                ' If the context is "The pandemic, spanning the whole globe started in 2019." and the answer is "The pandemic started in the year 2019.", your response should be "5".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the answer is "The pandemic was a significant global event.", your response should be "2".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the answer is "The pandemic caused many deaths.", your response should be "2".'
                f'Here are the Context: "{context}" and the Answer: "{answer}".'
            ),
        },
    ]

    result = evalSendToLLM(messages)
    return result


def evaluate_answer_relevance(
    queries, answers, evaluator="sem_similarity", scorer=None
):
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
        # print(f"Answer: {answer}")
        # print(f"Query: {query}")
        # Evaluate context relevance based on chosen evaluator
        if evaluator == "sem_similarity":
            measure = semantic_similarity(answer, query, scorer=scorer)
        elif evaluator == "llm_judge":
            measure = llm_binary_answer_relevance(answer, query)
        elif evaluator == "ROUGE-2":
            measure = ROUGE(answer, query)
        # Convert measure to float and append to list
        scores[i] = round(float(measure), 3)
    return scores


def llm_binary_answer_relevance(answer, query):
    messages = [
        {
            "role": "user",
            "content": (
                "Given the following query and answer,"
                " Give a rating from 1 to 5."
                " Respond with 1 if the answer is not relevant to the query at all."
                " Respond with 2 if the answer is slightly relevant to the query."
                " Respond with 3 if the answer is moderately relevant to the query."
                " Respond with 4 if the answer is mostly relevant to the query."
                " Respond with 5 if the answer is completely relevant to the query."
                ' Your response must strictly and only be a single integer from "1" to "5" and no additional text.'
                " Some Examples:"
                " If none of the nouns in the answer are present in the query, the answer is not relevant."
                ' If the answer is "I do not know" or "there is no mention of {...} in the given context", your response should be "1".'
                ' If the query is "What color is the sky?" and the answer is "The sky is blue", your response should be "5".'
                ' If the query is "What color is the grass?" and the answer is "The sky is blue", your response should be "1".'
                ' If the query is "At what temperature does water boil?" and the answer is "Water boils at 100 degrees Celsius", your response should be "5".'
                ' If the query is "At what temperature does water freeze?" and the answer is "Water boils at 100 degrees Celsius", your response should be "1".'
                ' If the query is "What year did the corona pandemic start?" and the answer is "The pandemic, spanning the whole globe started in 2019.", your response should be "5".'
                ' If the query is "What year did the corona pandemic start?" and the answer is "The pandemic, was a global event and lead to many deaths.", your response should be "1".'
                ' If the query is "What is the capital of France?" and the answer is "Paris is the capital of France.", your response should be "5".'
                ' If the query is "What is the capital of France?" and the answer is "France is a country in Europe.", your response should be "1".'
                ' If the query is "What is the capital of Germany?" and the answer is "Berlin is a major city in Germany.", your response should be "4".'
                ' If the query is "What is the capital of Germany?" and the answer is "Berlin is the capital of Germany.", your response should be "5".'
                ' If the query is "What is the capital of Germany?" and the answer is "Germany is a country in Europe.", your response should be "1".'
                ' If the query is "What is the capital of Germany?" and the answer is "Berlin is a city in Germany.", your response should be "4".'
                ' If the query is "What is the capital of Germany?" and the answer is "Berlin is a large city.", your response should be "3".'
                ' If the query is "What is the capital of Germany?" and the answer is "Berlin is known for its history.", your response should be "2".'
                f'Here are the Query: "{query}" and the Answer: "{answer}".'
            ),
        },
    ]
    result = evalSendToLLM(messages)
    return result


def evaluate_correctness(
    answers, ground_truths, evaluator="sem_similarity", scorer=None
):
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
        if evaluator == "sem_similarity":
            measure = semantic_similarity(answer, ground_truth, scorer=scorer)
        elif evaluator == "llm_judge":
            measure = llm_binary_correctness(answer, ground_truth)
        elif evaluator == "ROUGE-2":
            measure = ROUGE(answer, ground_truth)

        scores[i] = round(float(measure), 3)
    # print(f" Scores: {scores}")
    return scores


def llm_binary_correctness(answer, ground_truth):
    messages = [
        {
            "role": "system",
            "content": (
                "Given the following answer and ground-truth,"
                " Give a rating from 1 to 5."
                " Respond with 1 if the answer is not correct based on the ground-truth at all."
                " Respond with 2 if the answer is slightly correct based on the ground-truth."
                " Respond with 3 if the answer is moderately correct based on the ground-truth."
                " Respond with 4 if the answer is mostly correct based on the ground-truth."
                " Respond with 5 if the answer is completely correct based on the ground-truth."
                ' Your response must strictly and only be a single integer from "1" to "5" and no additional text.'
                " Some Examples:"
                ' If none of the nouns in the answer are present in the ground-truth, the answer is not correct. Thus your response should be "1".'
                ' If the answer is "yes" and the ground-truth is "no", your response should be "1".'
                ' If the answer is "no" and the ground-truth is "yes", your response should be "1".'
                ' If the answer is "I do not know" and the ground-truth is "yes", your response should be "1".'
                ' If the answer is "there is no mention of {...} in the given context" and the ground-truth is "yes" or "no", your response should be "1".'
                ' If the answer is "yes, the shirt is dark blue" and the ground-truth is "yes", your response should be "5".'
                ' If the answer is "yes" and the ground-truth is "yes", your response should be "5".'
                ' If the answer is "The sky is blue" and the ground-truth is "The sky is blue", your response should be "5".'
                ' If the answer is "The sky is blue" and the ground-truth is "The sky is clear", your response should be "4".'
                ' If the answer is "The sky is clear and blue" and the ground-truth is "The sky is blue", your response should be "4".'
                ' If the answer is "The sky is somewhat blue" and the ground-truth is "The sky is blue", your response should be "3".'
                ' If the answer is "The sky is blue with some clouds" and the ground-truth is "The sky is blue", your response should be "3".'
                ' If the answer is "The sky is light blue" and the ground-truth is "The sky is blue", your response should be "2".'
                ' If the answer is "The sky is blueish" and the ground-truth is "The sky is blue", your response should be "2".'
                f'Here are the Answer: "{answer}" and the ground-truth: "{ground_truth}".'
            ),
        },
    ]
    result = evalSendToLLM(messages)
    # print(f"A: {answer}")
    # print(f"GT: {ground_truth}")
    # print(f"Result: {result}")
    # print("----------")

    return result


# Function to calculate BERTScore semantic similarity
def semantic_similarity(candidate, reference, scorer=None):
    # Example texts
    # BERTScore calculation
    P, R, F1 = scorer.score([candidate.lower()], [reference.lower()])
    # print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    return R.mean()


# Function to calculate ROUGE-N (ROUGE-1, ROUGE-2)
def ROUGE(candidate, reference, n=1):
    # Calcuate ROGUE Score between candidate and reference
    # Helper function to tokenize sentences into words
    def tokenize(sentence):
        return re.findall(r"\w+", sentence.lower())

    reference = tokenize(reference)
    candidate = tokenize(candidate)
    ref_ngrams = [" ".join(reference[i : i + n]) for i in range(len(reference) - n + 1)]
    cand_ngrams = [
        " ".join(candidate[i : i + n]) for i in range(len(candidate) - n + 1)
    ]

    ref_count = Counter(ref_ngrams)
    cand_count = Counter(cand_ngrams)

    overlap = sum((cand_count & ref_count).values())
    total_ref = sum(ref_count.values())
    # total_cand = sum(cand_count.values())

    recall = overlap / total_ref if total_ref > 0 else 0
    # precision = overlap / total_cand if total_cand > 0 else 0
    # f1_score = (
    #    2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    # )

    return recall


def evalSendToLLM(
    messages,
    LLM_NAME=LLM_NAME,
    LLM_URL=LLM_URL,
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
        "model": LLM_NAME,
        "messages": messages,
        "temperature": model_temp,
        "max_tokens": answer_size,
        "presence_penalty": presence_pen,
        "repeat_penalty": repeat_pen,
    }
    endpoint = LLM_URL + "/chat/completions"
    # print("Sending query to OpenAI endpoint: " + endpoint)
    # Safe guard
    response = requests.post(endpoint, headers=headers, json=data)
    if not response.status_code == 200:
        print(f"LLM failed with status code: {response.status_code}")
        raise Exception("LLM failed")
        return
    # print("Received response...")
    # if reponse is 200, extract the generated response as json object
    report = response.json()
    if "choices" in report:
        if len(report["choices"]) > 0:  # Always take the first choice.
            result = report["choices"][0]["message"]["content"]
        else:
            result = "No result generated!"
    else:
        result = report
    return result


def eval_context_goldPassages(rag_elements, goldPassages):
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
