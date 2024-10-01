# Imports
import numpy as np
import requests
from bert_score import BERTScorer
from csv_helpers import read_pipe_results_from_csv, write_eval_results_to_csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Constants
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3.1:latest"
scorer = BERTScorer(model_type="roberta-base")


def evaluate_context_relevance(
    queries,
    contexts=None,
    contexts_ids=None,
    goldPassages=None,
    evaluator="sem_similarity",
    vectorDB=None,
):
    # Type checking
    if not isinstance(queries, list):
        raise TypeError(
            f"Queries must be of type list, but got {type(queries).__name__}."
        )

    # Initialize scores array
    scores = np.zeros((len(queries), len(contexts[0])), dtype=float)
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
                measure = semantic_similarity(single_context, query)
            elif evaluator == "llm_judge":
                measure = llm_binary_context_relevance(single_context, query)
            # Convert measure to float and append to list
            # print(f"Measure: {measure}")
            measurements.append(round(float(measure), 3))
        # Compute mean context relevance over all contexts per query

        # measurements = np.array(measurements)
        # print(f"Measurements: {measurements}")
        scores[i] = measurements

    return scores


def llm_binary_context_relevance(context, query):
    messages = [
        {
            "role": "system",
            "content": (
                "Given the following context and query,"
                " Give a binary rating, either 0 or 1."
                " Respond with 0 if an answer to the query cannot be derived from the given context."
                " Respond with 1 if an answer to the query can be derived from the given context."
                ' Your response must strictly and only be a single integer "0" or "1" and no additional text.'
                " Some Examples:"
                ' If the context is "The sky is blue" and the query is "What color is the sky?", your response should be "1".'
                ' If the context is "The sky is blue" and the query is "What color is the grass?", your response should be "0".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "At what temperature does water boil?", your response should be "1".'
                ' If the context is "Water boils at 100 degrees Celsius" and the query is "At what temperature does water freeze?", your response should be "0".'
                ' If the context is "The pandemic, spanning the whole globe started in 2019. " and the query is "What year did the corona pandemic start?", your response should be "1".'
                ' If the context is "The pandemic, was a global event and lead to many deaths. " and the query is "What year did the corona pandemic start?", your response should be "0".'
            ),
        },
        {
            "role": "user",
            "content": f'Here are the Context: "{context}" and the Query: "{query}".',
        },
    ]

    result = evalSendToLLM(messages)
    return result


def evaluate_faithfulness(
    answers, contexts, evaluator="sem_similarity", contexts_ids=None
):
    # Type checking
    if not isinstance(answers, list):
        raise TypeError(
            f"Answers must be of type list, but got {type(answers).__name__}."
        )

    # Initialize scores array
    scores = np.zeros((len(answers), len(contexts[0])), dtype=float)
    for i, (answer, context) in tqdm(
        enumerate(zip(answers, contexts)), total=len(answers)
    ):
        measurements = []
        for single_context in context:
            # Insert here evaluation measure of retrieved context
            # print(f"Context: {single_context}")  #
            if evaluator == "sem_similarity":
                measure = semantic_similarity(single_context, answer)
            elif evaluator == "llm_judge":
                measure = llm_binary_faithfulness(single_context, answer)

            measurements.append(round(float(measure), 3))
        # Compute mean faithfulness over all contexts per answer
        scores[i] = np.array(measurements)  # Insert evaluation measure here

    return scores


def llm_binary_faithfulness(context, answer):
    messages = [
        {
            "role": "system",
            "content": (
                "Given the following context and answer,"
                " Give a binary rating, either 0 or 1."
                " Respond with 0 if the answer is not sufficiently grounded in the context."
                " Respond with 1 if the answer is sufficiently grounded in the context."
                ' Your response must strictly and only be a single integer "0" or "1" and no additional text.'
                " Some Examples:"
                ' If the context is "The sky is blue" and the answer is "The sky is blue", your response should be "1".'
                ' If the context is "The sky is blue" and the answer is "The grass is green", your response should be "0".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water boils at 100 degrees Celsius", your response should be "1".'
                ' If the context is "Water boils at 100 degrees Celsius" and the answer is "Water freezes at 0 degrees Celsius", your response should be "0".'
                ' If the context is "The pandemic, spanning the whole globe started in 2019." and the answer is "The pandemic started in 2019.", your response should be "1".'
                ' If the context is "The pandemic, was a global event and lead to many deaths." and the answer is "The pandemic started in 2019.", your response should be "0".'
            ),
        },
        {
            "role": "user",
            "content": f'Here are the Context: "{context}" and the Answer: "{answer}".',
        },
    ]

    result = evalSendToLLM(messages)
    return result


def evaluate_answer_relevance(queries, answers, evaluator="sem_similarity"):
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
            measure = semantic_similarity(answer, query)
        elif evaluator == "llm_judge":
            measure = llm_binary_answer_relevance(answer, query)
        # Convert measure to float and append to list
        scores[i] = round(float(measure), 3)
    return scores


def llm_binary_answer_relevance(answer, query):
    messages = [
        {
            "role": "system",
            "content": (
                "Given the following query and answer,"
                " Give a binary rating, either 0 or 1."
                " Respond with 0 if the answer is not relevant to the query."
                " Respond with 1 if the answer is relevant to the query."
                ' Your response must strictly and only be a single integer "0" or "1" and no additional text.'
                " Some Examples:"
                ' If the query is "What color is the sky?" and the answer is "The sky is blue", your response should be "1".'
                ' If the query is "What color is the grass?" and the answer is "The sky is blue", your response should be "0".'
                ' If the query is "At what temperature does water boil?" and the answer is "Water boils at 100 degrees Celsius", your response should be "1".'
                ' If the query is "At what temperature does water freeze?" and the answer is "Water boils at 100 degrees Celsius", your response should be "0".'
                ' If the query is "What year did the corona pandemic start?" and the answer is "The pandemic, spanning the whole globe started in 2019.", your response should be "1".'
                ' If the query is "What year did the corona pandemic start?" and the answer is "The pandemic, was a global event and lead to many deaths.", your response should be "0".'
            ),
        },
        {
            "role": "user",
            "content": f'Here are the Query: "{query}" and the Answer: "{answer}".',
        },
    ]
    result = evalSendToLLM(messages)
    return result


def evaluate_correctness(answers, ground_truths, evaluator="sem_similarity"):
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
            measure = semantic_similarity(answer, ground_truth)
        elif evaluator == "llm_judge":
            measure = llm_binary_correctness(answer, ground_truth)

        scores[i] = round(float(measure), 3)
    print(f" Scores: {scores}")
    return scores


def llm_binary_correctness(answer, ground_truth):
    messages = [
        {
            "role": "system",
            "content": (
                "Given the following answer and ground-truth,"
                "Analyse the answer and ground-truth without consulting prior knowledge."
                " Give a binary rating, either 0 or 1."
                " Respond with 1 if the answer is true based on the ground-truth."
                " Respond with 0 if the answer is incorrect based on the ground-truth."
                'Your response must strictly and only be a single integer "0" or "1" and no additional text.'
                " Some Examples:"
                ' If the answer is "yes" and the ground-truth is "no", your response should be "0".'
                ' If the answer is "yes" and the ground-truth is "yes", your response should be "1".'
                ' If the answer is "no" and the ground-truth is "yes", your response should be "0".'
                ' If the answer is "there is no mention of {...} in the given context" and the ground-truth is "yes" or "no", your response should be "0".'
                ' If the answer is "yes, the shirt is dark blue" and the ground-truth is "yes", your response should be "1".'
                ' If the answer is "I am not sure" or "I do not know" and the ground-truth is "yes" or "no", your response should be "0".'
            ),
        },
        {
            "role": "user",
            "content": f'Here are the  Answer: "{answer}" and the ground-truth: "{ground_truth}".',
        },
    ]
    result = evalSendToLLM(messages)
    print(f"A: {answer}")
    print(f"GT: {ground_truth}")
    print(f"Result: {result}")
    print("----------")

    return result


def semantic_similarity(candidate, reference, scorer=scorer):
    # Example texts
    # BERTScore calculation
    P, R, F1 = scorer.score([candidate], [reference])
    # print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    return P.mean()


def evaluate(rag_elements, method=None, given_queries=None, evaluator="sem_similarity"):
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
            scores = evaluate_context_relevance(given_queries, contexts=None)
        else:
            contexts = [element["contexts"] for element in rag_elements]
            queries = [element["question"] for element in rag_elements]
            contexts_ids = [element["contexts_ids"] for element in rag_elements]
            scores = evaluate_context_relevance(
                queries, contexts, contexts_ids=contexts_ids, evaluator=evaluator
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
        scores = evaluate_correctness(answers, ground_truths, evaluator)
        return scores

    if method == "all":
        queries = [element["question"] for element in rag_elements]
        contexts = [element["contexts"] for element in rag_elements]
        contexts_ids = [element["contexts_ids"] for element in rag_elements]

        answers = [element["answer"] for element in rag_elements]
        ground_truths = [element["ground_truth"] for element in rag_elements]

        cr_scores = evaluate_context_relevance(
            queries, contexts, evaluator=evaluator, contexts_ids=contexts_ids
        )

        f_scores = evaluate_faithfulness(
            answers, contexts, evaluator=evaluator, contexts_ids=contexts_ids
        )
        ar_scores = evaluate_answer_relevance(queries, answers, evaluator)
        c_scores = evaluate_correctness(answers, ground_truths, evaluator)

        return {
            "context_relevance": cr_scores,
            "faithfulness": f_scores,
            "answer_relevance": ar_scores,
            "correctness": c_scores,
        }


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
