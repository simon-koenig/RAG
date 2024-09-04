# Imports
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Constants
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"


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

    scores = {}
    ##
    ## Evaluate context relevance without pipeline run beforehand
    ## In this case Please provide a vectorDB object to retrieve documents from index.
    ##
    ## If no contexts are provided, retrieve top 3 documents from index based on query
    ## for each document.

    if contexts is None:  # Extend context to list of nones to match queries length
        contexts = [None] * len(queries)

    # Loop over all queries with their respective contexts
    for query, context in zip(queries, contexts):
        if context is None:
            # Retrieve top 3 documents from index based on query
            context, ids = vectorDB.retrieveDocuments(query, 3)

        measurements = []
        # Loop over all contexts for a query
        for single_context in context:
            print(f"Context: {single_context}")

            # Evaluate context relevance based on chosen evaluator
            if evaluator == "sem_similarity":
                measure = semantic_similarity(single_context, query)
            elif evaluator == "llm_judge":
                measure = llm_binary_context_relevance(single_context, query)
            # Convert measure to float and append to list
            measurements.append(round(float(measure), 3))
        # Compute mean context relevance over all contexts per query
        scores[query] = np.array(measurements)

    return scores


def llm_binary_context_relevance(context, query):
    messages = [
        {
            "role": "system",
            "content": "Given the following context and query,"
            " Give a binary rating, either 0 or 1."
            " Respond with 0 if an answer to the query cannot be derived from the given context. "
            "Respond with 1 if an answer to the query can be derived from the given context.  "
            'Strictly respond with  either  "0" or "1"'
            'The output must strictly and only be a single integer "0" or "1" and no additional text.',
        },
        {"role": "user", "content": f"Context: {context} ; Query: {query}"},
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
    scores = {}
    for answer, context in zip(answers, contexts):
        measurements = []
        for single_context in context:
            # Insert here evaluation measure of retrieved context
            print(f"Context: {single_context}")  #
            if evaluator == "sem_similarity":
                measure = semantic_similarity(single_context, answer)
            elif evaluator == "llm_judge":
                measure = llm_binary_faithfulness(single_context, answer)

            measurements.append(round(float(measure), 3))
        # Compute mean faithfulness over all contexts per answer
        scores[answer] = np.array(measurements)  # Insert evaluation measure here

    return scores


def llm_binary_faithfulness(context, answer):
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

    scores = {}
    for answer, query in zip(answers, queries):
        print(f"Answer: {answer}")
        print(f"Query: {query}")
        # Evaluate context relevance based on chosen evaluator
        if evaluator == "sem_similarity":
            measure = semantic_similarity(answer, query)
        elif evaluator == "llm_judge":
            measure = llm_binary_answer_relevance(answer, query)
        # Convert measure to float and append to list
        scores[query] = round(float(measure), 3)
    return scores


def llm_binary_answer_relevance(answer, query):
    messages = [
        {
            "role": "system",
            "content": "Given the following query and answer,"
            "Analyse the question and answer without consulting prior knowledge."
            " Determine if the answer is relevant to the question."
            " Give a binary rating, either 0 or 1."
            " Consider whether the answer addresses all parts of question asked."
            " Respond with 0 if the answer is relevant to the question"
            " Respond with 1 if the answer in not relevant to the question"
            ' Strictly respond with  either  "0" or "1"'
            'The output must strictly and only be a single integer "0" or "1" and no additional text.',
        },
        {"role": "user", "content": f"Query: {query} ; Answer: {answer}"},
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

    scores = {}
    for answer, ground_truth in zip(answers, ground_truths):
        print(f"Answer: {answer}")
        print(f"Ground truth: {ground_truth}")
        if evaluator == "sem_similarity":
            measure = semantic_similarity(answer, ground_truth)
        elif evaluator == "llm_judge":
            measure = llm_binary_correctness(answer, ground_truth)

        scores[answer] = round(float(measure), 3)
    return scores


def llm_binary_correctness(answer, ground_truth):
    messages = [
        {
            "role": "system",
            "content": "Given the following answer and ground truth,"
            "Analyse the answer and ground truth without consulting prior knowledge."
            " Determine if the answer matches the ground truth in meaning."
            " Give a binary rating, either 0 or 1."
            " Respond with 1 if the answer matches the ground truth in meaning."
            " Respond with 0 if the answer does not match the ground truth in meaning."
            ' Strictly respond with  either  "0" or "1"'
            'The output must strictly and only be a single integer "0" or "1" and no additional text.',
        },
        {
            "role": "user",
            "content": f"Answer: {answer} ; ground truth: {ground_truth}",
        },
    ]
    result = evalSendToLLM(messages)
    return result


def semantic_similarity(sentence1, sentence2):
    model = SentenceTransformer("all-mpnet-base-v2")
    # multi-qa-MiniLM-L6-cos-v1, cheap model for dev
    # all-mpnet-base-v2 , more performant model, but slower
    sentence1_vec = model.encode([sentence1])

    sentence2_vec = model.encode([sentence2])

    similarity_score = model.similarity(
        sentence1_vec, sentence2_vec
    )  # Default is cosine simi
    print(f"\n Similarity Score = {similarity_score} ")

    return similarity_score


def eval(rag_elements, method=None, given_queries=None, evaluator="sem_similarity"):
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
        contexts_ids = [element["context_ids"] for element in rag_elements]

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
    contexts = [element["contexts"] for element in rag_elements]
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
