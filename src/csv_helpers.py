# Writee results of RAG pipeline evaluation to CSV file
import csv


def write_context_relevance_to_csv(filename, scores, evaluator):
    """
    Writes context relevance scores to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        scores (dict): A dictionary of query-context relevance scores.
        evaluator (str): The name of the evaluator.

    Returns:
        None
    """
    # Open the CSV file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the title row
        title = f"Query-Context-Relevance-{evaluator}"
        writer.writerow([title])

        # Assuming all contexts arrays have the same length
        num_contexts = len(next(iter(scores.values())))
        # Write the header row
        header = ["Query"] + [f"Context_{i}_Score" for i in range(num_contexts)]
        writer.writerow(header)

        # Write the data rows
        for query, contexts_scores in scores.items():
            row = [
                query
            ] + contexts_scores.tolist()  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


def write_faithfulness_to_csv(filename, scores, evaluator):
    """
    Writes faithfulness scores to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        scores (dict): A dictionary of answer-context faithfulness scores.
        evaluator (str): The name of the evaluator.

    Returns:
        None
    """
    # Open the CSV file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the title row
        title = f"Answer-Context-Faithfulness-{evaluator}"
        writer.writerow([title])

        # Assuming all contexts arrays have the same length
        num_contexts = len(next(iter(scores.values())))
        # Write the header row
        header = ["Answer"] + [f"Context_{i}_Score" for i in range(num_contexts)]
        writer.writerow(header)

        # Write the data rows
        for llm_answer, contexts_scores in scores.items():
            row = [
                llm_answer
            ] + contexts_scores.tolist()  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


def write_answer_relevance_to_csv(filename, scores, evaluator):
    """
    Writes answer relevance scores to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        scores (dict): A dictionary of query-answer relevance scores.
        evaluator (str): The name of the evaluator.

    Returns:
        None
    """
    # Open the CSV file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the title row
        title = f"Query-Answer-Relevance-{evaluator}"
        writer.writerow([title])

        # Write the header row
        header = ["Query", "Answer_Score"]
        writer.writerow(header)

        # Write the data rows
        for query, llm_answer_score in scores.items():
            row = [query, llm_answer_score]  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


def write_correctness_to_csv(filename, scores, evaluator):
    """
    Writes correctness scores to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        scores (dict): A dictionary of ground-truth answer correctness scores.
        evaluator (str): The name of the evaluator.

    Returns:
        None
    """
    # Open the CSV file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the title row
        title = f"Ground-Truth-Answer-Correctness-{evaluator}"
        writer.writerow([title])

        # Write the header row
        header = ["Ground_Truth", "Answer_Score"]
        writer.writerow(header)

        # Write the data rows
        for answer, llm_answer_score in scores.items():
            row = [answer, llm_answer_score]  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


def write_pipe_results_to_csv(data, filename):
    # Define the column names
    fieldnames = [
        "question",
        "answer",
        "contexts",
        "contexts_ids",
        "ground_truth",
        "goldPassages",
    ]

    # Write to CSV file
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write rows
        for entry in data:
            # Convert lists to strings for CSV format
            entry["contexts"] = ", ".join(entry["contexts"])
            entry["contexts_ids"] = ", ".join(map(str, entry["contexts_ids"]))
            entry["goldPassages"] = ", ".join(map(str, entry["goldPassages"]))

            writer.writerow(entry)

    print(f"Data has been written to {filename}")


# Function to get from rag results csv to pipe.rag_elements data structure
def read_pipe_results_from_csv(filename):
    # Initialize the list to store the data
    data = []

    # Open the CSV file
    with open(filename, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        # Read each row in the CSV file
        for row in reader:
            # Convert strings back to lists
            if row["contexts"]:
                row["contexts"] = [
                    context.strip()
                    for context in row["contexts"].split(" Context: ")
                    if context
                ]
            else:
                row["contexts"] = []

            row["contexts_ids"] = (
                list(map(int, row["contexts_ids"].split(", ")))
                if row["contexts_ids"]
                else []
            )
            row["goldPassages"] = (
                list(map(int, row["goldPassages"].split(", ")))
                if row["goldPassages"]
                else []
            )

            # Append the row (as a dictionary) to the data list
            data.append(row)

    print(f"Data has been read from {filename}")
    return data
