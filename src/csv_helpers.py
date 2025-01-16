# Writee results of RAG pipeline evaluation to CSV file
import csv
import os

import pandas as pd


def get_csv_files_from_dir(dir: str) -> list[str]:
    """
    Retrieves a list of CSV files from the specified directory.

    Args:
        dir (str): The path to the directory to search for CSV files.

    Returns:
        list[str]: A list of filenames (with .csv extension) found in the directory, sorted alphabetically.
    """
    csv_files = []
    for file in sorted(os.listdir(dir)):
        if file.endswith(".csv"):
            csv_files.append(file)
            # print(file)  # Print for debugging
    return csv_files


def write_context_relevance_to_csv(filename: str, scores: dict, evaluator: str) -> None:
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


def write_faithfulness_to_csv(filename: str, scores: dict, evaluator: str) -> None:
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


def write_answer_relevance_to_csv(filename: str, scores: dict, evaluator: str) -> None:
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


def write_correctness_to_csv(filename: str, scores: dict, evaluator: str) -> None:
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
        header = ["Answer", "Answer_Score"]
        writer.writerow(header)

        # Write the data rows
        for answer, llm_answer_score in scores.items():
            row = [answer, llm_answer_score]  # Convert the numpy array to a list
            writer.writerow(row)

    print(f"Data written to {filename} successfully.")


def write_pipe_results_to_csv(data: list[dict], filename: str) -> None:
    """
    Writes a list of dictionaries to a CSV file with specified column names.

    Args:
        data (list[dict]): A list of dictionaries containing the data to be written to the CSV file.
            Each dictionary should have the following keys:
                - "question" (str): The question text.
                - "answer" (str): The answer text.
                - "contexts" (list[str]): A list of context strings.
                - "contexts_ids" (list[int]): A list of context IDs.
                - "ground_truth" (str): The ground truth answer.
                - "goldPassages" (list[str]): A list of gold passage strings (optional).

        filename (str): The name of the CSV file to write the data to.

    Returns:
        None

    Example:
        data = [
            {
                "question": "What is AI?",
                "answer": "Artificial Intelligence",
                "contexts": ["Context 1", "Context 2"],
                "contexts_ids": [1, 2],
                "ground_truth": "Artificial Intelligence",
                "goldPassages": [42, 43]
            }
        write_pipe_results_to_csv(data, "results.csv")
    """
    print(f"Writing results to csv file: {filename}")
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
            # If gold passages are given: otherwise, leave as empty string
            if entry["goldPassages"]:
                entry["goldPassages"] = ", ".join(map(str, entry["goldPassages"]))

            writer.writerow(entry)

    print(f"Data has been written to {filename}")


def read_pipe_results_from_csv(filename: str) -> list[dict]:
    """
    Reads pipe results from a CSV file and converts them into a list of dictionaries.
    Each dictionary represents the results for a single query.

    Args:
        filename (str): The name of the CSV file to read the data from.

    Returns:
        list[dict]: A list of dictionaries containing the data read from the CSV file.
            Each dictionary will have the following keys:
                - "question" (str): The question text.
                - "answer" (str): The answer text.
                - "contexts" (list[str]): A list of context strings.
                - "contexts_ids" (list[int]): A list of context IDs.
                - "ground_truth" (str): The ground truth answer.
                - "goldPassages" (list[int]): A list of gold passage IDs.
    """
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


def write_eval_results_to_csv(
    eval_results: dict,
    eval_results_dir: str,
    pipe_results_file: str,
    select: str,
    evaluator: str,
    slice_for_dev: int = None,
    write_context: bool = False,
) -> None:
    """
    Write evaluation results to a CSV file by adding correctness and other metrics to the existing CSV file.

    Args:
        eval_results (dict): A dictionary containing evaluation results with keys such as 'context_relevance',
                             'faithfulness', 'answer_relevance', and 'correctness'.
        eval_results_dir (str): The directory where the evaluation results CSV file will be saved.
        pipe_results_file (str): The path to the existing CSV file containing pipeline results.
        select (str): A string used to differentiate between different evaluation runs.
        evaluator (str): The name of the evaluator used for the evaluation.
        slice_for_dev (Optional[int], optional): An optional parameter to slice the data for development purposes. Defaults to None.
        write_context (bool, optional): A flag indicating whether to include the 'contexts' column in the output CSV file. Defaults to False.

    Returns:
        None: This function does not return any value. It writes the updated DataFrame to a CSV file.
    """

    # Get the pipe results file name
    print(f"Pipe results file: {pipe_results_file}")
    pipe_results_file_name = os.path.basename(pipe_results_file).split(".csv")[0]
    print(f"Pipe results file name: {pipe_results_file_name}")
    # Define the file paths
    eval_results_file = (
        f"{eval_results_dir}/{pipe_results_file_name}{select}_{evaluator}.csv"
    )
    # Print for debugging
    # print(f"Eval results file name: {eval_results_file}")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(pipe_results_file)

    # Add a new column with default values None
    df["Correct"] = None
    df["CR"] = None
    df["Faithfulness"] = None
    df["AR"] = None

    # Skip the contexts column for a slim csv file
    if write_context is False:
        # Drop the contexts column
        df = df.drop(columns=["contexts"])

    # Get the results for each evaluation metric
    cr_results = eval_results.get("context_relevance", None)
    faithfulness_results = eval_results.get("faithfulness", None)
    ar_results = eval_results.get("answer_relevance", None)
    correct_results = eval_results.get("correctness", None)

    # Check if the results are not None, if they are, set them to a default value
    if cr_results is None:
        cr_results = [None] * len(df)
    if faithfulness_results is None:
        faithfulness_results = [None] * len(df)
    if ar_results is None:
        ar_results = [None] * len(df)
    if correct_results is None:
        correct_results = [None] * len(df)

    # Update the DataFrame with the evaluation results

    df.loc[: len(correct_results) - 1, "Correct"] = correct_results
    df.loc[: len(ar_results) - 1, "AR"] = ar_results

    print(f" len of cr results: {len(cr_results)}")
    print(f" len of faithfulness results: {len(faithfulness_results)}")

    # Pad the cr_results to the length of df
    # Pad each sublist in cr_results to ensure they all have the same length as the DataFrame
    for i, (rowCR, rowFaith) in enumerate(zip(cr_results, faithfulness_results)):
        if rowCR is not None:
            df.loc[i, "CR"] = str(rowCR)
        if rowCR is None:
            df.loc[i, "CR"] = rowCR
        if rowFaith is not None:
            df.loc[i, "Faithfulness"] = str(rowFaith)
        if rowFaith is None:
            df.loc[i, "Faithfulness"] = rowFaith

    # Save the updated DataFrame back to a CSV file
    df.to_csv(eval_results_file, index=False, quoting=csv.QUOTE_ALL)

    print(f"Results written to {eval_results_file} successfully.")
