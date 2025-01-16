# Imports
import sys

sys.path.append("./dev/")
sys.path.append("./src/")


from csv_helpers import (
    read_pipe_results_from_csv,
    write_eval_results_to_csv,
    write_pipe_results_to_csv,
)
from evaluate import evaluate
from pipe import RagPipe


def pipe_single_setting_run(
    parameters: tuple,
    queries: list,
    ground_truths: list,
    goldPassages: list,
    documentDB: str,
    LLM_URL: str,
    LLM_NAME: str,
    n_sample_queries: int,
    write_to_dir: str,
) -> None:
    """
    Run a single setting of the pipeline with the given parameters and data.

    Args:
        parameters (tuple): A tuple containing the pipeline configuration parameters.
        queries (list): A list of queries to be processed by the pipeline.
        ground_truths (list): A list of ground truth answers corresponding to the queries.
        goldPassages (list): A list of gold passage IDs corresponding to the queries.
        documentDB (str): The path to the document database.
        LLM_URL (str): The URL of the language model.
        LLM_NAME (str): The name of the language model.
        n_sample_queries (int): The number of sample queries to be processed.
        write_to_dir (str): The directory where the results will be written.

    Returns:
        None
    """
    # Unpack parameter permutation
    (
        query_expansion_val,
        rerank_val,
        prepost_context_val,
        background_reversed_val,
        num_ref_lim_val,
    ) = parameters

    # Load the RagPipe
    pipe = RagPipe()
    pipe.connectVectorStore(documentDB)
    pipe.connectLLM(LLM_URL, LLM_NAME)

    # Set pipeline configurations
    pipe.setConfigs(
        lang="EN",
        query_expansion=query_expansion_val,
        rerank=rerank_val,
        prepost_context=prepost_context_val,
        background_reversed=background_reversed_val,
        search_ref_lex=8,
        search_ref_sem=8,
        num_ref_lim=num_ref_lim_val,
        model_temp=0.0,
        answer_token_num=50,
    )

    # Run pipeline
    # With slice of rag elements for dev. Slice is designed to take equally distributed queries up to n_sample_queries
    n_queries = len(queries)
    k = n_queries // n_sample_queries
    queries = queries[::k][:n_sample_queries]
    ground_truths = ground_truths[::k][:n_sample_queries]
    if goldPassages is not None:
        goldPassages = goldPassages[::k][:n_sample_queries]

    pipe.run(
        questions=queries,
        ground_truths=ground_truths,
        goldPassagesIds=goldPassages,
        nThreads=1,
    )

    print("Pipeline run completed.")

    ##  Filename determines:  parameter setting.

    csv_file_path = write_to_dir
    csv_file_path += f"quExp{query_expansion_val}_"
    csv_file_path += f"rerank{rerank_val}_"
    csv_file_path += f"cExp{prepost_context_val}_"
    csv_file_path += f"backRev{background_reversed_val}_"
    csv_file_path += f"numRefLim{num_ref_lim_val}_"
    csv_file_path += f"{LLM_NAME}_"
    csv_file_path += ".csv"

    write_pipe_results_to_csv(pipe.rag_elements, csv_file_path)


def eval_single_pipe_result(
    pipe_results_file_name: str,
    pipe_results_dir: str,
    eval_results_dir: str,
    select: str,
    evaluator: str,
    slice_for_dev: int,
) -> None:
    """
    Evaluate the results of a single pipeline run and write the evaluation results to a CSV file.

    Args:
        pipe_results_file_name (str): The name of the file containing the pipeline results.
        pipe_results_dir (str): The directory where the pipeline results file is located.
        eval_results_dir (str): The directory where the evaluation results should be written.
        select (str): The selection criteria for evaluation.
        evaluator (str): The evaluator to be used for evaluation.
        slice_for_dev (int): The number of elements to slice from the pipeline results for development evaluation.

    Returns:
        None
    """

    # Only look at files with quExp1_rerank1_cExpFalse_backRevFalse_numRef
    pipe_results_file = f"{pipe_results_dir}/{pipe_results_file_name}"
    # print(pipe_results_file)

    # Read pipe results from CSV
    pipe_results = read_pipe_results_from_csv(filename=pipe_results_file)

    # Test print results
    # for elem in pipe_results:
    #    pprint(elem)

    # Evaluate pipe results
    eval_results = evaluate(
        rag_elements=pipe_results[:slice_for_dev],
        select=select,
        evaluator=evaluator,
    )
    # pprint(eval_results)

    # Write results of a single pipe run to a csv file
    write_eval_results_to_csv(
        eval_results=eval_results,
        eval_results_dir=eval_results_dir,
        pipe_results_file=pipe_results_file,
        select=select,
        evaluator=evaluator,
        slice_for_dev=slice_for_dev,
    )

    # Done message
    print(f"Done! Eval results written to {eval_results_dir}/{pipe_results_file_name}.")
