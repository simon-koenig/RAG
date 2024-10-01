# Imports
import sys
import threading
import time
from functools import partial
from pprint import pprint

import numpy as np

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
    parameters,
    queries,
    ground_truths,
    goldPassages,
    documentDB,
    LLM_URL,
    LLM_NAME,
    n_slice_rag_elements,
    write_to_dir,
):
    # Unpack parameter permuation
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
        search_ref_lex=4,
        search_ref_sem=4,
        num_ref_lim=num_ref_lim_val,
        model_temp=0.0,
        answer_token_num=50,
    )

    # Run pipeline
    # With slice of rag elements for dev
    print(f"goldPassages: {goldPassages}")
    pipe.run(
        questions=queries[:n_slice_rag_elements],
        ground_truths=ground_truths,
        goldPassagesIds=goldPassages,
    )

    print("Pipeline run completed.")
    # Print results
    # for elem in pipe.rag_elements:
    #    pprint(elem)

    # Write results to csv file
    # Build the csv file path for the current parameter setting
    csv_file_path = write_to_dir
    csv_file_path += f"quExp{query_expansion_val}_"
    csv_file_path += f"rerank{rerank_val}_"
    csv_file_path += f"cExp{prepost_context_val}_"
    csv_file_path += f"backRev{background_reversed_val}_"
    csv_file_path += f"numRefLim{num_ref_lim_val}_"
    csv_file_path += ".csv"

    write_pipe_results_to_csv(pipe.rag_elements, csv_file_path)


def eval_single_pipe_result(
    pipe_results_file_name,
    pipe_results_dir,
    eval_results_dir,
    method,
    evaluator,
    slice_for_dev,
):
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
        method=method,
        evaluator=evaluator,
    )
    # pprint(eval_results)

    # Write results of a single pipe run to a csv file
    write_eval_results_to_csv(
        eval_results=eval_results,
        eval_results_dir=eval_results_dir,
        pipe_results_file=pipe_results_file,
        method=method,
        evaluator=evaluator,
        slice_for_dev=slice_for_dev,
    )

    # Done message
    print(f"Done! Eval results written to {eval_results_dir}/{pipe_results_file_name}.")
