# Test of rag evaluation
# Imports
import sys

sys.path.append("./dev/")
sys.path.append("./src/")
sys.path.append("./")

from config import LLM_URL, MARQO_URL, MARQO_URL_GPU
from csv_helpers import write_pipe_results_to_csv
from dataset_helpers import DatasetHelpers
from pipe import RagPipe
from vector_store import VectorStore

# Define LLM Name if necessary
LLM_NAME = "llama3.1:latest"

# Load QM queries
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = (
    datasetHelpers.loadMiniBiosqa()
)  # Mini Bios

# Load the VectorStore
documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex("minibios-qa-gpu")  # Connect to the minibio
stats = documentDB.getIndexStats()
print(stats)

# Load the RagPipe
pipe = RagPipe()
pipe.connectVectorStore(documentDB)
pipe.connectLLM(LLM_URL, LLM_NAME)


##
## Set parameters for the pipeline
##


pipe.setConfigs(
    lang="EN",
    query_expansion=3,
    rerank=False,
    prepost_context=False,
    background_reversed=False,
    search_ref_lex=2,
    search_ref_sem=2,
    num_ref_lim=4,
    model_temp=0.1,
    answer_token_num=50,
)


# Run pipeline
pipe.run(
    questions=queries[:1],
    ground_truths=ground_truths[:1],
    goldPassagesIds=goldPassages[:1],
)

# Print results
# for elem in pipe.rag_elements:
#    pprint(elem)


write_pipe_results_to_csv(pipe.rag_elements, "./rag_results/query_expansion_test.csv")
print("Results written to csv file.")
