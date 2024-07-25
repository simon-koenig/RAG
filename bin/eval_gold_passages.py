# Usage example of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")
from csv_write import write_context_relevance_to_csv
from dataset_helpers import DatasetHelpers
from pipe import RagPipe
from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"


# Load the dataset
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = (
    datasetHelpers.loadMiniBiosqa()
)  # Mini wiki

# Load the VectorStore
documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
# View indexes
print(documentDB.getIndexes())
# Connect to the miniwikiindex
documentDB.connectIndex("minibios-qa-gpu")
stats = documentDB.getIndexStats()
print(stats)
# Load the RagPipe
pipe = RagPipe()
pipe.connectVectorStore(documentDB)
pipe.connectLLM(LLM_URL, LLM_NAME)

# Run the rag pipeline, with or without newIngest. Recommended to try with newIngest=False first
pipe.run(
    queries,
    ground_truths,
    corpus_list,
    newIngest=False,
    maxDocs=10,
    maxQueries=1,
    lang="EN",
    rerank=False,
    prepost_context=False,
)

# Print results
for elem in pipe.rag_elements:
    pprint(elem)

# Evaluate the rag pipeline, methods = "context_relevance", "answer_relevance", "faithfulness", "correctness", "all"
# evaluator = "sem_similarity", "llm_judge"
matches = pipe.eval_context_goldPassages(goldPassages)
print(matches)


# Write the scores to a csv file
# write_goldPassages_to_csv("example_file_path.csv", scores, evaluator="sem_similarity")
