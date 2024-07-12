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

# Run eval in a few lines
parameters = {
    "chunking_params": {
        "chunk_size": 512,  # Number of charakters in a chunk
        "chunk_overlap": 128,  # Number of overlapping charakters in a chunk
        "chunk_method": "recursive",  # Method of chunking
    },
    "index_settings": {
        "split_length": 2,  # Number of elmenents in a split
        "split_method": "sentence",  # Method of splitting
        "split_overlap": 0,  # Number of overlapping tokens in a split
        "distance_metric": "prenormalized-angular",  # Distance metric for ann
        "model": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",  # Model for vector embedding
    },
    "retrieval_settings": {
        "k_docs_to_retrieve ": 5,  # Number of documents to retrieve
        "query_threshold": 0.5,  # Query threshold
        "pre_post_context_size": 100,  # Number of tokens to add to context
        "rerank": True,  # Rerank documents
        "prepost_context": True,  # Add pre and post context
    },
    "llm_settings": {
        "llm_name": "llama3",  # Name of the llm
        "model_temp": 0.5,  # Model temperature
        "answer_size": 100,  # Number of tokens in answer
    },
}

# Load the dataset
datasetHelpers = DatasetHelpers(parameters["chunking_params"])
corpus_list, queries, ground_truths = datasetHelpers.loadQM()  # Mini wiki

# Load the VectorStore
documentDB = VectorStore(MARQO_URL)  # Connect to marqo client via python API
# View indexes
print(documentDB.getIndexes())
# Connect to the miniwikiindex
documentDB.connectIndex("ait-qm")
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
scores = pipe.eval(method="context_relevance", evaluator="sem_similarity")
print(scores)

# Write the scores to a csv file
write_context_relevance_to_csv(
    "example_file_path.csv", scores, evaluator="sem_similarity"
)
