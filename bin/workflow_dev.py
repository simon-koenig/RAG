# Usage example of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("./dev/")
from components import DatasetHelpers, RagPipe, VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"

# Run eval in a few lines
parameters = {
    "chunking_params": {
        "chunk_size": 512,  # Number of charakters in a chunk
        "chunk_overlap": 128,  # Number of overlapping charakters in a chunk
        "chunk_method": "recursive",  # Method of chunking
    },
    "index_settings": {
        "split_method": "sentence",  # Method of splitting
        "split_overlap": 0,  # Number of overlapping tokens in a split
        "distance_metric": "prenormalized-angular",  # Distance metric for ann
        "model": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",  # Model for vector embedding
        "pre_post_context_size": 100,  # Number of tokens to add to context
    },
    "retrieval_settings": {
        "k_docs_to_retrieve ": 5,  # Number of documents to retrieve
        "model_temp": 0.5,  # Model temperature
        "answer_size": 100,  # Number of tokens in answer
        "query_threshold": 0.5,  # Query threshold
    },
}


# Load the dataset
datasetHelpers = DatasetHelpers(parameters["chunking_params"])
corpus_list, queries, ground_truths = datasetHelpers.loadMiniWiki()

# Load the VectorStore
documentDB = VectorStore(MARQO_URL)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex("miniwikiindex")  # Connect to the miniwikiindex

# Or create a new Index
# documentDB.createIndex("miniwikiindex",settings=None)  # Create a new index

# Load the RagPipe
pipe = RagPipe()
pipe.connectVectorStore(documentDB)
pipe.connectLLM(LLM_URL, LLM_NAME)

# Run the rag pipeline and ingest
pipe.run(
    queries,
    ground_truths,
    corpus_list,
    newIngest=False,
    maxDocs=1000,
    maxQueries=2,
    rerank=True,
    prepost_context=True,
)


# Evaluate the rag pipeline, methods = "context_relevance", "answer_relevance", "faithfulness", "correctness", "all"
# evaluator = "sem-similarity", "llm-judge"
# scores = pipe.eval(method="correctness", evaluator="llm-judge")
# print(scores)
