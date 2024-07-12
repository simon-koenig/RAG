# Test of rag evaluation
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


# Load QM queries
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = datasetHelpers.loadMiniBioasq()


# Load the VectorStore
documentDB = VectorStore(MARQO_URL)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
# documentDB.connectIndex("minibios-qa")  # Connect to the miniwikiindex / ait-qm / minibios-qa
index_settings = {
    "split_length": 2,  # Number of elmenents in a split
    "split_method": "sentence",  # Method of splitting
    "split_overlap": 0,  # Number of overlapping tokens in a split
    "distance_metric": "prenormalized-angular",  # Distance metric for ann
    "model": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",  # Model for vector embedding
}

# Connect to the miniwikiindex / ait-qm / minibios-qa
documentDB.connectIndex("minibios-qa")
# Load the RagPipe
pipe = RagPipe()
pipe.connectVectorStore(documentDB)
pipe.connectLLM(LLM_URL, LLM_NAME)

import time

maxDocs = 5
# start = time.time()
# documentDB.indexDocuments(corpus_list, maxDocs)  # Add documents to the index
# end = time.time()
# print(f"Time for indexing 40.200 documents: {end - start} seconds")


# Run rag pipeline for test quieries of qm dataset
stats = documentDB.getIndexStats()
print(stats)
nQueries = 100
start = time.time()
pipe.run(
    queries,
    ground_truths=ground_truths,
    corpus_list=corpus_list,
    newIngest=False,
    maxDocs=1000000,
    maxQueries=nQueries,
    lang="DE",
    rerank=False,
    prepost_context=False,
)

end = time.time()
print(f"Time for running {nQueries} queries through the rag pipeline: {end - start}")

for element in pipe.rag_elements:
    pprint(element)
# start = time.time()
# Evaluate the pipeline
# evaluator = "sem_similarity", "llm_judge"
scores = pipe.eval(method="context_relevance", evaluator="sem_similarity")
pprint(scores)
# Write context relevance to csv file
# Define the CSV file name

csv_file = f"Context_Relevance_ait_qm_DE.csv"
write_context_relevance_to_csv(csv_file, scores, evaluator="sem_similarity")
