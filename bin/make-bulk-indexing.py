# Test of rag evaluation
# Imports
import sys
import time
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")
sys.path.append("./")
from config import LLM_URL, MARQO_URL, MARQO_URL_GPU
from dataset_helpers import DatasetHelpers
from vector_store import VectorStore

##
## Load Dataset
##

datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, _ = datasetHelpers.loadMiniWiki()

##
## Load the VectorStore
##
documentDB_GPU = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB_GPU.getIndexes())  # Print all indexes


##
## Delete old index
##

documentDB_GPU.deleteIndex("minwiki-gpu")  # Delete the old index
print(documentDB_GPU.getIndexes())  # Print all indexes to check if the index is deleted


##
## Create new index
##

index_settings = {
    "split_length": 2,  # Number of elmenents in a split
    "split_method": "sentence",  # Method of splitting
    "split_overlap": 0,  # Number of overlapping tokens in a split
    "distance_metric": "prenormalized-angular",  # Distance metric for ann
    "model": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",  # Model for vector embedding
}

documentDB_GPU.createIndex("miniwiki-gpu", index_settings)  # Create a new index


##
## Index documents
##


maxDocs = 100000  # Number of documents to index
documentDB_GPU.connectIndex("miniwiki-gpu")  # Connect to the minibio
start = time.time()
documentDB_GPU.indexDocuments(corpus_list, maxDocs)  # Add documents to the index
end = time.time()
## Time for indexing 100 documents on marqo with cpu
print(f"Time for indexing {maxDocs} documents: {end - start} seconds")


print(documentDB_GPU.getIndexStats())  # Print index stats
