# Test of rag evaluation
# Imports
import sys
from pprint import pprint

import openpyxl

sys.path.append("./dev/")
from components import DatasetHelpers, RagPipe, VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"

# Load QM queries
datasetHelpers = DatasetHelpers()
(corpus_list, queries, ground_truths) = datasetHelpers.loadMiniWiki()


# Load the VectorStore
documentDB = VectorStore(MARQO_URL)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex("miniwikiindex")  # Connect to the miniwikiindex


# Load the RagPipe
pipe = RagPipe()
pipe.connectVectorStore(documentDB)
pipe.connectLLM(LLM_URL, LLM_NAME)


# Run rag pipeline for test quieries of qm dataset
pipe.run(
    queries,
    ground_truths=ground_truths,
    corpus_list=corpus_list,
    newIngest=True,
    maxDocs=1000,
    maxQueries=2,
    lang="EN",
    rerank=True,
    prepost_context=True,
)

for rag_element in pipe.rag_elements:
    print(rag_element["question"])
    print(rag_element["answer"])
    pprint(rag_element["contexts"])
    print("\n\n")


# Evaluate the pipeline
# evaluator = "sem_similarity", "llm_judge"
scores = pipe.eval(method="context_relevance", evaluator="sem_similarity")
pprint(scores)
