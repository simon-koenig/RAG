# Usage example of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("../dev/")
from components import DatasetHelpers, RagPipe, VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"

# Run eval in a few lines

# Load the dataset
datasetHelpers = DatasetHelpers()
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
    queries, ground_truths, corpus_list, newIngest=False, maxDocs=1000, maxQueries=3
)


# Evaluate the rag pipeline, methods = "context_relevance", "answer_relevance", "faithfulness", "all"
scores = pipe.eval(method="all")
pprint(scores)
