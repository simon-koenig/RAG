# RAG - pipeline and evaluation

This repository provides a RAG ecosystem.  The goal is give the user tools to build and evaluate a costum RAG pipeline.

Tune paramters with the help of 3 evalution metrics: 
- RAG Triad (Combining Context Relevance, Answer Relevance and Faithfulness) 
- Answer correctness
- Search results recall


## Requirements
- Tested with 
Ubuntu 22.04.3
- Run the following command to install the required packages:
`pip install -r requirements.txt`

## RAG pipeline

- Index documents to vector database
- Ask query
- Retrieve documents related to query from database
- Define a prompt with instructions to an LLM on how it should process information
- Combine retrieved documents with query and send to LLM 
- Get LLM answer based on the provided context


## RAG Evaluation

Tune your system with the help of evalution metrics: 
- RAG Triad (Combining Context Relevance, Answer Relevance and Faithfulness) 
- Answer correctness
- Search results recall

Tunable parameters (among others):
- Reranking of lexical and semantic search results
- Context expansion
- Query expansion
- Number of sources
- Large language model


## Define your API Endpoints in your configuration file and copy to config.ini
```console
LLM_URL = 
MARQO_URL = 
RERANKER_ENDPOINT = 
```


## Workflow example

Imports

```python
from csv_write import write_correctness_to_csv
from dataset_helpers import DatasetHelpers
from pipe import RagPipe
from vector_store import VectorStore
from evaluate import evaluate
```

Import API ENDPOINTS, e.g.:

```python
from config import LLM_URL, MARQO_URL, MARQO_URL_GPU
```

Create vector database object:

```python
documentDB = VectorStore(MARQO_URL)
```


Connect Index
```python
documentDB.connectIndex("my-favourite-index")
```

or Create Index 
```python
documentDB.createIndex("my-first-index")
```

Add documents to index
```python
documents = [
    {
                "id": "chunk001",
                "text": "Basketball was invented by Dr. James Naismith in December 1891 in Springfield, Massachusetts."
        },
        {
                "id": "chunk002",
                "description": "The first game of basketball was played on December 21, 1891, and it ended with a score of 1-0."
        },
]

documentDB.indexDocuments(documents)
```

Create pipeline object
```python
pipe = RagPipe()
```

Connect LLM to pipeline
```python
pipe.connectLLM(LLM_URL, LLM_NAME)
```

Link pipeline object to vector data base
```python
pipe.connectVectorStore(documentDB)
```

Run a single test query with rag pipeline
```python
pipe.answerQuery("Who invented basketball?")
```

Or run the rag pipeline with costum settings and multiple queries 
```python
pipe.setConfigs(
    lang="EN",
    query_expansion=1,
    rerank=False,
    prepost_context=False,
    background_reversed=False,
    search_ref_lex=8,
    search_ref_sem=8,
    num_ref_lim=4,
    model_temp=0.1,
    answer_token_num=50,
)
```

Run the pipeline with a list of queries
```python
queries = ["Who is him", "Who invented basketball?", "What is the temperature of the sun?"]
pipe.run(
        queries,
)
```

Look at the results 
```python
for elem in pipe.rag_elements:
        print(elem)
```

Evaluate the results

```python
methods = "context_relevance", "answer_relevance", "faithfulness", "correctness", "all"

evaluator = "llm_judge", "ROUGE-1"
scores = evaluate(method="all", evaluator="ROUGE-1")

print(scores)
```

Write evaluation results to a csv file

```python
write_correctness_to_csv(filename="my-first-results", scores=scores, evaluator="ROUGE-1")
```



Load dataset


```python
# Load Mini Wikipedia dataset
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths = datasetHelpers.loadMiniWiki()
```


## Other useful methods

Set costum prompt
```python
userPrompt = "Give an answer using simple language."
# userPrompt = "Verwende wissenschaftliche Sprache für deine Antwort."
pipe.setCostumPrompt(userPrompt)
```


Get stats of a specific index:

```python
stats = documentDB.getIndexStats()
print(stats)
```

View indexes:

```python
print(documentDB.getIndexes())
```

Delete Index:

```python
documentDB.deleteIndex("my-least-favourite-index")
```

Delete all documents in current index:

```python
documentDB.emptyIndex()
```


## Example run
- If you have access to the AIT servers, you can run some of the scripts in bin/
- Have a look at bin/example-rag.py 
- Run `python3 bin/example-rag.py` from the root directory.


## License
AIT internal use only !
If you have any questions contact simon.koenig@ait.ac.at

