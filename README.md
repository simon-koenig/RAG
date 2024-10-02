# Parameter Development AI(T) Chatbot -- Under Developement

This repository aims at building a comprehensive RAG system evaluation. The goal is to give easy to use paramter tuning for RAG systems.

Paramters can be tweaked and tuned with the help of 4 evalution metrics: 

- Context Relevance
- Answer Relevance
- Faithfullness
- Correctness

TODO: Give brief explanation on metrics


## Requirements
- Tested with 
Ubuntu 22.04.3
- Run the following command to install the required packages:
`pip install -r requirements.txt`


## Dataset preparation
Depending on the dataset you want to use. 
Generic dataset import - under developement. 

## Tunable parameters
- Reranking of index search results
- Enriching index search results with pre text and post text
- Large language model
- Temperature of large language model 


## RAG system methodology

- Index documents into vector database
- Ask query
- Retrieve documents related to query from database
- Define a prompt with instructions to an llm on how it should process information
- Combine retrieved documents with query and send to llm 
- Get llm answer based on the provided context

## Workflow example

Imports

```python
from csv_write import write_correctness_to_csv
from dataset_helpers import DatasetHelpers
from pipe import RagPipe
from vector_store import VectorStore
from evaluate import evaluate
```

Define API ENDPOINTS, e.g.:

```python
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"
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
    search_ref_lex=2,
    search_ref_sem=2,
    num_ref_lim=4,
    model_temp=0.1,
    answer_token_num=50,
)

# queries = ["Who is him", "Who invented basketball?", "What is the temperature of the sun?"]
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

evaluator = "sem_similarity", "llm_judge"
scores = evaluate(method="all", evaluator="sem_similarity")

print(scores)
```

Write evaluation results to a csv file

```python
write_correctness_to_csv("my-first-results", scores, evaluator="sem_similarity")
```



Load dataset


```python
# Load Mini Wikipedia dataset
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths = datasetHelpers.loadMiniWiki()
```


## Other useful methods


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
- Have a look at bin/example-rag.py 
- Run `python3 bin/example-rag.py` from the root directory.



## License

AIT internal use only !

