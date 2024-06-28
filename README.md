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

## Tunable parameters, with examples
- chunking_params:
    - chunk_size: 512  # Number of characters in a chunk
    - chunk_overlap: 0  # Number of overlapping characters in a chunk
    - chunk_method: "recursive"  # Method of chunking
- index_settings:
    - split_length: 2  # Number of elements in a split
    - split_method: "sentence"  # Method of splitting
    - split_overlap: 0  # Number of overlapping tokens in a split
    - distance_metric: "prenormalized-angular"  # Distance metric for ANN
    - model: "flax-sentence-embeddings/all_datasets_v4_mpnet-base"  # Model for vector embedding
- retrieval_settings:
    - k_docs_to_retrieve: 5  # Number of documents to retrieve
    - query_threshold: 0.5  # Query threshold
    - pre_post_context_size: 100  # Number of tokens to add to context
    - rerank: True  # Rerank documents
    - prepost_context: True  # Add pre and post context
- llm_settings:
    - llm_name: "llama3"  # Name of the LLM
    - model_temp: 0.5  # Model temperature
    - answer_size: 100  # Number of tokens in answer


## RAG system methodology

- Index documents into vector database
- Ask query
- Retrieve documents related to query from database
- Define a prompt with instructions to an llm on how it should process information
- Combine retrieved documents with query and send to llm 
- Get llm answer based on the provided context

## Example Usage

- Have a look at bin/example-rag.py 
- Run `python3 bin/example-rag.py` from the root directory.

