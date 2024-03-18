# Parameter Development AI(T) Chatbot -- Under Developement

This chatbot uses [marqo](https://www.marqo.ai/), [minio](https://min.io/), [OpenaiAPI](https://platform.openai.com/) and an open source LLM. 
and in particular implements the ideas suggested in [this article](https://medium.com/creator-fund/building-search-engines-that-think-like-humans-e019e6fb6389).

## Requirements

This project was tested under Ubuntu 20.4 with 32GB and 16GB Main Memory - not recommended to run with less than 16GB Main Memory. 

It is assumed that `docker` is installed (tested on version 25.0.3).

## Configuration

Values for the configurations must be defined by copying the configuration template:

    cp config.ini.template config.ini
    
And configuration values must be defined, e.g.

    [DEFAULT]
    INPUT_DIRECTORY = {the directory of where pdfs to be indexed are stored}
    OUTPUT_DIRECTORY = {the dirctory to which indexed PDFs are moved}
    MARQO_ENDPOINT = {marqo api url, typically http://localhost:8882}
    GPT4ALLMODEL = {the GPT4All-compatible model to be loaded}
    GPT4ALLMODELPATH = {the GPT4All-compatible model path}
    [OPENAI]
    USEOPENAI = {If True, then use the defined API endpoint, otherwise default to the local GPT4All model}
    APIKEY = {API key for the endpoint}
    ENDPOINT = {URL of the OpenAI-compatible API endpoint}
    MAXTOKENS = {This is the size of the context window of the selected model, typically 2047}
    [MINIO]
    MINIO_ENDPOINT = {minio Server URL - but this must be in the format localhost:9000}
    MINIO_ACCESS_KEY = {minio access key}
    MINIO_SECRET_KEY = {minio secret key}

It is assumed that at least one index exists at the MARQO_ENDPOINT. If no index exists, the index ``test`` will automatically be created.
It is also possible to manually create an index outside of the application using ``curl``:

    curl -X POST "http://{MARQO_ENDPOINT}/indexes/{index_name}"

Note that ``index_name`` must consist of lowercase characters (including numbers and special characters, no spaces).

Ingested documents are stored in a [minio](https://min.io/) instance. The object id is the first sixteen hex characters of the sha256 hash of the file.
A corresponding bucket (also named ``index_name``) will be created on the minio instance, and files will be uploaded to this bucket.
If a file already exists in that bucket (based on its sha256 hash), it will not be uploaded but rather will be deleted from the input folder.
The ``MINIO_ACCESS_KEY`` and ``MINIO_SECRET_KEY`` must be generated on the minio console. Refer to the [minio README](minio.md) for more details.

Note that the GPT4AllMODELPATH is not defined, then the model is assumed to be stored in

    /home/{user}/.cache/gpt4all

Larger models will of course require more memory.

Several GPT4all models are already available here:

    \\s3store7.d03.arc.local\FSDSS1802\LLM\gpt4all_models

If ``USEOPENAI = True`` then the model name will be queried from the supplied ``ENDPOINT``

## Start-up

**Build**

GPT4All depends on the [llama.cpp project](https://github.com/ggerganov/llama.cpp). Please follow the installation instructions there.

Create a suitable virtual environment with Python 3 and install the requirements:

    pip install -r requirements.txt

**Run**

To start ``marqo``:

    docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest

The marqo server will be available on: http://localhost:8882

Note that this docker image relies on "docker-in-docker" and will download a large docker image inside it.

Therefore, once this image is running, it is best to simply use

    docker stop marqo
    docker start marqo

to stop and start the marqo service.

## Running the PoC

To run the application itself

    ./param-test.sh






    ./qua.sh

The Informed Chatbot PoC will be available on: http://localhost:8511

To attach to the session:

    tmux attach -t qua

In the tmux window, click ``ctrl-b d`` to detach from the tmux session.

To stop the application

    tmux kill-session -t qua

## Using the PoC

The active index is selected on the left sidebar. The model available at the endpoint is displayed, but cannot be changed (as changing the endpoint
requires re-starting streamlit).The configuration of model parameters is also set on the sidebar. 

### Query Tab

In the first column, the language of the response (EN or DE) can be selected. Best results occur when the response language and the language of the query are consistent. It is also better if these languages match the language of the material in the source index. But there can be interesting results with a German query against an English index, and vice versa. 

Queries should be submitted throught the Query text field. Query responses are not streamed and typically take 1-2 minutes to generate for a local model, and 5 to 10 seconds for a remote (GPU-hosted) model. Reponse times also depend on the model parameters.

Chatbot responses, along with the index sources that prompted the response, will be displayed in the second column. Sources provide minio links to the referenced files, which are downloaded and opened in a PDF app when clicked.

### The Search Tab

The user can select the number of desired results (from 1 to 10) with the associated slider control.

In the first column, the user can enter a query, which can be either keywords or natural sentences.

In the second column, search results will be displayed, along with minio links to the referenced files, which are downloaded and opened in a PDF app when clicked.

### The Index Tab

In the first column, the user can dynamically create a new marqo index, along with some key indexing paramters [explained here](https://docs.marqo.ai/0.0.21/API-Reference/indexes/).

In the second column, it is possible to delete the active index. This action cannot be undone and removes the entire marqo index and the entire associate minio bucket, including all files. Thus the user is prompted to confirm this irrevocable action.

### The Ingest Tab

File uploads will be directed to the active index (selected on the sidebar). In the first column, the ``Ingest`` button will index any PDF files in the configured ``input`` folder and move them to the configured ``ingested`` folder, where a subfolder for the selected index (defined on the sidebar) will be created.

In the second column, a single file or multiple files can be uploaded either through a file browser or by using the drag-and-drop field. Once files have been selected, click the ``Upload All`` button to start the ingest process. At present only PDF files can be selected this way. Each paragraph is indexed separately. The indexing process takes ca. one second per page. The files are also uploaded to the configured minio endpoint, in a bucket with the same name as the index. If the bucket does not exit, it will be created.

In both cases, a hash value for the selected file is calculated and checked against existing hashes in the minio repository. An error is reported if the file already exists.

The text is chunked according to paragraphs by the ``PyMuPDF`` library. As a new feature, the number of tokens in a given chunk is calculated by the OpenAI API ``/embeddings`` endpoint, if the OpenAI compatible model has been selected. With pre-computed token length, we can later be sure that the input prompt plus expected number of output tokens does not exceed the context window of the underlying model.

The third column allows a file object to be replaced. First, a single file must be uploaded. Then the file for target replacement must be selected. Note that the new file may have the same name as the file to be replaced; however, if the file has not changed, the hash value will remain unchanged and nothing will happen. Otherwise, the old file is removed from the marqo index, from minio, and from the ingested folder, and the new file is uploaded.

## Open Issues

* The type of text splitting and indexing for search queries is not optimal for Chatbot queries