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

## Running the Parameter Test

To run the application itself

    ./param-test.sh


## Using the Parameter Testing

The User has to update setting by giving (y/n) answers via CLI interface. Then a predefined query is run against the chosen index with the LLLM, marqo and minio DB endpoints defined in config.ini. 
Output is printed to the CLI.

## Open Issues

The same query in english and german retrieves divergent sources from the vector db. Marqo might calculate less similarity if language of datapoint is different to language in query. 

Something is wrong while retrieving. The scores are way to high. ( ~ 8 ). Solved: Lexical search yields higher score rates.

endpoint/chat/completions: https://www.codecademy.com/learn/intro-to-open-ai-gpt-api/modules/intro-to-open-ai-gpt-api/cheatsheet