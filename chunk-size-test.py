import datetime
import os
import shutil
import hashlib
import fitz
import configparser
from gpt4all import GPT4All
import marqo
import requests
import streamlit as st
from minio import Minio
import pprint

# from https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file


def hash_bytestr_iter(bytesiter, hasher, ashexstr=True):
    for block in bytesiter:
        hasher.update(block)
    return hasher.hexdigest() if ashexstr else hasher.digest()


def file_as_blockiter(afile, blocksize=65536):
    with afile:
        block = afile.read(blocksize)
        while len(block) > 0:
            yield block
            block = afile.read(blocksize)

# from https://discuss.streamlit.io/t/how-to-sanitize-user-input-for-markdown/828/4


def escape_markdown(text):
    MD_SPECIAL_CHARS = "\`*_{}[]()#+-.!"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\"+char)
    text = text.strip()
    return text


def main():
    """
    Chunk size Parameter Testing 
    """

    print("*****************************************************************")
    print("*****************************************************************")
    print("*****************************************************************")
    print("******** Parameter Tuning Chatbot - Text ingest chunk size ******")
    print("*****************************************************************")
    print("*****************************************************************")
    print("*****************************************************************")

    def get_model(endpoint):
        # Get the model name from the OpenAI endpoint
        url = endpoint + '/models'
        try:
            result = requests.get(url).json()
        except:
            print("Error:")
            print("No response from the OpenAI endpoint: " + ENDPOINT)
            print("Please ensure that OpenAI is running and re-start the application.")
            exit(1)
        try:
            return result["data"][0]["id"]
        except:
            print("Error:")
            print("Bad response from the OpenAI endpoint: " + ENDPOINT)
            print(
                "Please ensure that OpenAI is properly configured and re-start the application.")
            exit(1)

    def get_marqo_client(murl):
        # Create a connection with the marqo instance
        return marqo.Client(url=murl)

    def get_minio_client(url, access_key, secret_key):
        # Create a connection with the minio instance
        return Minio(url, access_key=access_key, secret_key=secret_key, secure=False, region="ait")

    def get_gptj_model(model, path):
        # Create a database session object that points to the URL.
        if path == "":
            return GPT4All(model)
        else:
            return GPT4All(model, model_path=path)

    def get_config(config_file):
        # Read the configuration file
        parser = configparser.ConfigParser()
        parser.read(config_file)
        return parser

    def get_indices():
        # Get the available marqo indices
        index_results = []
        model_results = []
        try:
            indices = mq.get_indexes()
            models = mq.get_loaded_models()
        except:
            print(
                f"Error: No response from the marqo endpoint: { MARQO_ENDPOINT } \n")
            print(f"Please ensure that marqo is running and re-start the application.")
            exit(1)
        if len(indices["results"]) < 1:
            mq.create_index("test")
            index_results.append("test")
            print("Error:")
            print(
                "No indices found at the marqo endpoint: {MARQO_ENDPOINT} \n")
            print("A test index has been created.")
        else:
            for index in indices["results"]:
                index_results.append(index.index_name)
            index_results.sort()
        if len(models["models"]) < 1:
            st.subheader("Error:")
            print(
                f"No models found at the marqo endpoint: { MARQO_ENDPOINT } \n ")
            print("Please add a model and re-start the application.")
            exit(1)
        else:
            for model in models["models"]:
                model_results.append(model["model_name"])

        return index_results, model_results

    def get_token_count(text):
        # try to count the number of tokens in this text segment
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {APIKEY}'
        }
        data = {
            "model": OPENAIMODEL,
            "input": text
        }
        endpoint = ENDPOINT + '/embeddings'
        reply = requests.post(endpoint, headers=headers, json=data).json()
        if "usage" in reply:
            tokens = reply["usage"]["total_tokens"]
        else:
            # OpenAI suggests that, on average, there are four characters per token
            # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
            tokens = len(text) / 4
        return tokens

    cfg = get_config('config.ini')
    INPUT_DIRECTORY = cfg['DEFAULT']['INPUT_DIRECTORY']
    OUTPUT_DIRECTORY = cfg['DEFAULT']['OUTPUT_DIRECTORY']
    MARQO_ENDPOINT = cfg['DEFAULT']['MARQO_ENDPOINT']
    GPT4ALLMODEL = cfg['DEFAULT']['GPT4ALLMODEL']
    GPT4ALLMODELPATH = cfg['DEFAULT']['GPT4ALLMODELPATH']
    USEOPENAI = cfg['OPENAI']['USEOPENAI']
    APIKEY = cfg['OPENAI']['APIKEY']
    ENDPOINT = cfg['OPENAI']['ENDPOINT']
    MAXTOKENS = int(cfg['OPENAI']['MAXTOKENS'])
    MINIO_ENDPOINT = cfg['MINIO']['MINIO_ENDPOINT']
    MINIO_ACCESS_KEY = cfg['MINIO']['MINIO_ACCESS_KEY']
    MINIO_SECRET_KEY = cfg['MINIO']['MINIO_SECRET_KEY']

    # Use OpenAI API or local GPTALLMODEL
    uselocal = True
    used_model = GPT4ALLMODEL + " (local)"
    if USEOPENAI == "True":
        uselocal = False
        OPENAIMODEL = get_model(ENDPOINT)
        used_model = OPENAIMODEL + " (remote)"

    mq = get_marqo_client(MARQO_ENDPOINT)
    # Here the current indices and models are retrieved.
    indices, marqo_models = get_indices()
    # Select default index and marqo model
    marqo_model = marqo_models[0]  # Can be tuned
    index = indices[0]  # Can be tuned

    minioclient = get_minio_client(
        MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)

    if (uselocal):
        gptj = get_gptj_model(GPT4ALLMODEL, GPT4ALLMODELPATH)

    ##
    # Create Index
    ##

    # marqo_model = st.selectbox("Marqo indexing model", marqo_models)
    # space_type = st.selectbox("Distance measure", ("cosinesimil", "l1", "l2", "linf"))
    # split_method = st.selectbox("Split method", ("sentence", "word", "character", "passage"))
    # new_index = st.text_input('New Index identifier')
    space_type = "cosinesimil"  # Can be tuned
    split_method = "sentence"  # Can be tuned

    if input("Create a new Index? [y/n] ") == ("y" or "Y"):
        new_index = input("Name of new Index: ")  # has to be set by user

        if len(new_index) > 0:
            new_index = new_index.lower()
            if new_index in indices:
                print(f"Index already exists: {new_index} ")
            else:
                index_settings = {
                    "index_defaults": {
                        "treat_urls_and_pointers_as_images": False,
                        "model": marqo_model,
                        "normalize_embeddings": True,
                        "text_preprocessing": {
                            "split_length": 2,
                            "split_overlap": 0,
                            "split_method": split_method
                        },
                        "image_preprocessing": {
                            "patch_method": None
                        },
                        "ann_parameters": {
                            "space_type": space_type,
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    },
                    "number_of_shards": 5,
                    "number_of_replicas": 1
                }
                try:
                    mq.create_index(new_index, settings_dict=index_settings)
                    print("New index created: " + new_index)
                    # force an update of the index selectbox
                    indices, marqo_models = get_indices()
                    index = new_index  # set new index to current index
                except:
                    print("Failed to created new index: " +
                          new_index + " - check marqo endpoint!")
        else:
            print("Error: no index identifier defined.")

    else:
        # Select index and paramters
        if input(f"Current Document Index is {index}. Keep it? (y/n) ") == ("n" or "N"):
            new_index = input(f"Choose between: \n {indices}\n")
            if new_index not in indices:
                print(f"Index {new_index} not available. ")
            else:
                index = new_index
                print(f"Index set to {new_index}")

    def upload(filename, input_file):
        # Now compute the hash value of the file, take the first 16 characters as a unique identifier
        fuid = hash_bytestr_iter(file_as_blockiter(
            open(input_file, 'rb')), hashlib.sha256())[:16]
        # check if a minio bucket corresponding to the index exists; if not, create it
        if not minioclient.bucket_exists(index):
            print("Creating minio bucket corresponding to index: " + index)
            minioclient.make_bucket(index)
        # check if the object already exists
        found = False
        try:
            minioclient.stat_object(index, fuid)
            found = True
            print("An object already exists with this id: " + fuid)
            st.write("Aborted: " + filename + " already exists in " + index)
        except:
            print("A new object will be added to minio with id: " + fuid)

        if (found):
            # remove the file from the ingest directory
            os.remove(input_file)
            return False
        else:
            # now upload this file to the minio endpoint
            minioclient.fput_object(index, fuid, input_file)
            # add an index record to marqo
            mq.index(index).add_documents([
                {
                    "filename": filename,
                    "fuid": fuid,
                    "index": True
                }],
                tensor_fields=["filename"],
            )
            # extract text from the PDF
            with fitz.open(input_file) as doc:
                pgnum = 1
                for page in doc:
                    # TODO: Maybe adapt this to be sentences not paragraphs ("blocks")
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if block[6] == 0:  # that is, if it is a text block
                            parag = block[4]
                            paranum = block[5]
                            if "\n" in parag:
                                parag = parag.replace("\n", " ")
                            if "- " in parag:
                                parag = parag.replace("- ", "")
                            if "ﬁ" in parag:
                                parag = parag.replace("ﬁ", "fi")
                            if "ﬂ" in parag:
                                parag = parag.replace("ﬂ", "fl")
                            parag = parag.encode(
                                "utf-8", errors='replace').decode("utf-8")
                            letters = sum(c.isalpha() for c in parag)
                            # Changed values here to represent a sentence. toy approach for test purposes. TODO: Adapt to be more general.
                            # we are only interested in paragraphs with a certain number of characters (not just numbers), but also not too long
                            if letters > 256 and letters < 1024:
                                if not uselocal:
                                    tokens = get_token_count(parag)
                                else:
                                    tokens = len(parag) / 4
                                try:
                                    mq.index(index).add_documents([
                                        {
                                            "Title": filename,
                                            "Text": parag,
                                            "Page": pgnum,
                                            "Paragraph": paranum,
                                            "tokens": tokens,
                                            "fuid": fuid
                                        }],
                                        # Arguments in tensor_fields will have vectors generated for them. For best recall and performance, minimise the number of arguments in tensor_fields.
                                        tensor_fields=["Title", "Text"],

                                    )
                                except:
                                    st.write("Ingest error: " + filename + ", page: " +
                                             str(pgnum) + ", paragraph: " + str(paranum))
                    pgnum += 1
            return True

    ##
    # Ingest Data with variable chunk size to index
    ##

    print("Directory ingest to index: " + index)
    print("Add Files from the input directory:")

    # Ingest new files

    if input("Ingest new files [y/n]") == ("y" or "Y"):
        # Ingest files in INPUT_DIRECTORY
        directory = os.fsencode(INPUT_DIRECTORY)
        files = os.listdir(directory)
        numentries = len(files)
        if numentries < 1:
            print("No files available in the input folder.")
        else:
            numingested = 0
            # Add a placeholder for the status bar
            for file in files:
                filename = os.fsdecode(file)
                numingested += 1
                print(f'Ingesting file:\n{filename}')
                if filename.endswith(".pdf"):
                    input_file = os.path.join(INPUT_DIRECTORY, filename)
                    output_directory = os.path.join(OUTPUT_DIRECTORY, index)
                    if not (os.path.exists(output_directory)):
                        os.makedirs(output_directory)
                    output_file = os.path.join(
                        OUTPUT_DIRECTORY, index, filename)
                    if upload(filename, input_file):
                        # move the file to the output directory
                        os.rename(input_file, output_file)
            print("Ingest completed.")
    else:
        print("No ingest performed!")

    ##
    # Modify query prompt
    ##

    system_prompt_en = 'You are a helpful assistant. You are designed to be as helpful as possible while providing only factual information. Context information is given in the following paragraph: {BG} Given this context information and not prior knowledge, answer the following user query.'

    system_prompt_de = 'Sie sind ein hilfreicher Assistent. Sie sollen so hilfreich wie möglich sein und nur sachliche Informationen liefern. Kontextinformationen finden Sie im folgenden Absatz: {BG} Beantworten Sie anhand dieser Kontextinformationen und ohne Vorkenntnisse die folgende Benutzeranfrage.'

    system_prompt_en_token_count = 45

    system_prompt_de_token_count = 65

    # Set to True if background information should be inserted at a specific place in the prompt.
    if input("Define the system prompt ? (y/n)") == ("y" or "Y"):
        print("Use the string {BG} to indicate where the background information should be inserted in the prompt, otherwise, background will be appended to the end of the prompt.")
        print(
            "Example Usage\n: Use this information: {BG} to answer the query:  ")
        system_prompt_en = input("System Prompt (EN)")
        system_prompt_de = input("System Prompt (DE)")
        system_prompt_en_token_count = get_token_count(system_prompt_en)
        system_prompt_de_token_count = get_token_count(system_prompt_de)

    ##
    # Run Query string + background from index against LLM
    ##

    print("Model Parameters")
    print(f"Used model: {used_model}")
    # num_ref = st.slider("Number of Sources", min_value=1, max_value=5, value=3, step=1)
    # answer_size = st.slider("Response Length", min_value=64, max_value=512, value=256, step=64)
    # repeat_pen = st.slider("Repeat Penalty", min_value=-2.0, max_value=2.0, value=1.1, step=0.1)
    # presence_pen = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=1.2, step=0.1)
    # model_temp = st.slider("Temperature", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
    # query_type = st.radio("Search type", ('Semantic', 'Lexical'))
    # lang = st.radio('Response Language', ('EN', 'DE'))
    # query_threshold = st.slider("Query Threshold", min_value=0.6, max_value=0.9, value=0.75, step=0.01)
    # query = st.text_area("LLM Query")

    num_ref = 3
    answer_size = 256
    repeat_pen = 1.0
    presence_pen = 1.0
    model_temp = 0.1
    query_type = "Lexical"
    lang = "EN"
    query_threshold = 0.6
    query = "How many bengal tigers live in the indian subcontinent now?"

    ##
    # Submit Query
    ##

    if query == "":
        print("Error: empty query")
    else:
        if query_type == "Semantic":
            results = mq.index(index).search(
                q=query, searchable_attributes=["Text"]
            )
        else:
            results = mq.index(index).search(
                q=query, search_method=marqo.SearchMethods.LEXICAL, searchable_attributes=[
                    "Text"]
            )
        numhits = len(results["hits"])
        if numhits > 0:
            # create a dict of reference URLs, so that minio does not need to be called every time
            durldict = dict()
            if lang == "DE":
                query = query + " Bitte auf Deutsch antworten!"
            # Will hold text from database sources. Query + background is the new query for the LLM.
            background = ""
            sources = ""
            # Initialize token count with the expected answer size and the query length
            # OpenAI suggests that, on average, there are four characters per token
            # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
            numtokens = answer_size + (len(query)/4)
            if numhits > num_ref:
                numhits = num_ref
            num_sources = 0
            for i in range(numhits):
                score = float(results["hits"][i]["_score"])
                if score >= query_threshold:
                    # get a download link from minio, see https://min.io/docs/minio/linux/developers/python/API.html#get_presigned_url
                    fuid = results["hits"][i]["fuid"]
                    if fuid in durldict:
                        durl = durldict[fuid]
                    else:
                        durl = minioclient.get_presigned_url("GET", index, fuid, expires=datetime.timedelta(
                            days=1), response_headers={"response-content-type": "application/pdf"})
                        durldict[fuid] = durl
                    # Add the pre-calculated token length of the source text
                    # In this manner, we ensure that the context window of model is not exceeded by the number tokens in the sources, which was counted at ingest
                    numtokens += results["hits"][i]["tokens"]
                    sourcetext = results["hits"][i]["Text"]
                    sourcetext = escape_markdown(sourcetext)
                    if numtokens < MAXTOKENS:
                        scorestring = f'{results["hits"][i]["_score"]:.2f}'
                        refstring = "[" + results["hits"][i]["Title"] + ", page " + str(
                            results["hits"][i]["Page"]) + ", paragraph " + str(results["hits"][i]["Paragraph"]) + "]\n"
                        sources += str(i+1) + ". " + sourcetext + \
                            " [" + refstring + \
                            "](" + durl + ") (Score: " + scorestring + ")\n"
                        background += results["hits"][i]["Text"] + " "
                        num_sources += 1
                    else:
                        print(
                            "Model token limit exceeded, sources reduced to " + str(i))
                        break
            if num_sources > 0:
                background = "\n" + background + "\n"
                if lang == "EN":
                    if "{BG}" in system_prompt_en:
                        background = system_prompt_en.replace(
                            "{BG}", background)
                    else:
                        background = system_prompt_en + background
                    # Pre-computed number of tokens of the English background and instructions prompt
                    numtokens += system_prompt_en_token_count
                elif lang == "DE":
                    if "{BG}" in system_prompt_de:
                        background = system_prompt_de.replace(
                            "{BG}", background)
                    else:
                        background = system_prompt_de + background
                    # Pre-computed  number of tokens of the German background and instructions prompt
                    numtokens += system_prompt_de_token_count
                messages = [{"role": "system", "content": background}, {
                    "role": "user", "content": query}]
                # Print prompt to the LLM
                # pprint.pprint(messages)
                print('Generating response...')
                # Use local or openAPI model.
                if (uselocal):
                    report = gptj.chat_completion(messages, default_prompt_header=False, streaming=False, temp=model_temp,
                                                  repeat_penalty=repeat_pen, n_ctx=numtokens, n_predict=answer_size)
                else:
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {APIKEY}'
                    }
                    data = {
                        "model": OPENAIMODEL,
                        "messages": messages,
                        "temperature": model_temp,
                        "max_tokens": answer_size,
                        "presence_penalty": presence_pen,
                        "repeat_penalty": repeat_pen
                    }
                    endpoint = ENDPOINT + '/chat/completions'
                    print("Sending query to OpenAI endpoint: " + endpoint)
                    report = requests.post(
                        endpoint, headers=headers, json=data).json()
                    print("Received response...")
                    if "choices" in report:
                        if len(report["choices"]) > 0:
                            result = report["choices"][0]["message"]["content"]
                        else:
                            result = "No result generated!"
                    else:
                        result = report
                    print("Response: \n")
                    pprint.pprint(result)
                    print("Sourcers: \n")
                    pprint.pprint(sources)
            else:
                print("No results over the threshold (" +
                      str(query_threshold) + ") returned for that query.")
        else:
            print("Error:\n")
            print(
                "No results returned from a search on that query. Perhaps the index is empty?")

    ##
    # Store output and assign a quality/objective to the output.
    ##

    # answers = []
    # objective = analyse(answers)


if __name__ == '__main__':
    main()
