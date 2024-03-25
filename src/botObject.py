# Make chatbot object

##
# Pseudo:
##


# chatbot()
# init with params set to default values
# params: params_to_be_tuned / such as temp, chunk size, etc.
#         vector db Index
#         database (minio)
#         LLM
# methods:get, set parameters

#         create Index
#         delte IndexError
#         modify IndexError
#         upload files to index
#         delete files in index
#         modify prompt
#         answer query with option to open a local port to display in web browser
#         store answers in file
#         analyse method / provide question answer pairs and compute chatbot performance on those

import configparser
import datetime
import hashlib
import os
import pprint
import re
import shutil
from dataclasses import dataclass

import fitz
import marqo
import requests
import streamlit as st
from langchain.text_splitter import (
    CharacterTextSplitter,  # need to install langchain
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
)
from minio import Minio


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
        text = text.replace(char, "\\" + char)
    text = text.strip()
    return text


def word_count(text):
    return len(re.findall(r"\w+", text))


###
### Make Assistant Class
###


@dataclass
class Assistant:
    """
    Chatbos as Assistant. Use the assistant to interact with a RAG powerde LLM.
    Set vector DB, NoSQL DB and LLM endpoints in configuration file.
    Set LLM and index parameters that suit your application.
    """

    config_file: str
    chunking_params: dict  # TODO: This should be index specific.
    system_prompt_en: str = "You are a helpful assistant. You are designed to be as helpful as possible while providing only factual information. Context information is given in the following paragraphs."
    system_prompt_de: str = "Sie sind ein hilfreicher Assistent. Sie sollen so hilfreich wie möglich sein und nur sachliche Informationen liefern. Kontextinformationen finden Sie im folgenden Absätze."
    system_prompt_en_token_count: int = word_count(system_prompt_en)
    system_prompt_de_token_count: int = word_count(system_prompt_de)
    num_ref: int = 3
    answer_size: int = 256
    repeat_pen: float = 1.0
    presence_pen: float = 1.0
    model_temp: float = 0.1
    query_type: str = "Lexical"
    lang: str = "EN"
    query_threshold: float = 0.6

    def get_config(self, config_file):
        # Read the configuration file
        parser = configparser.ConfigParser()
        parser.read(config_file)
        return parser

    def setConfiguration(self):
        cfg = self.get_config(config_file=self.config_file)
        self.INPUT_DIRECTORY = cfg["DEFAULT"]["INPUT_DIRECTORY"]
        self.OUTPUT_DIRECTORY = cfg["DEFAULT"]["OUTPUT_DIRECTORY"]
        self.MARQO_ENDPOINT = cfg["DEFAULT"]["MARQO_ENDPOINT"]
        self.GPT4ALLMODEL = cfg["DEFAULT"]["GPT4ALLMODEL"]
        self.GPT4ALLMODELPATH = cfg["DEFAULT"]["GPT4ALLMODELPATH"]
        self.USEOPENAI = cfg["OPENAI"]["USEOPENAI"]
        self.APIKEY = cfg["OPENAI"]["APIKEY"]
        self.ENDPOINT = cfg["OPENAI"]["ENDPOINT"]
        self.MAXTOKENS = int(cfg["OPENAI"]["MAXTOKENS"])
        self.MINIO_ENDPOINT = cfg["MINIO"]["MINIO_ENDPOINT"]
        self.MINIO_ACCESS_KEY = cfg["MINIO"]["MINIO_ACCESS_KEY"]
        self.MINIO_SECRET_KEY = cfg["MINIO"]["MINIO_SECRET_KEY"]

        if self.USEOPENAI == "True":
            self.OPENAIMODEL = self.get_model(self.ENDPOINT)
            self.used_model = self.OPENAIMODEL + " (remote)"

            self.mq = self.get_marqo_client(self.MARQO_ENDPOINT)
            # Here the current indices and models are retrieved.
            self.indices, self.marqo_models = self.putIndices()
            # Select default index and marqo model
            self.marqo_model = self.marqo_models[0]  # TODO:  Can be tuned
            self.index = self.indices[0]  # TODO: Can be tuned. Can it really?

            self.minioclient = self.get_minio_client(
                self.MINIO_ENDPOINT, self.MINIO_ACCESS_KEY, self.MINIO_SECRET_KEY
            )

    def setChunkingParams(self, chunking_params):
        self.chunk_method = chunking_params["chunk_method"]
        self.chunk_size = chunking_params["chunk_size"]
        self.chunk_overlap = chunking_params["chunk_overlap"]

    def get_model(self, endpoint):
        # Get the model name from the OpenAI endpoint
        url = endpoint + "/models"
        try:
            result = requests.get(url).json()
        except:
            print("Error:")
            print("No response from the OpenAI endpoint: " + endpoint)
            print("Please ensure that OpenAI is running and re-start the application.")
            exit(1)
        try:
            return result["data"][0]["id"]
        except:
            print("Error:")
            print("Bad response from the OpenAI endpoint: " + endpoint)
            print(
                "Please ensure that OpenAI is properly configured and re-start the application."
            )
            exit(1)

    def get_marqo_client(self, murl):
        # Create a connection with the marqo instance
        return marqo.Client(url=murl)

    def get_minio_client(self, url, access_key, secret_key):
        # Create a connection with the minio instance
        return Minio(
            url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
            region="ait",
        )

    def createIndex(self, name, settings):
        try:
            split_method = settings["split_method"]
            distance_metric = settings["distance_metric"]
            model = settings["model"]

        except:
            print(
                f"Settings could not be parsed to create a new index with name: {name}"
            )

        name = name.lower()
        if name in self.indices:
            print(f"Index already exists: {name} ")
        else:
            index_settings = {
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": model,
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": split_method,
                    },
                    "image_preprocessing": {"patch_method": None},
                    "ann_parameters": {
                        "space_type": distance_metric,
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
                "number_of_shards": 5,
                "number_of_replicas": 1,
            }
            try:
                self.mq.create_index(name, settings_dict=index_settings)
                print("New index created: " + name)
                # force an update of the index selectbox
                self.indices, self.marqo_models = self.putIndices()
                self.index = name  # set new index to current index
            except:
                print(
                    "Failed to created new index: " + name + " - check marqo endpoint!"
                )

    def deleteIndex(self, name):
        # Delete index by name
        try:
            # get the list of objects in the present bucket
            try:
                objects = self.minioclient.list_objects(name)
                # remove all objects in the bucket
                for object in objects:
                    self.minioclient.remove_object(name, object.object_name)
                # now remove the bucket itself
                self.minioclient.remove_bucket(name)
            except:
                print(
                    "No minio bucket found. Index deletion occurred before any files were added"
                )
            # now remove the marqo index
            self.mq.delete_index(name)
            # now remove the files -- For now, no need to delete the files locally.
            # output_directory = os.path.join(self.OUTPUT_DIRECTORY, name)
            # if os.path.exists(output_directory):
            #     shutil.rmtree(output_directory)
            # else:
            #     print("No associated file directories found and hence none deleted.")

            print(f" Sucessfuylly deleted Index: {name}")
        except:
            print("Unable to delete: " + name)

    # Returns marqo index and marqo models
    def putIndices(self):
        # Get the available marqo indices
        index_results = []
        model_results = []
        try:
            indices = self.mq.get_indexes()
            models = self.mq.get_loaded_models()
        except:
            print(
                f"Error: No response from the marqo endpoint: { self.MARQO_ENDPOINT } \n"
            )
            print("Please ensure that marqo is running and re-start the application.")
            exit(1)
        if len(indices["results"]) < 1:
            self.mq.create_index("test")
            index_results.append("test")
            print("Error:")
            print(f"No indices found at the marqo endpoint: {self.MARQO_ENDPOINT} \n")
            print("A test index has been created.")
        else:
            for index in indices["results"]:
                index_results.append(index.index_name)
            index_results.sort()
        if len(models["models"]) < 1:
            print("Error")
            print(f"No models found at the marqo endpoint: { self.MARQO_ENDPOINT } \n ")
            print("Please add a model and re-start the application.")
            exit(1)
        else:
            for model in models["models"]:
                model_results.append(model["model_name"])

        return index_results, model_results

    # Text chunking
    def chunkText(self, text, method="recursive", chunk_size=1024, chunk_overlap=128):
        if method == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            splitted_text = splitter.split_text(text)

        elif method == "sentence":
            splitter = NLTKTextSplitter()
            splitted_text = splitter.split_text(text)

        elif method == "fixed_size":
            splitter = CharacterTextSplitter(
                separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            splitted_text = splitter.split_text(text)

        return [chunk for chunk in splitted_text]

    # Upload file(s) to index
    def upload(self, filename, input_file, index, chunking_params):
        # Now compute the hash value of the file, take the first 16 characters as a unique identifier
        fuid = hash_bytestr_iter(
            file_as_blockiter(open(input_file, "rb")), hashlib.sha256()
        )[:16]
        # check if a minio bucket corresponding to the index exists; if not, create it
        if not self.minioclient.bucket_exists(index):
            print("Creating minio bucket corresponding to index: " + index)
            self.minioclient.make_bucket(index)
        # check if the object already exists
        found = False
        try:
            self.minioclient.stat_object(index, fuid)
            found = True
            print("An object already exists with this id: " + fuid)
            print("Aborted: " + filename + " already exists in " + index)
        except:
            print("A new object will be added to minio with id: " + fuid)

        if found:
            # remove the file from the ingest directory
            # os.remove(input_file)
            return False
        else:
            # now upload this file to the minio endpoint
            self.minioclient.fput_object(index, fuid, input_file)
            # add an index record to marqo
            self.mq.index(index).add_documents(
                [{"filename": filename, "fuid": fuid, "index": True}],
                tensor_fields=["filename"],
            )
            # extract text from the PDF
            with fitz.open(input_file) as doc:
                pgnum = 1
                for page in doc:
                    # Set chunking variables
                    chunks = self.chunkText(
                        text=page.get_text(),
                        method=chunking_params["chunk_method"],
                        chunk_size=chunking_params["chunk_size"],
                        chunk_overlap=chunking_params["chunk_overlap"],
                    )
                    paranum = 1
                    for chunk in chunks:
                        if chunk:  # Create an entry for each block in the vector db
                            if "\n" in chunk:
                                chunk = chunk.replace("\n", " ")
                            if "- " in chunk:
                                chunk = chunk.replace("- ", "")
                            if "ﬁ" in chunk:
                                chunk = chunk.replace("ﬁ", "fi")
                            if "ﬂ" in chunk:
                                chunk = chunk.replace("ﬂ", "fl")
                            chunk = chunk.encode("utf-8", errors="replace").decode(
                                "utf-8"
                            )
                            letters = sum(c.isalpha() for c in chunk)
                            # Changed values here to represent a sentence. toy approach for test purposes. TODO: Adapt to be more general.
                            # we are only interested in paragraphs with a certain number of characters (not just numbers), but also not too long
                            if letters:
                                # TODO: Make with token count function by openAI API get_token_count
                                tokens = word_count(chunk)
                                try:
                                    self.mq.index(index).add_documents(
                                        [
                                            {
                                                "Title": filename,
                                                "Text": chunk,
                                                "Page": pgnum,
                                                "Paragraph": paranum,
                                                "tokens": tokens,
                                                "fuid": fuid,
                                            }
                                        ],
                                        # Arguments in tensor_fields will have vectors generated for them. For best recall and performance, minimise the number of arguments in tensor_fields.
                                        tensor_fields=["Title", "Text"],
                                    )
                                    # Up paragraph number
                                    paranum += 1
                                except:
                                    st.write(
                                        "Ingest error: "
                                        + filename
                                        + ", page: "
                                        + str(pgnum)
                                        + ", paragraph: "
                                        + str(paranum)
                                    )
                    pgnum += 1
            return True

    # Ingest files to index
    def ingestFromDirectory(self, settings, index=""):
        # Set default index to be current self.index
        if index == "":
            index = self.index

        print("Directory ingest to index: " + index)
        # Ingest new files
        try:
            # Ingest files from INPUT_DIRECTORY
            directory = os.fsencode(self.INPUT_DIRECTORY)
            print(f"Directory for ingest: {directory}")
            files = os.listdir(directory)
            print(f"Files for ingest: {files}")
            numentries = len(files)
            if numentries < 1:
                print("No files available in the input folder.")
            else:
                numingested = 0
                # Iterate over files
                for file in files:
                    filename = os.fsdecode(file)
                    numingested += 1
                    print(f"Ingesting file:\n{filename}")
                    if filename.endswith(".pdf"):
                        input_file = os.path.join(self.INPUT_DIRECTORY, filename)
                        # ******** For now no moving from input to output director *********
                        # output_directory = os.path.join(self.OUTPUT_DIRECTORY, index)
                        # if not (os.path.exists(output_directory)):
                        #     os.makedirs(output_directory)
                        # output_file = os.path.join(
                        #     self.OUTPUT_DIRECTORY, index, filename
                        # )
                        # Perform Upload
                        self.upload(filename, input_file, index, settings)
                    # move the file to the output directory
                    # os.rename(input_file, output_file)
                    else:
                        print(
                            f"No .pdf provided. File not ingested into index: {index}"
                        )
                print("Ingest completed.")
        except:
            print("No ingest performed!")

    def ingestFiles(self, upload_files, settings, index=""):
        # Set default index to be current self.index
        if index == "":
            index = self.index

        print("Manually file ingest to index: " + index)
        try:
            output_directory = os.path.join(self.OUTPUT_DIRECTORY, index)
            if not (os.path.exists(output_directory)):
                os.makedirs(output_directory)
            for file in upload_files:
                _, filename = os.path.split(file)
                output_file = os.path.join(self.OUTPUT_DIRECTORY, index, filename)
                if os.path.exists(output_file):
                    print(
                        f"A file with this name has already been ingested:{output_file}"
                    )
                else:
                    # write to the ingested directory
                    with open(output_file, "wb") as f:
                        f.write(file.getbuffer())
                    if self.upload(filename, file, index, settings):
                        print(f"File ingested: {filename}")

        except:
            print(f" Ingest did not work for {filename}. Only pdfs are accepted!")

    def deleteFiles(self, index):
        # Deletes all files in index
        objectkeys = {}
        all = self.mq.index(index).search(q="", limit=1000, filter_string="index:false")
        if len(all["hits"]) < 1:
            print("Index is empty, there is nothing to delete")
        else:
            delete_ids = []
            for hit in all["hits"]:
                delete_ids.append(hit["_id"])
                # Delete object from minio with fuid
                self.minioclient.remove_object(index, hit["fuid"])
                # Deletes documents on the delete_ids list
                self.mq.index(index).delete_documents(ids=hit["_id"])

    def modifyPrompt(self, text, lang="EN"):
        if lang == "EN":
            self.system_prompt_en = text
            self.system_prompt_en_token_count = word_count(text)

        elif lang == "DE":
            self.system_prompt_de = text
            self.system_prompt_de_token_count = word_count(text)

        else:
            print(
                f" Wrong language submitted. Prompt remains unchanged: {self.system_prompt_en} "
            )

    ###
    ### Combine backround, prompt and query to the llm and get an answer.
    ###

    def submitQuery(
        self,
        query="Tell user he did not submit a query!",
        llm_params={},
        update_params=False,
    ):
        # TODO: Submit query to LLM and against Index. Geneate Output.
        if update_params is True:
            num_ref = (
                llm_params["num_ref"]
                if llm_params.get("num_ref", False)
                else self.num_ref
            )

            answer_size = (
                llm_params["answer_size"]
                if llm_params.get("answer_size", False)
                else self.answer_size
            )

            repeat_pen = (
                llm_params["repeat_pen"]
                if llm_params.get("repeat_pen", False)
                else self.repeat_pen
            )

            presence_pen = (
                llm_params["presence_pen"]
                if llm_params.get("presence_pen", False)
                else self.presence_pen
            )

            model_temp = (
                llm_params["model_temp"]
                if llm_params.get("model_temp", False)
                else self.model_temp
            )
            query_type = (
                llm_params["query_type"]
                if llm_params.get("query_type", False)
                else self.query_type
            )
            lang = llm_params["lang"] if llm_params.get("lang", False) else self.lang
            query_threshold = (
                llm_params["query_threshold"]
                if llm_params.get("query_threshold", False)
                else self.query_threshold
            )
            system_prompt_de = (
                llm_params["system_prompt_de"]
                if llm_params.get("system_prompt_de", False)
                else self.system_prompt_de
            )
            system_prompt_en = (
                llm_params["system_prompt_en"]
                if llm_params.get("system_prompt_en", False)
                else self.system_prompt_en
            )

            system_prompt_de_token_count = word_count(system_prompt_de)
            system_prompt_en_token_count = word_count(system_prompt_en)
        else:
            num_ref = self.num_ref
            answer_size = self.answer_size
            repeat_pen = self.repeat_pen
            presence_pen = self.presence_pen
            model_temp = self.model_temp
            query_type = self.query_type
            lang = self.lang
            query_threshold = self.query_threshold
            system_prompt_de = self.system_prompt_de
            system_prompt_en = self.system_prompt_en
            system_prompt_de_token_count = self.system_prompt_de_token_count
            system_prompt_en_token_count = self.system_prompt_en_token_count

        # query = "How many bengal tigers live in the indian subcontinent now?"

        if query != "":
            if query_type == "Semantic":
                results = self.mq.index(self.index).search(
                    q=query, searchable_attributes=["Text"]
                )
            else:
                results = self.mq.index(self.index).search(
                    q=query,
                    search_method=marqo.SearchMethods.LEXICAL,
                    searchable_attributes=["Text"],
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
                numtokens = answer_size + word_count(query)
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
                            durl = self.minioclient.get_presigned_url(
                                "GET",
                                self.index,
                                fuid,
                                expires=datetime.timedelta(days=1),
                                response_headers={
                                    "response-content-type": "application/pdf"
                                },
                            )
                            durldict[fuid] = durl
                        # Add the pre-calculated token length of the source text
                        # In this manner, we ensure that the context window of
                        # is not exceeded by the number tokens in the sources, which was counted at ingest
                        numtokens += results["hits"][i]["tokens"]
                        sourcetext = results["hits"][i]["Text"]
                        sourcetext = escape_markdown(sourcetext)
                        if numtokens < self.MAXTOKENS:
                            scorestring = f'{results["hits"][i]["_score"]:.2f}'
                            refstring = (
                                "["
                                + results["hits"][i]["Title"]
                                + ", page "
                                + str(results["hits"][i]["Page"])
                                + ", paragraph "
                                + str(results["hits"][i]["Paragraph"])
                                + "]\n"
                            )
                            sources += (
                                str(i + 1)
                                + ". "
                                + sourcetext
                                + " ["
                                + refstring
                                + "]("
                                + durl
                                + ") (Score: "
                                + scorestring
                                + ")\n"
                            )
                            background += results["hits"][i]["Text"] + " "
                            num_sources += 1
                        else:
                            print(
                                "Model token limit exceeded, sources reduced to "
                                + str(i)
                            )
                            break
                if num_sources > 0:
                    # build an new query structure
                    # it consists of prompt, background, query
                    if lang == "EN":
                        prompt = system_prompt_en
                        # Pre-computed number of tokens of the English background and instructions prompt
                        numtokens += system_prompt_en_token_count
                        query = (
                            "Given this context information and not prior knowledge, answer the following user query. "
                            + query
                        )
                        numtokens += word_count(query)
                    elif lang == "DE":
                        prompt = system_prompt_de
                        # Pre-computed  number of tokens of the German background and instructions prompt
                        numtokens += system_prompt_de_token_count
                        query = (
                            "Beantworten Sie anhand dieser Kontextinformationen und ohne Vorkenntnisse die folgende Benutzeranfrage. "
                            + query
                        )
                        numtokens += word_count(query)
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": background},
                        {"role": "user", "content": query},
                    ]
                    print(f"The prompt is = {prompt}")
                    print("Generating response...")
                    # Use local or openAPI model.
                    # if uselocal:
                    #     report = gptj.chat_completion(
                    #         messages,
                    #         default_prompt_header=False,
                    #         streaming=False,
                    #         temp=model_temp,
                    #         repeat_penalty=repeat_pen,
                    #         n_ctx=numtokens,
                    #         n_predict=answer_size,
                    #     )
                    if True:
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.APIKEY}",
                        }
                        data = {
                            "model": self.OPENAIMODEL,
                            "messages": messages,
                            "temperature": model_temp,
                            "max_tokens": answer_size,
                            "presence_penalty": presence_pen,
                            "repeat_penalty": repeat_pen,
                        }
                        endpoint = self.ENDPOINT + "/chat/completions"
                        print("Sending query to OpenAI endpoint: " + endpoint)
                        report = requests.post(
                            endpoint, headers=headers, json=data
                        ).json()
                        print("Received response...")
                        if "choices" in report:
                            if (
                                len(report["choices"]) > 0
                            ):  # TODO: We are always taking the first choice.
                                result = report["choices"][0]["message"]["content"]
                            else:
                                result = "No result generated!"
                        else:
                            result = report
                        print("Response: \n")
                        pprint.pprint(result)
                        # print("Sources: \n")
                        # pprint.pprint(sources, width=100)
                else:
                    print(
                        "No results over the threshold ("
                        + str(query_threshold)
                        + ") returned for that query."
                    )
            else:
                print("Error:\n")
                print(
                    "No results returned from a search on that query. Perhaps the index is empty?"
                )

        else:
            print("Error: Empty query!")

    #####################################################################################
    ########################## Get and Set Functions ####################################
    #####################################################################################
    def getIndices(self):
        return self.indices

    def getIndexModels(self):
        return self.marqo_models

    def getCurrentIndex(self):
        return self.index

    def getCurrentIndexModel(self):
        return self.marqo_model

    def setIndex(self, index):
        self.index = index
        print(self.index)

    def setIndexModel(self, marqo_model):
        self.marqo_model = marqo_model
        print(self.marqo_model)

    def getFilesInIndex(self, index):
        files = []
        # for item in self.minioclient.list_objects(index, recursive=True):
        #     files.append(
        #         # self.minioclient.fget_object(index, item.object_name, item.object_name)
        #         item
        #     )
        # files = self.mq.index("index").get_documents()
        print("Does not work just yet.")
        # for object in self.minioclient.list_objects(index):
        #     files.append(object.object_name)
        result = self.mq.index(index).search(
            q="",
            limit=1000,
            filter_string="index:true",  # Gets the 1000 first matches
        )

        # Get all filenames
        [files.append(hit["filename"]) for hit in result["hits"]]

        return files

    def getObjectsinBucket(self, index):
        files = []
        for item in self.minioclient.list_objects(index):
            files.append(item)

        return files


def main():
    print("************ Main ************")
    A = Assistant(config_file="config.ini", chunking_params=["recursive", 1024, 128])
    A.setConfiguration()
    pprint.pprint(A.getIndices(), width=20)
    pprint.pprint(A.getCurrentIndex(), width=20)
    # Set new index for fun
    A.setIndex("animal-facts")

    # Create test index for fun

    # A.createIndex(
    #    index_name="test-index", split_method="sentence", distance_metric="cosinesimil"
    # )
    pprint.pprint(A.getIndices(), width=20)
    pprint.pprint(A.getCurrentIndex(), width=20)
    # Get files from index
    pprint.pprint(A.getObjectsinBucket("animal-facts"))
    A.submitQuery()
    llm_params = {
        "num_ref": 3,
        "answer_size": 24,
        "repeat_pen": 1.0,
        "presence_pen": 1.0,
        "model_temp": 0.1,
        "query_type": "Lexical",
        "lang": "EN",
        "query_threshold": 0.6,
    }

    A.submitQuery(llm_params=llm_params, update_params=True)


if __name__ == "__main__":
    main()
