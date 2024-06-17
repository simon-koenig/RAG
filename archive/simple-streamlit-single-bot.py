import configparser
import datetime
import re

import marqo
import requests
import streamlit as st
from minio import Minio

##
## Manually Define marqo client, minio client, index, lang, num_ref,maxtokens
##  and all LLM parameters (temp, presence_pen, etc.)


@st.cache_resource
def get_model(endpoint):
    # Get the model name from the OpenAI endpoint
    url = endpoint  #  + "/open-mixtral-8x7b"  # "/Mixtral-8x7B-v0.1"  # + "/models"
    try:
        result = requests.get(url).json()
    except:
        st.subheader("Error:")
        st.write("No response from the OpenAI endpoint: " + url)
        st.write("Please ensure that OpenAI is running and re-start the application.")
        exit(1)
    try:
        return result["data"][0]["id"]
    except:
        st.subheader("Error:")
        st.write("Bad response from the OpenAI endpoint: " + ENDPOINT)
        st.write(
            "Please ensure that OpenAI is properly configured and re-start the application."
        )
        exit(1)


@st.cache_resource
def get_marqo_client(murl):
    # Create a connection with the marqo instance
    return marqo.Client(url=murl)


@st.cache_resource
def get_minio_client(url, access_key, secret_key):
    # Create a connection with the minio instance
    return Minio(
        url,
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
        region="ait",
    )


@st.cache_data
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
        st.subheader("Error:")
        st.write("No response from the marqo endpoint: " + MARQO_ENDPOINT)
        st.write("Please ensure that marqo is running and re-start the application.")
        exit(1)
    if len(indices["results"]) < 1:
        mq.create_index("test")
        index_results.append("test")
        st.subheader("Error:")
        st.write("No indices found at the marqo endpoint: " + MARQO_ENDPOINT)
        st.write("A test index has been created.")
    else:
        for index in indices["results"]:
            index_results.append(index.index_name)
        index_results.sort()
    if len(models["models"]) < 1:
        st.subheader("Error:")
        st.write("No models found at the marqo endpoint: " + MARQO_ENDPOINT)
        st.write("Please add a model and re-start the application.")
        exit(1)
    else:
        for model in models["models"]:
            model_results.append(model["model_name"])

    return index_results, model_results


def get_token_count(text):
    # try to count the number of tokens in this text segment
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APIKEY}",
    }
    data = {"model": OPENAIMODEL, "input": text}
    endpoint = ENDPOINT + "/embeddings"
    reply = requests.post(endpoint, headers=headers, json=data).json()
    if "usage" in reply:
        tokens = reply["usage"]["total_tokens"]
    else:
        # OpenAI suggests that, on average, there are four characters per token
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        tokens = len(text) / 4
    return tokens


def escape_markdown(text):
    MD_SPECIAL_CHARS = "\`*_{}[]()#+-.!"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\" + char)
    text = text.strip()
    return text


def word_count(text):
    return len(re.findall(r"\w+", text))


# Set parameters
cfg = get_config("config.ini")
INPUT_DIRECTORY = cfg["DEFAULT"]["INPUT_DIRECTORY"]
OUTPUT_DIRECTORY = cfg["DEFAULT"]["OUTPUT_DIRECTORY"]
MARQO_ENDPOINT = cfg["DEFAULT"]["MARQO_ENDPOINT"]
GPT4ALLMODEL = cfg["DEFAULT"]["GPT4ALLMODEL"]
GPT4ALLMODELPATH = cfg["DEFAULT"]["GPT4ALLMODELPATH"]
USEOPENAI = cfg["OPENAI"]["USEOPENAI"]
APIKEY = cfg["OPENAI"]["APIKEY"]
ENDPOINT = cfg["OPENAI"]["ENDPOINT"]
MAXTOKENS = int(cfg["OPENAI"]["MAXTOKENS"])
MINIO_ENDPOINT = cfg["MINIO"]["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = cfg["MINIO"]["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = cfg["MINIO"]["MINIO_SECRET_KEY"]

OPENAIMODEL = "mixtral"  # get_model(ENDPOINT)
used_model = OPENAIMODEL + " (remote)"

mq = get_marqo_client(MARQO_ENDPOINT)
indices, marqo_models = get_indices()

minioclient = get_minio_client(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)

lang = "EN"
answer_size = 512
num_ref = 3
query_threshold = 0.5
model_temp = 0.1
presence_pen = 1.0
repeat_pen = 1.0

if "system_prompt_en" not in st.session_state:
    st.session_state.system_prompt_en = "You are a helpful assistant. You are designed to be as helpful as possible while providing only factual information. Context information is given in the following paragraphs."
if "system_prompt_de" not in st.session_state:
    st.session_state.system_prompt_de = "Sie sind ein hilfreicher Assistent. Sie sollen so hilfreich wie möglich sein und nur sachliche Informationen liefern. Kontextinformationen finden Sie im folgenden Absätze."
if "system_prompt_en_token_count" not in st.session_state:
    st.session_state.system_prompt_en_token_count = 30
if "system_prompt_de_token_count" not in st.session_state:
    st.session_state.system_prompt_de_token_count = 38

##
## Sidebar
##

with st.sidebar:
    if st.button("Show terms of use"):
        st.write("When mistrust comes in, loves goes out. --Irish proverb")
    terms_of_use = False
    if st.checkbox("Accept terms of use"):
        terms_of_use = True
    st.subheader("Document Index")
    index_placeholder = st.empty()
    with index_placeholder.container():
        index = st.selectbox("Select Index", indices)

st.title("AIT Chatbot ")
st.caption(
    " 🕹️ Try me! But be careful, my answers might be incorrect and I can give you no warranties! 🕹️"
)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Ask me Anything! Frage mich etwas!"}
    ]

# This prints whole conversation
for msg in st.session_state.messages:
    avatar = "🦜" if msg["role"] == "assistant" else "🤸"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])


###
### Convsersation starts here
###
if query := st.chat_input():
    if not terms_of_use:
        st.info("Please accept the terms of use on the left sidebar to proceed.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user", avatar="🤸").write(query)
    # Augment

    # Get sources from index
    results = mq.index(index).search(q=query, searchable_attributes=["Text"])
    numhits = len(results["hits"])
    if numhits > 0:
        # create a dict of reference URLs, so that minio does not need to be called every time
        durldict = dict()
        if lang == "DE":
            query = query + " Bitte auf Deutsch antworten!"
        background = ""  # Will hold text from database sources. Query + background is the new query for the LLM.
        sources = ""
        # Initialize token count with the expected answer size and the query length
        # OpenAI suggests that, on average, there are four characters per token
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        numtokens = answer_size + (len(query) / 4)
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
                    durl = minioclient.get_presigned_url(
                        "GET",
                        index,
                        fuid,
                        expires=datetime.timedelta(days=1),
                        response_headers={"response-content-type": "application/pdf"},
                    )
                    durldict[fuid] = durl
                # Add the pre-calculated token length of the source text
                # In this manner, we ensure that the context window of model is not exceeded by the number tokens in the sources, which was counted at ingest
                numtokens += results["hits"][i]["tokens"]
                sourcetext = results["hits"][i]["Text"]
                sourcetext = escape_markdown(sourcetext)
                if numtokens < MAXTOKENS:
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
                    st.write("Model token limit exceeded, sources reduced to " + str(i))
                    break
        if num_sources > 0:
            # build an new query structure
            # it consists of prompt, background, query
            if lang == "EN":
                prompt = st.session_state.system_prompt_en
                # Pre-computed number of tokens of the English background and instructions prompt
                numtokens += st.session_state.system_prompt_en_token_count
                query = (
                    "Given this context information and not prior knowledge, answer the following user query. "
                    + query
                )
                numtokens += 16
            elif lang == "DE":
                prompt = st.session_state.system_prompt_de
                # Pre-computed  number of tokens of the German background and instructions prompt
                numtokens += st.session_state.system_prompt_de_token_count
                query = (
                    "Beantworten Sie anhand dieser Kontextinformationen und ohne Vorkenntnisse die folgende Benutzeranfrage. "
                    + query
                )
                numtokens += 28

            # To keep context of previous questions and answers we update the messages constantly
            # with previous information

            # previous_queries = ""
            # for msg in st.session_state.messages:
            #     if msg["role"] == "user":
            #         previous_queries += msg["content"]

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": background},  # + previous_queries},
                {"role": "user", "content": query},
            ]

            with st.spinner("Generating response..."):
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {APIKEY}",
                }
                data = {
                    "model": OPENAIMODEL,
                    "messages": messages,
                    "temperature": model_temp,
                    "max_tokens": answer_size,
                    "presence_penalty": presence_pen,
                    "repeat_penalty": repeat_pen,
                }
                endpoint = ENDPOINT + "/chat/completions"
                print("Sending query to OpenAI endpoint: " + endpoint)
                for message in messages:
                    print(message)
                report = requests.post(endpoint, headers=headers, json=data).json()
                print("Received response...")
                if "choices" in report:
                    if len(report["choices"]) > 0:
                        result = report["choices"][0]["message"]["content"]
                    else:
                        result = "No result generated!"
                else:
                    result = report
                # st.subheader("Response:")
                # Add result to chat
                st.session_state.messages.append(
                    {"role": "assistant", "content": result}
                )
                # Print results by llm
                st.chat_message("assistant", avatar="🦜").write(result)
                # Print sources by vector db lookup
                st.chat_message("assistant", avatar="🦜").write(sources)

                # st.subheader("Sources:")
                # TODO: Display pdfs in own column or pop up with semantic reader of source paragraph.
                # st.chat_message(sources)
                # Append answer to chat history
                # st.session_state.messages.append(
                #     {"role": "assistant", "content": result}
                # )
        else:
            st.write(
                "No results over the threshold ("
                + str(query_threshold)
                + ") returned for that query."
            )
    else:
        st.subheader("Error:")
        st.write(
            "No results returned from a search on that query. Perhaps the index is empty?"
        )
