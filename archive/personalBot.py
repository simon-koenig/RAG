import configparser
import datetime
import re

import marqo
import requests
import streamlit as st
from minio import Minio

##
## Assistant for personal use
##


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


@st.cache_data
def get_config(config_file):
    # Read the configuration file
    parser = configparser.ConfigParser()
    parser.read(config_file)
    return parser


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


st.title("Simons Chatbot ")
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
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user", avatar="🤸").write(query)

    messages = [
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
        st.session_state.messages.append({"role": "assistant", "content": result})
        # Print results by llm
        st.chat_message("assistant", avatar="🦜").write(result)
