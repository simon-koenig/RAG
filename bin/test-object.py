import sys

sys.path.insert(
    0, "/home/simon/master_project/software/chatbot/llm_param_dev"
)  # Adapt this to be your own system path
from src.botObject import Assistant


def main():
    print("************ Main ************")
    A = Assistant(config_file="config.ini", chunking_params=["recursive", 1024, 128])
    A.setConfiguration()
    # TODO: Set default index. Right now setIndex() is required.
    A.setIndex("animal-facts")
    A.submitQuery()

    ##
    ## TODO: Make list of possible entry values for parameters.
    ##

    llm_params = {
        "num_ref": 3,
        "answer_size": 256,
        "repeat_pen": 1.0,
        "presence_pen": 1.0,
        "model_temp": 0.1,
        "query_type": "Semantic",
        "lang": "EN",
        "query_threshold": 0.6,
    }

    index_params = {
        "config_file": "config.ini",
        "chunking_params": [1024, 128, "recursive"],
        "system_prompt_en": "You are a helpful assistant. You are designed to be as helpful as possible while providing only factual information. Context information is given in the following paragraphs.",
        "system_prompt_de": "Sie sind ein hilfreicher Assistent. Sie sollen so hilfreich wie möglich sein und nur sachliche Informationen liefern. Kontextinformationen finden Sie im folgenden Absätze.",
        "system_prompt_en_token_count": 30,
        "system_prompt_de_token_count": 38,
        # TODO: Have the token count be computed implicitly. Also the promp is not an index param but an llm param.
    }

    A.submitQuery(
        query="Can a bengal tiger lift another tiger into the air?",
        llm_params=llm_params,
        update_params=True,
    )


if __name__ == "__main__":
    main()
