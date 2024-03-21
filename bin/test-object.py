import sys

sys.path.insert(
    0, "/home/simon/master_project/software/chatbot/llm_param_dev"
)  # Adapt this to be your own system path
from src.botObject import Assistant


def main():
    print("************ Main ************")
    llm_params = {
        "num_ref": 3,  # Range: 1 - 10
        "answer_size": 32,  # Range: 128 - 1024
        "repeat_pen": 1.0,  # Range:
        "presence_pen": 1.0,
        "model_temp": 0.5,
        "query_type": "Semantic",
        "lang": "EN",
        "query_threshold": 0.6,
        "system_prompt_en": "You are a helpful assistant. Use soccer analogies in your response please.",
        "system_prompt_de": "Sie sind ein hilfreicher Assistent. Sie sollen so hilfreich wie möglich sein und nur sachliche Informationen liefern. Kontextinformationen finden Sie im folgenden Absätze.",
    }
    ##
    ## TODO: Make list of possible entry values for parameters.
    ##
    chunking_params = {
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "chunk_method": "recursive",
    }

    index_params = {
        "config_file": "config.ini",
        "split_method": "sentence",
        "distance_metric": "cosinesimil",
        "model": "hf/all_datasets_v4_MiniLM-L6",
    }

    A = Assistant(config_file="config.ini", chunking_params=chunking_params)
    A.setConfiguration()
    # TODO: Set default index. Right now setIndex() is required.
    A.setIndex("aitqm")
    # A.submitQuery()

    # A.submitQuery(
    #     query="Can a bengal tiger lift another tiger into the air?",
    #     llm_params=llm_params,
    #     update_params=True,
    # )

    # Make and delete new index with parameters
    A.createIndex(name="new-index", settings=index_params)  # Does not work yet.
    A.ingestFromDirectory(
        settings=chunking_params,
        index="new-index",
    )
    A.deleteFiles(
        index="new-index"
    )  # Does not work yet. Filter string is wrong in index search. Probably set to fuids.
    A.deleteIndex(name="new-index")


if __name__ == "__main__":
    main()
