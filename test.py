import configparser
import datetime
import hashlib
import os
import pprint
import re
import shutil

# Ingest files from INPUT_DIRECTORY
directory = os.fsencode("/home/simon/master_project/software/chatbot/data/animals")
print(f"Directory for ingest: {directory}")
files = os.listdir(directory)
print(f"Files for ingest: {files}")

lang = "EN"
answer_size = 512
num_ref = 3
query_threshold = 0.5
model_temp = 0.1
presence_pen = 1.0
repeat_pen = 1.0


curl -X POST http://10.103.251.100:8882/indexes/animal-facts/search \
            --header 'Content-Type: application/json' \
            --data '{
                    "q": "what are bengal tigers",
                    "searchableAttributes": ["Text"]
            }'