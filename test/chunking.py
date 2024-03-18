# Aim of this file is to develop different chunking strategies. Can vary from file to file or paragraph to paragraph.

##
# Chunking
##

from langchain.text_splitter import CharacterTextSplitter  # need to install langchain
import fitz
# Funciton to import text from a pdf
file = fitz.open("../../data/animals/Bengal-tiger-facts.pdf")
text = ""
for page in file:
    text += page.get_text()


# Langchain for text splitting
chunk_size = 1024
chunk_overlap = 256
# 1. Fixed size chunking
splitter = CharacterTextSplitter(separator="\n",
                                 chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap)


splitted_text = splitter.create_documents([text])


# 2. Content Aware Chunking
from langchain.text_splitter import NLTKTextSplitter
splitter = NLTKTextSplitter()
splitted_text = splitter.split_text(text)

# 3. Recursive chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
)

splitted_text = splitter.create_documents([text])


### Function to chunk text based on 3 different methods. 
### Text has to be provided. Default values available for method, chunk size and chunk overlap. 
    
def chunkText(text, method="recursive", chunk_size=1024, chunk_overlap=128):

    if method == "recursive":
        splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap)
        
        splitted_text = splitter.split_text(text)

    elif method == "sentence":
        splitter = NLTKTextSplitter()
        splitted_text = splitter.split_text(text)
        
    elif method == "fixed_size":
        splitter = CharacterTextSplitter(separator="\n",
                                 chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap)
        splitted_text = splitter.split_text(text)
         
    


    return [chunk for chunk in splitted_text]


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
result = chunkText(text,method="fixed_size")
for chunk in result:
     print(chunk)
     print("#################################")



