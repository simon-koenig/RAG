import marqo

MARQO_URL = "http://10.103.251.104:8880"
mq = marqo.Client(url=MARQO_URL)

mq.create_index("my-first-index", model="hf/e5-base-v2")

mq.index("my-first-index").add_documents(
    [
        {
            "Title": "The Travels of Marco Polo",
            "Description": "A 13th-century travelogue describing Polo's travels",
        },
        {
            "Title": "Extravehicular Mobility Unit (EMU)",
            "Description": "The EMU is a spacesuit that provides environmental protection, "
            "mobility, life support, and communications for astronauts",
            "_id": "article_591",
        },
    ],
    tensor_fields=["Description"],
    device="cuda",
)

results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)
print(results)

results = mq.index("my-first-index").delete()
print(results)
