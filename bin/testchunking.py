# Test of rag evaluation
# Imports
import sys
from pprint import pprint

import openpyxl

sys.path.append("./dev/")
from components import (
    DatasetHelpers,
    RagPipe,
    VectorStore,
    chunkText,
    write_context_relevance_to_csv,
)

textInit = """
Blackberries, part of the Rubus genus, have a rich history and are cherished for their nutritional benefits. Native to Europe, North America, and Asia, they were used medicinally by ancient Greeks, Romans, and Native Americans. These hardy plants thrive in temperate climates, and come in three main types: erect, semi-erect, and trailing. Nutrient-dense, they are high in vitamin C, vitamin K, fiber, and antioxidants, which help combat chronic diseases. Culinary uses are diverse, from fresh consumption to baking, jams, and savory sauces. Blackberries also hold cultural significance, symbolizing protection and healing, and remain a beloved summer fruit.
"""
print(f" Length of textInit: {len(textInit)}")
chunktText = chunkText(textInit, "sentence", 256, 0)


for chunk in chunktText:
    print(chunk)
    print("\n\n")

pprint(chunktText)
print(f" len of chunktText: {len(chunktText)}")
