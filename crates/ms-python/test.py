import memory_search
import json

print("Initializing MemorySearch engine...")
ms = memory_search.MemorySearch()

print("\n--- Indexing ---")
# Index our README or source code
result = ms.index("../ms-core/src", mode="codebase")
print(result)

print("\n--- Searching ---")
# Run a 3-arm search
json_str = ms.search(
    query="chunker configuration overlap",
    mode="codebase",
    top_k=3,
    rerank=True,
    rag=False
)

results = json.loads(json_str)

for i, res in enumerate(results):
    print(f"\nResult #{i + 1} (Score: {res['score']:.4f})")
    print(f"Source: {res['metadata']['source']}")
    print(f"Text: {res['text'].strip()}")
