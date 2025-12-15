from huggingface_hub import hf_hub_download
import onnxruntime as ort
from transformers import AutoTokenizer

# Download from the 🤗 Hub
model_id = "onnx-community/embeddinggemma-300m-ONNX"
model_path = hf_hub_download(model_id, subfolder="onnx", filename="model.onnx") # Download graph
hf_hub_download(model_id, subfolder="onnx", filename="model.onnx_data") # Download weights
session = ort.InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Run inference with queries and documents
prefixes = {
  "query": "task: search result | query: ",
  "document": "title: none | text: ",
}
query = prefixes["query"] + "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
documents = [prefixes["document"] + x for x in documents]

inputs = tokenizer([query] + documents, padding=True, return_tensors="np")

_, sentence_embedding = session.run(None, inputs.data)
print(sentence_embedding.shape)  # (5, 768)

# Compute similarities to determine a ranking
query_embeddings = sentence_embedding[0]
document_embeddings = sentence_embedding[1:]
similarities = query_embeddings @ document_embeddings.T
print(similarities)  # [0.30109745 0.635883 0.49304956 0.48887485]

# Convert similarities to a ranking
ranking = similarities.argsort()[::-1]
print(ranking)  # [1 2 3 0]
