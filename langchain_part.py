from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

connections.connect("default", host="localhost", port="19530")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = embedding_model.get_sentence_embedding_dimension()

collection_name = "company_data"
collection = Collection(name=collection_name)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def search(query, top_k=5):
    query_embedding = embedding_model.encode([query]).tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(data=query_embedding, anns_field="embedding", param=search_params, limit=top_k, output_fields=["text"])
    retrieved_texts = [result.entity.get('text') for result in results[0]]
    return retrieved_texts

def rag(query, top_k=5, max_length=50):
    retrieved_texts = search(query, top_k=top_k)
    context = " ".join(retrieved_texts)
    prompt = f"Context: {context} Query: {query} Answer:"
    response = generate_text(prompt, max_length=max_length)
    return response

query = "What information do we have about customer payments?"
answer = rag(query)
print("Generated Answer:", answer)

