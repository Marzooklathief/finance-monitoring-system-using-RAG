import os
import pathway as pw
import requests
import pandas as pd
import faiss  # Vector Database
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cohere  # Alternative API-based LLM

# 🔹 Load API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "GL8NLD7SAIWFX9EJ")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "P5mva9xfGj5PR1e9SiDcCoLAXWrHq3XH86VoE4rw")

# 🔹 Fetch Financial News (API Call Limit Handling)
def fetch_financial_news():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json().get("feed", [])

        data = [{"headline": item["title"], "content": item["summary"]} for item in news_data[:20]]  # Limit to 20 articles
        return pd.DataFrame(data) if data else pd.DataFrame(columns=["headline", "content"])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame(columns=["headline", "content"])

# 🔹 Convert Data to Pathway Table
news_df = fetch_financial_news()
news_data = pw.debug.table_from_pandas(news_df)

# 🔹 Select Relevant Columns
processed_data = news_data.select(title=pw.this.headline, body=pw.this.content)

# 🔹 Hugging Face Embedding Model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Convert Text to Vectors (Batch Optimized)
news_vectors = []
vector_dim = 384  # Dimension of MiniLM embeddings

if not news_df.empty:
    news_texts = (news_df["headline"] + " " + news_df["content"]).tolist()
    
    batch_size = 5  # Lower batch size to stay within API limits
    for i in range(0, len(news_texts), batch_size):
        batch = news_texts[i : i + batch_size]
        batch_vectors = embed_model.encode(batch, convert_to_numpy=True)  # Batch processing
        news_vectors.extend(batch_vectors)

# 🔹 FAISS Indexing (Error Handling)
if news_vectors:
    vector_index = faiss.IndexFlatL2(vector_dim)
    np_vectors = np.array(news_vectors, dtype=np.float32)
    vector_index.add(np_vectors)
else:
    vector_index = faiss.IndexFlatL2(vector_dim)  # Default empty index

# 🔹 FAISS Retriever Class
class FAISSRetriever:
    def __init__(self, index, news_df):
        self.index = index
        self.news_df = news_df

    def retrieve(self, query_vector):
        if self.index.ntotal == 0:
            return []  # No data in FAISS
        D, I = self.index.search(np.array([query_vector], dtype=np.float32), k=3)  # Retrieve top 3
        return [self.news_df.iloc[i] for i in I[0] if i < len(self.news_df)]

retriever = FAISSRetriever(vector_index, news_df)

# 🔹 Query Function with Alternative LLMs (Mistral-7B or Cohere)
def query_rag(user_query, use_cohere=True):
    query_vector = embed_model.encode(user_query, convert_to_numpy=True)
    retrieved_docs = retriever.retrieve(query_vector)

    if not retrieved_docs:
        return {"query": user_query, "response": "No relevant news found."}

    # Format retrieved docs
    context = "\n\n".join([f"Title: {doc['headline']}\nContent: {doc['content']}" for doc in retrieved_docs])
    
    prompt = f"Based on the following financial news articles, answer the query: {user_query}\n\n{context}"

    if use_cohere:
        # 🔹 Cohere API
        co = cohere.Client(COHERE_API_KEY)
        response = co.generate(prompt=prompt, model="command-r-plus")
        return {"query": user_query, "response": response.generations[0].text}
    else:
        # 🔹 Mistral-7B (Local)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct", torch_dtype=torch.float16).to("cuda")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = model.generate(input_ids, max_length=512)
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return {"query": user_query, "response": response_text}

# 🔹 Run Pathway Computation
pw.run()
