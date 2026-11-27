
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.5-flash-lite"

DB_DIR = "chroma_db"
POLICY_FILE = "hr_Leave_Policy.txt"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_DIR)

collection = chroma_client.get_or_create_collection(
    name="hr_leave_policy",
    metadata={"hnsw:space": "cosine"}
)


# Generate embeddings
def embed(texts):
    embeddings = []

    for text in texts:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

        embeddings.append(response.embeddings[0].values)

    return embeddings


# Split policy into chunks
def chunk_policy(text):
    sections = text.split("\n#")
    chunks = []

    for index, section in enumerate(sections):
        section = section.strip()

        if section:
            if index > 0:
                section = "#" + section

            chunks.append(section)

    return chunks


# Index HR policy
def index_policy():
    if not Path(POLICY_FILE).exists():
        print("HR Leave Policy file not found.")
        return

    policy = Path(POLICY_FILE).read_text(
        encoding="utf-8"
    )

    chunks = chunk_policy(policy)

    if not chunks:
        print("No policy content found.")
        return

    embeddings = embed(chunks)

    ids = [
        f"leave_policy_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": POLICY_FILE,
            "section": f"Section {i + 1}"
        }
        for i in range(len(chunks))
    ]

    # Replace previous indexed policy
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("HR Leave Policy indexed successfully!")


# Retrieve relevant policy chunks
def retrieve(question, k=3):
    query_embedding = embed([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count())
    )

    retrieved_chunks = []

    for i, document in enumerate(results["documents"][0]):
        retrieved_chunks.append({
            "text": document,
            "source": results["metadatas"][0][i]["source"],
            "section": results["metadatas"][0][i]["section"]
        })

    return retrieved_chunks


# Generate answer with citations
def answer_question(question):
    chunks = retrieve(question)

    context = ""

    for chunk in chunks:
        context += (
            f"\n[{chunk['source']} - {chunk['section']}]\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
You are an HR Leave Policy Assistant.

Answer employee questions using ONLY the provided HR policy context.

Rules:
1. Do not invent leave rules or benefits.
2. If information is missing, clearly say it is not specified
   in the provided HR Leave Policy.
3. Give clear and direct answers.
4. Cite the source filename and section for your answer.

HR Leave Policy Context:
{context}

Employee Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    print("\nAnswer:")
    print(response.text)
    return response.text




if __name__ == "__main__":
    index_policy()

    while True:
        question = input(
            "\nAsk an HR Leave Policy question (or type exit): "
        ).strip()

        if question.lower() == "exit":
            print("Exiting HR Leave Policy Assistant.")
            break

        if not question:
            print("Please enter a question.")
            continue

        if collection.count() == 0:
            print("No policy data indexed.")
            break

        try:
            answer_question(question)
        except Exception as e:
            print("Error:", e)
