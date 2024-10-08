import os
import openai
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API Key not found. Make sure it's set in the .env file.")

# Step 1: Define Google Drive PDF URLs
pdf_urls = [
    "https://drive.google.com/uc?export=download&id=1Eapn6DZe8XEw17Ym3AmbDfqcnELNUhtr",  
    "https://drive.google.com/uc?export=download&id=1Kkfq-o_unq4izWkMR3OrorQaNc4mCaNh",  
    "https://drive.google.com/uc?export=download&id=1EiS2WaLWtOHo8pYZqao3tz1EztU-iMKD"  
]

# Step 2: Download PDFs from Google Drive
pdf_folder_path = './pdfs/'
os.makedirs(pdf_folder_path, exist_ok=True)

for pdf_url in pdf_urls:
    file_id = pdf_url.split('=')[-1]
    pdf_filename = os.path.join(pdf_folder_path, f"{file_id}.pdf")
    
    print(f"Downloading {pdf_filename} from {pdf_url}")
    
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_filename, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download {pdf_url}")

# Step 3: Load downloaded PDF documents
pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in the folder: {pdf_folder_path}")

documents = []
for pdf_file in pdf_files:
    print(f"Loading {pdf_file}")
    pdf_loader = PyPDFLoader(pdf_file)  # Load each PDF
    documents.extend(pdf_loader.load())  # Add documents to the list

# Step 4: Split documents into chunks (helps with large documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

if not docs:
    raise ValueError("No document chunks were created after splitting.")

print(f"Loaded {len(docs)} document chunks.")

# Step 5: Initialize OpenAI embeddings with the API key
embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Generate embeddings
embeddings = []
for doc in docs:
    try:
        embedding = embedding_function.embed_documents([doc.page_content])
        embeddings.append(embedding)
    except Exception as e:
        print(f"Error generating embedding for document: {doc.page_content[:100]}... Error: {e}")

if not embeddings:
    raise ValueError("Failed to generate any embeddings for the documents.")

print(f"Generated embeddings for {len(embeddings)} document chunks.")

# Step 6: Create FAISS vector store from documents using the OpenAI embeddings
vector_store = FAISS.from_documents(docs, embedding_function)

# Step 7: Define a query function and search the vector store
def retrieve_documents(query, k=3):
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    
    if not docs_and_scores:
        return "No relevant documents found."
    
    # Combine the contents of all retrieved documents into a single string (without displaying scores)
    combined_content = " ".join([doc.page_content for doc, _ in docs_and_scores])
    
    return combined_content

# Step 8: Generation using OpenAI (to create a detailed combined answer)
def generate_answer(query, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a highly skilled political and economical news analyst. "
                        "You are well-versed in analyzing parliamentary debates, government policies, and "
                        "discussions on various political and economic matters. "
                        "Your task is to provide accurate summaries and answer questions in a clear, "
                        "concise, and detailed manner, drawing on your understanding of parliamentary debates "
                        "and current events. Use the provided context to answer the user's questions related to these debates."
                    )
                },
                {"role": "user", "content": f"Question: {query} Context: {context}\n\nPlease provide a detailed and well-explained answer based on the given context."}
            ],
            max_tokens=400,
            temperature=0.7,
            top_p=0.9
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."

# Step 9: RAG function (Retrieve and Generate combined answer)
def rag(query):
    retrieved_content = retrieve_documents(query)
    
    # Limit context size to avoid hitting the token limit for GPT-3.5-turbo
    if len(retrieved_content) > 3000:
        retrieved_content = retrieved_content[:3000]
    
    # Generate a final answer using the combined context
    final_answer = generate_answer(query, retrieved_content)
    
    return final_answer

# Example usage of RAG function
if __name__ == "__main__":
    query = "On 25 september during oral questions session what questions were asked about economy?"
    answer = rag(query)
    print(f"Final Answer: {answer}")
