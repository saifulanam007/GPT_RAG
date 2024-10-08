import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load documents (multiple PDFs from a folder)
pdf_folder_path = r'C:\Users\Lenovo\Downloads\hocdedebate\*'  # Path to your folder containing multiple PDFs
pdf_files = glob.glob(pdf_folder_path + "*.pdf")  # Load all PDF files from the folder

documents = []
for pdf_file in pdf_files:
    pdf_loader = PyPDFLoader(pdf_file)  # Load each PDF
    documents.extend(pdf_loader.load())  # Add documents to the list

# Step 2: Split documents into chunks (helps with large documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Step 3: Initialize OpenAI embeddings with the API key
embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Step 4: Create FAISS vector store from documents using the OpenAI embeddings
vector_store = FAISS.from_documents(docs, embedding_function)

# Step 5: Define a query function and search the vector store
def retrieve_documents(query, k=3):
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    
    # Combine the contents of all retrieved documents into a single string (without displaying scores)
    combined_content = " ".join([doc.page_content for doc, _ in docs_and_scores])
    
    return combined_content

# Step 6: Generation using OpenAI (to create a detailed combined answer)
def generate_answer(query, context):
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

# Step 7: RAG function (Retrieve and Generate combined answer)
def rag(query):
    retrieved_content = retrieve_documents(query)
    
    # Limit context size to avoid hitting the token limit for GPT-3.5-turbo
    if len(retrieved_content) > 3000:
        retrieved_content = retrieved_content[:3000]
    
    # Generate a final answer using the combined context
    final_answer = generate_answer(query, retrieved_content)
    
    return final_answer
