# Import necessary functions for handling PDFs, splitting text, and downloading embeddings
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

# Import Pinecone gRPC client and serverless specifications
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
# Import Pinecone vectorstore integration from LangChain
# Embed each chunk and upsert the embeddings into your Pinecone index.
from langchain.vectorstores import Pinecone
# Import the library to load environment variables from a `.env` file
from dotenv import load_dotenv

# Import the OS library for accessing environment variables
import os

# Load environment variables from a `.env` file
load_dotenv()

# Retrieve the Pinecone API key from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# Explicitly set the Pinecone API key in the environment (useful for Pinecone initialization)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load data from a PDF file located in the 'Data/' directory
extracted_data = load_pdf_file(data='Data/')

# Split the extracted text into smaller chunks for embedding
text_chunks = text_split(extracted_data)

# Download pre-trained embeddings from Hugging Face
embeddings = download_hugging_face_embeddings()

# Initialize the Pinecone client using the API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the name of the Pinecone index to be created
index_name = "medicalbot"

# Create a new Pinecone index for storing embeddings
pc.create_index(
    name=index_name,       # Name of the index
    dimension=384,         # Dimensionality of the embeddings
    metric="cosine",       # Similarity metric to be used (cosine similarity)
    spec=ServerlessSpec(   # Specify serverless deployment configuration
        cloud="aws",       # Cloud provider (AWS)
        region="us-east-1" # Region for the serverless deployment
    ) 
)

# Embed the text chunks and upsert the resulting embeddings into the Pinecone index
docsearch = Pinecone(
    documents=text_chunks,  # List of text chunks to be embedded
    index_name=index_name,  # Name of the index to store embeddings
    embedding=embeddings,   # Embedding function to generate embeddings
)
