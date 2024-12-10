# Import necessary classes from LangChain to load PDFs, split text, and download embeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings 

# Function to extract data from PDF files
def load_pdf_file(data):
    # Initialize DirectoryLoader to load all PDF files from the specified directory
    loader = DirectoryLoader(
        data,  # The directory path containing PDF files
        glob="*.pdf",  # File pattern to match (in this case, all PDFs)
        loader_cls=PyPDFLoader  # Class responsible for loading PDF files
    )
    
    # Load the documents from the specified directory
    documents = loader.load()

    # Return the loaded documents
    return documents

# Function to split the extracted text into smaller chunks
def text_split(extracted_data):
    # Initialize the text splitter to split text into chunks of 500 characters, with a 20-character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    
    # Split the extracted data (list of documents) into smaller chunks
    text_chunks = text_splitter.split_documents(extracted_data)
    
    # Return the resulting text chunks
    return text_chunks

# Function to download pre-trained embeddings from HuggingFace
def download_hugging_face_embeddings():
    # Initialize the HuggingFace embedding model using a specific pre-trained model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  
    
    # Return the embeddings model
    return embeddings
