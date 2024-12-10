# Import required libraries and modules
from flask import Flask, render_template, jsonify, request  # Import Flask for web server and rendering templates
from src.helper import download_hugging_face_embeddings  # Import the function to download HuggingFace embeddings
from langchain_community.vectorstores import Pinecone  # Updated import: Import Pinecone for vector store operations from langchain_community
from langchain_groq import ChatGroq  # Import the Groq model for AI-powered language model operations
from langchain.chains import create_retrieval_chain  # Import chain to link document retrieval and model for Q&A tasks
from langchain.chains.combine_documents import create_stuff_documents_chain  # Import document processing chain for Q&A
from langchain_core.prompts import ChatPromptTemplate  # Import class to handle prompt templates for chat
from src.prompt import *  # Import custom prompt configurations from a local module
import os  # Import os module to interact with environment variables
from dotenv import load_dotenv  # Import dotenv to load environment variables from a .env file

# Initialize Flask web application
app = Flask(__name__)  # Create a Flask app instance to handle web requests

# Load environment variables from the .env file
load_dotenv()  # Load environment variables from a .env file to manage sensitive data like API keys

# Retrieve Pinecone API key from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')  # Fetch the Pinecone API key securely from environment variables
# Retrieve Groq API key from environment variables
groq_API_KEY = os.environ.get('groq_API_KEY')  # Fetch the Groq API key securely from environment variables

# Set the API keys as environment variables for use in the app
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY  # Set Pinecone API key in environment variables for later use
os.environ["groq_API_KEY"] = groq_API_KEY  # Set Groq API key in environment variables for later use

# Download pre-trained embeddings from HuggingFace
embeddings = download_hugging_face_embeddings()  # Download pre-trained HuggingFace embeddings for document processing

# Define the name of the Pinecone index to be used
index_name = "medical-chatbot"  # Define the Pinecone index name used for storing and retrieving vector embeddings

# Initialize Pinecone vector store using an existing index and embeddings
docsearch = Pinecone.from_existing_index(
    index_name=index_name,  # Use the specified Pinecone index name
    embedding=embeddings  # Use the downloaded HuggingFace embeddings for vector search
)

# Set up the retriever to perform similarity-based search with top-k results (3 in this case)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Setup similarity search to retrieve top-3 results

# Initialize the Groq model for generating responses based on input queries
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Specify the model name for Groq (e.g., Mixtral model)
    temperature=0,  # Set temperature for response variability (0 for deterministic outputs)
    max_tokens=None,  # No limit on the number of tokens for the model's response
    timeout=None,  # No timeout for API requests
    max_retries=2,  # Maximum number of retries for the request
    # other parameters can be specified as needed
)

# Create a chat prompt template using system and human messages
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # System-level prompt setup
        ("human", "{input}"),  # Placeholder for the user's input message
    ]
)

# Create a document processing chain that uses the Groq model and defined prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)  # Chain that processes documents for Q&A using Groq model

# Create a retrieval chain that links document retrieval with the Q&A chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)  # Retrieval-Augmented Generation chain for Q&A

# Route to render the homepage (chat interface)
@app.route("/")
def index():
    return render_template('chat.html')  # Render the chat.html template for the homepage, showing the chat interface

# Route to handle incoming chat messages and generate responses
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]  # Retrieve the message from the user input form via HTTP POST request
    input = msg  # Assign the message to the input variable
    print(input)  # Print the user's input to the console for debugging purposes
    response = rag_chain.invoke({"input": msg})  # Process the input message through the retrieval and Q&A chain
    print("Response : ", response["answer"])  # Print the generated response to the console for debugging
    return str(response["answer"])  # Return the response as a string to be displayed in the chat interface

# Run the Flask web server on the specified host and port
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)  # Start the app on all network interfaces at port 8080 for development
