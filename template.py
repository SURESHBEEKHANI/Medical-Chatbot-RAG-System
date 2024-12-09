# Import necessary modules for working with file paths and logging
import os  # For interacting with the operating system (e.g., checking if files/directories exist)
from pathlib import Path  # For handling file paths in a cross-platform way
import logging  # For logging messages during the execution of the script

# Configure logging to display messages with a timestamp and message content
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of file paths to be created if they do not already exist
list_of_files = [
    "src/__init__.py",  # A Python file typically used to mark a directory as a package
    "src/helper.py",    # A Python script, likely for helper functions
    "src/prompt.py",    # A Python script, possibly for handling prompts
    ".env",             # A configuration file for environment variables
    "setup.py",         # A Python script for package setup
    "app.py",           # A Python script, likely the main application entry point
    "research/trials.ipynb",  # A Jupyter Notebook file for research or experimentation
    "test.py"           # A Python script for testing purposes
]

# Loop through each file path in the list
for filepath in list_of_files:
    # Convert the file path string into a Path object for easier manipulation
    filepath = Path(filepath)
    
    # Split the file path into directory and file name components
    filedir, filename = os.path.split(filepath)
    
    # If the directory part of the path is not empty (i.e., the file is inside a folder)
    if filedir != "":
        # Create the directory if it does not already exist
        os.makedirs(filedir, exist_ok=True)
        # Log a message indicating that the directory has been created
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # Check if the file does not exist or if it exists but is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        # Create an empty file by opening it in write mode and immediately closing it
        with open(filepath, "w") as f:
            pass  # 'pass' is used here to do nothing after opening the file
        # Log a message indicating that the empty file has been created
        logging.info(f"Creating empty file: {filepath}")
    else:
        # Log a message if the file already exists and is not empty
        logging.info(f"{filename} already exists")
