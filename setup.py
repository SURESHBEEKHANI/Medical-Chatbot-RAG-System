# Importing necessary tools from the "setuptools" package to help create and manage a Python project
from setuptools import find_packages, setup  # type: ignore
setup(
    name='Medical-Chatbot-RAG-System',  # Giving the project a name
    version='0.0.0',  # Setting the version of the project (like a version number)
    author='SURESH BEEKHANI',  # The name of the person who created the project
    author_email='sureshbeekhani26@gmail.com',  # The email address of the author (so people can contact them)
    packages=find_packages(),  # Finding all the parts of the project (files and folders)
    install_requires=[]  # A list that tells which additional tools are needed (empty for now)
)

