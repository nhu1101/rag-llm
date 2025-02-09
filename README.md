Objective of the projects: 
Input the PDF file and ask the LLM model to extract the information from that 


Main Components: 
1. Process the PDF file using RecursiveCharacterTextSplitter() from langchain library
2. Embedding the text using HuggingFaceEmbeddings()
3. Create the vector database to store the embedding vectors
4. Using LLM pre-trained model (Vicuda) to implement the ask-answer from the text
