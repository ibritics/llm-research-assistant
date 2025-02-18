import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import os

# Set OpenAI API Key


# Function to create config.toml dynamically
def create_streamlit_config():
    # Define the config path inside the app's directory
    config_dir = os.path.join(os.getcwd(), ".streamlit")
    config_path = os.path.join(config_dir, "config.toml")

    # Ensure the .streamlit directory exists
    os.makedirs(config_dir, exist_ok=True)

    # TOML content for Streamlit theme
    config_content = """ 
    [theme]
    base="dark"
    primaryColor="#1DB954"
    backgroundColor="#121212"
    secondaryBackgroundColor="#1E1E1E"
    textColor="#FFFFFF"
    font="monospace"
    """

    # Write the config file
    with open(config_path, "w") as config_file:
        config_file.write(config_content)

# Create the config before running the app
create_streamlit_config()

# Streamlit App Code
def main():
    st.header('🌿 AI agent for Doctoral Students: Chat with Academic Papers 💬')
    st.sidebar.title('📚 LLM ChatApp using LangChain & OpenAI')
    key = st.text_input("Insert your OpenAI key (This app is deployed on Streamlit, ready privacy conditions before sharing your personal Key)")
    os.environ['OPENAI_API_KEY'] = key
    st.sidebar.markdown('''
    🚀 To connect with me: 
    - 🟢 [Linkedin](https://www.linkedin.com/in/ibritics/)
    - 🔗 [ResearchGate](https://www.researchgate.net/profile/Ibrahim-Israfilov)
    ''')
    pdf = st.file_uploader("Upload your Paper (PDF)", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(  # creating chunks with smaller sizes for use with OpenAI
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text) #Alternative solution here https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/
        # st.write(chunks[0])
        store_name = pdf.name[:-4]
        st.write(store_name)
# Checking if the name exists in order not to overwrite it. and importing with faiss.
        if os.path.exists(f"{store_name}.faiss"):
            VectorStore = faiss.read_index(f"{store_name}.faiss")
            st.write('Embeddings Loaded from FAISS file')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            VectorStore.save_local(f"{store_name}")
            st.write('Embeddings Created and Saved')
        query = st.text_input("Ask Question from your PDF File")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
#
# #### Getting digestible chunks
#
# #### Splitting to chunks
# chunks = text_splitter.split_text(text=text) # What is the difference between split_text and split_documents?
# #chunks[0]
# embeddings = OpenAIEmbeddings()
#
#
# ###Attention: Each embedding costs me money on OpenAI, therefore make sure to not to load again the same file by storing the file name.
# store_name = pdf.name[.-4] #extracting the name of file
#
# ####
#
# VectorStore = FAISS.from_texts(chunks,embeddings)
#
# query = "What is the actuality, research proposal and challenges in this research?"
#
# docs = VectorStore.similarity_search(query=query, k=3)
# llm = OpenAI()
#
# chain = load_qa_chain(llm = llm, chain_type='stuff')
# response = chain.run(input_documents=docs, question=query)
#
# save_faiss_index(vector_store, index_file, metadata_file)
#
# query = "What is the actuality, research proposal and challenges in this research?"
# docs = VectorStore.similarity_search(query)
#
#
# if os.path.exists(index_file) and os.path.exists(metadata_file):
# # Load FAISS safely
# vector_store = load_faiss_index(index_file, metadata_file, embeddings)
# st.write('Embeddings Loaded from Disk')
# else:
# # Create FAISS embeddings and save
# vector_store = FAISS.from_texts(chunks, embeddings)
# save_faiss_index(vector_store, index_file, metadata_file)
# st.write('Embeddings Created and Saved')
#
# query = st.text_input("Ask a question from your PDF file")
# if query:
# docs = vector_store.similarity_search(query=query, k=3)
# llm = OpenAI()
# chain = load_qa_chain(llm=llm, chain_type='stuff')
#
# with get_openai_callback() as cb:
#     response = chain.run(input_documents=docs, question=query)
#     print(cb)
# print(response)
#
#
# #######