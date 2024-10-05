import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

def main():
    # Load the OpenAI API key from secrets
    api_key = st.secrets["openai"]["api_key"]

    # Initialize OpenAI with the api_key
    llm = OpenAI(temperature=0, api_key=api_key, model="gpt-4o-mini")

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 💹")

    # File uploader for CSV
    csv_file = st.file_uploader("Upload your CSV file", type="csv")

    if csv_file is not None:
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Convert DataFrame into a string format for better retrieval
        csv_text = df.to_string()

        # Initialize the FAISS vector store
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_texts([csv_text], embeddings)

        # Create the Conversational Retrieval Chain
        retriever = vector_store.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

        # Input field for user question
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            try:
                with st.spinner(text="In progress..."):
                    # Get the answer from the chain
                    answer = qa_chain.run(user_question)
                    st.write("✔️ " + answer)
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
