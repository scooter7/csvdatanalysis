import pandas as pd
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

def main():
    # Load the OpenAI API key from secrets
    api_key = st.secrets["openai"]["api_key"]

    # Initialize OpenAI with the api_key
    openai.api_key = api_key

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

        # Input field for user question
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            try:
                with st.spinner(text="In progress..."):
                    # Generate the response from OpenAI directly
                    response = openai.chat.completions.create(
                        engine="gpt-4o-mini",  # Or whatever model you're using
                        prompt=f"Question: {user_question}\n\nBased on the following CSV data:\n{csv_text}",
                        max_tokens=100
                    )
                    answer = response.choices[0].text.strip()
                    st.write("✔️ " + answer)
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
