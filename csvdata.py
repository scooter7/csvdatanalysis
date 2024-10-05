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
                    # Generate the response from OpenAI using the chat model
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",  # The chat model you're using
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"Here is the CSV data:\n{csv_text}\n\nNow, please answer this question: {user_question}"}
                        ],
                        max_tokens=100
                    )
                    answer = response['choices'][0]['message']['content'].strip()
                    st.write("✔️ " + answer)
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
