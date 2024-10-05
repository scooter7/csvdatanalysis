from langchain_experimental.agents import create_csv_agent  # Updated import
from openai import OpenAI
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
        # Create the CSV agent
        agent = create_csv_agent(
            llm, 
            csv_file,
            verbose=False # Set True if you want to see detailed logs
        )

        # Input field for user question
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            try:
                with st.spinner(text="In progress..."):
                    # Get the answer from the agent
                    answer = agent.run(user_question)
                    st.write("✔️ " + answer)
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
