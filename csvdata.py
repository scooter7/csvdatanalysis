from langchain_experimental.agents import create_csv_agent  
from langchain_openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

def main():
    load_dotenv()

    # Load OpenAI API key from Streamlit secrets
    openai_api_key = st.secrets["openai_api_key"]

    llm = OpenAI(api_key=openai_api_key, temperature=0)

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 💹")

    csv_file = st.file_uploader("Upload your CSV file", type="csv")

    if csv_file is not None:
        agent = create_csv_agent(
    llm, 
    csv_file,
    verbose=False,  # Make it True if you want to see what's the model is doing behind the scenes
    allow_dangerous_code=True  # Enable the functionality to execute arbitrary code
)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            try:
                with st.spinner(text="In progress..."):
                    answer = agent.run(user_question)
                    st.write("✔️ " + answer)
            except:
                st.write("An exception occured. Please try again")

if __name__ == "__main__":
    main()
