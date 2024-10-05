import pandas as pd
import openai
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

        # Show the first 20 rows of the dataset in the UI
        st.subheader("First 20 rows of the dataset:")
        st.dataframe(df.head(20))  # Display the first 20 rows

        # Input field for user question
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            try:
                with st.spinner(text="In progress..."):
                    # Pre-process the data using Pandas based on the user's question
                    result = ""

                    # Example of handling a question: "How many people travel rarely in column C?"
                    if "travel rarely" in user_question.lower() and "column c" in user_question.lower():
                        if "C" in df.columns:
                            count_travel_rarely = df['C'].value_counts().get("Travel_Rarely", 0)
                            result = f"There are {count_travel_rarely} people who travel rarely in column C."
                        else:
                            result = "Column 'C' does not exist in the dataset."

                    # If no match, give a default response
                    if result == "":
                        result = "I'm sorry, I couldn't understand your question. Please ask something like 'How many people travel rarely in column C?'"

                    # Send the result as the final answer
                    st.write("✔️ " + result)
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
