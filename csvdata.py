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
                    # Convert the full DataFrame into a string for context
                    csv_text = df.to_string()

                    # Send the entire dataset and question to OpenAI in a single request
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",  # The chat model you're using
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Provide only concise and clear answers based on the full dataset provided."},
                            {"role": "user", "content": f"Here is the full CSV data:\n{csv_text}\n\nPlease answer this question concisely: {user_question}"}
                        ],
                        max_tokens=100  # Adjust token limit as needed
                    )
                    # Extract the final answer
                    final_answer = response.choices[0].message.content.strip()

                    # Output the concise final answer
                    st.write("✔️ " + final_answer.strip())
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
