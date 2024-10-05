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
                    if "travel rarely" in user_question.lower() and "column c" in user_question.lower():
                        if "C" in df.columns:
                            count_travel_rarely = df['C'].value_counts().get("Travel_Rarely", 0)
                            result = f"There are {count_travel_rarely} people who travel rarely."
                        else:
                            result = "Column 'C' does not exist in the dataset."

                    # More conditions can be added here for different types of analyses based on the question
                    else:
                        result = "I couldn't identify what to analyze from the question. Try asking something like 'How many people travel rarely in column C?'"

                    # Send the result to OpenAI for summarization or further context
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",  # The chat model you're using
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Provide a concise and insightful response based on the data analysis provided."},
                            {"role": "user", "content": f"Here is the data analysis result:\n{result}\n\nPlease provide any additional insights based on this analysis."}
                        ],
                        max_tokens=100
                    )

                    # Extract the final answer
                    final_answer = response.choices[0].message.content.strip()

                    # Output the concise final answer
                    st.write("✔️ " + final_answer.strip())
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
