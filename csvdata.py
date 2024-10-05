import pandas as pd
import openai
import streamlit as st

def process_query_with_pandas(df, query):
    """
    Process user's query with Pandas.
    This function handles dynamic query processing.
    """
    result = ""
    
    # Handling some example queries dynamically
    if "travel rarely" in query.lower() and "column c" in query.lower():
        if 'C' in df.columns:
            count_travel_rarely = df['C'].value_counts().get("Travel_Rarely", 0)
            result = f"There are {count_travel_rarely} people who travel rarely in column C."
        else:
            result = "Column 'C' does not exist in the dataset."

    elif "average age" in query.lower():
        if 'Age' in df.columns:
            avg_age = df['Age'].mean()
            result = f"The average age is {avg_age:.2f}."
        else:
            result = "The dataset does not have an 'Age' column."

    else:
        result = "I couldn't understand your query. Try asking about 'Travel_Rarely' or 'average age'."

    return result

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
                    # Step 1: Process the query with Pandas
                    pandas_result = process_query_with_pandas(df, user_question)

                    # Step 2: Use OpenAI to conversationally rephrase the Pandas result
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",  # The chat model you're using
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Rephrase the following information in a conversational tone."},
                            {"role": "user", "content": f"Here is the analysis result:\n{pandas_result}"}
                        ],
                        max_tokens=100
                    )

                    # Extract the final answer
                    final_answer = response.choices[0].message['content'].strip()

                    # Output the final answer in a conversational way
                    st.write("✔️ " + final_answer.strip())
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
