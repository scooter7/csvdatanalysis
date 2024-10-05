import pandas as pd
import openai
import streamlit as st

def analyze_data_with_pandas(df, user_question):
    """
    Use Pandas to dynamically analyze the CSV based on the user's natural language question.
    This is where you can extend the logic to handle different types of queries.
    """
    result = ""

    # Example: Handle simple analysis of specific columns like attrition
    if "attrition" in user_question.lower():
        if "Attrition" in df.columns and "Department" in df.columns:
            attrition_data = df[df['Attrition'] == "Yes"]
            attrition_count = attrition_data['Department'].value_counts()
            result = f"Attrition by Department:\n{attrition_count.to_string()}"
        else:
            result = "The dataset does not contain the required 'Attrition' or 'Department' columns."
    
    # Example: Handle summary statistics for the entire dataset
    elif "summary" in user_question.lower() or "statistics" in user_question.lower():
        result = df.describe().to_string()
    
    # You can extend this with more conditions based on the question.
    else:
        result = "I couldn't understand your question. Try asking something like 'What is the attrition by department?' or 'Can you give me a summary of the data?'"

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
                    # Step 1: Analyze the data with Pandas based on the user question
                    pandas_result = analyze_data_with_pandas(df, user_question)

                    # Step 2: Use OpenAI to provide a conversational response based on Pandas result
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",  # The chat model you're using
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Use the following data analysis result to answer the user's question in a conversational tone."},
                            {"role": "user", "content": f"Here is the analysis result:\n{pandas_result}\n\nPlease provide this result in a conversational style."}
                        ],
                        max_tokens=100
                    )

                    # Extract the final answer
                    final_answer = response.choices[0].message['content'].strip()

                    # Output the conversational final answer
                    st.write("✔️ " + final_answer)
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
