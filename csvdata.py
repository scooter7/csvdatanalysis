import pandas as pd
import openai
import streamlit as st

# Function to chunk large text into smaller pieces
def chunk_text(text, max_tokens):
    """Splits text into chunks, ensuring each chunk is within the token limit."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_tokens:
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

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

        # Convert DataFrame into a string format for better retrieval
        csv_text = df.to_string()

        # Define token limit (to be on the safe side, we use a limit lower than the max context size)
        max_tokens = 8000  # Adjust according to model's context length

        # Chunk the CSV text
        chunks = chunk_text(csv_text, max_tokens)

        # Input field for user question
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            try:
                with st.spinner(text="In progress..."):
                    aggregated_response = ""
                    # Process each chunk separately, then aggregate results
                    for chunk in chunks:
                        response = openai.chat.completions.create(
                            model="gpt-4o-mini",  # The chat model you're using
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Analyze this chunk of CSV data."},
                                {"role": "user", "content": f"Here is a portion of the CSV data:\n{chunk}"}
                            ],
                            max_tokens=100
                        )
                        # Collect all chunks of data for aggregation
                        aggregated_response += response.choices[0].message.content.strip() + " "

                    # Once all chunks have been processed, we ask for a final concise answer based on aggregated data
                    final_response = openai.chat.completions.create(
                        model="gpt-4o-mini",  # The chat model you're using
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Provide a concise answer based on the aggregated data."},
                            {"role": "user", "content": f"Here is the aggregated data from the CSV:\n{aggregated_response}\n\nNow, please answer this question concisely: {user_question}"}
                        ],
                        max_tokens=100
                    )
                    # Extract the final answer
                    final_answer = final_response.choices[0].message.content.strip()

                    # Output the concise final answer
                    st.write("✔️ " + final_answer.strip())
            except Exception as e:
                st.write(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
