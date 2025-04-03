import streamlit as st
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Set the page layout to wide
st.set_page_config(page_title="Educational Text Analysis Demo App", layout="wide")

# App Title and Introduction
st.title("Text Analysis Demo")
st.write(
    """
Welcome! This app demonstrates key text analysis techniques from this week in an interactive way.
"""
)

# Input text area for analysis (initial text is empty)
text_input = st.text_area("Enter your text:", "")

# Choose the operation with a placeholder option so nothing is selected by default.
options = ["Show Tokens", "Count Vectorize", "TF-IDF Vectorize"]
operation = st.radio("Select an operation:", options, index=None)

# Create two columns: left for interactive output, right for explanations.
col_left, col_right = st.columns(2)

# Check if text is provided; if not, display a message in the left column.
if text_input.strip() == "":
    with col_left:
        st.warning(
            "Please enter some text and select an operation above to see the results."
        )
else:
    if operation == "Show Tokens":
        with col_left:
            st.subheader("Output: Tokens")
            tokenizer = TreebankWordTokenizer()
            tokens = tokenizer.tokenize(text_input)
            st.write(tokens)

        with col_right:
            st.subheader("Explanation: Tokenization")
            st.write(
                """
            **Tokenization** is the process of splitting text into individual words or tokens.
            - **Why it matters:** It converts raw text into manageable pieces for further analysis.
            - **Example:** The NLTK `TreebankWordTokenizer` splits text into tokens while also handling punctuation.
            """
            )

    elif operation == "Count Vectorize":
        with col_left:
            st.subheader("Output: Count Vectorization")
            cv = CountVectorizer()
            docs = [text_input]  # Treat the input as one document.
            X = cv.fit_transform(docs)
            df_cv = pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out())
            st.dataframe(df_cv)

        with col_right:
            st.subheader("Explanation: Count Vectorization")
            st.write(
                """
            **Count Vectorization** converts text into a numerical format:
            - **How it works:** Each unique word becomes a feature (a column) in a document-term matrix.
            - **What we see:** The table displays the frequency (count) of each word in our text.
            - **Use case:** It provides a simple representation of text for tasks like classification or clustering.
            """
            )

    elif operation == "TF-IDF Vectorize":
        with col_left:
            st.subheader("Output: TF-IDF Vectorization")
            tfidf = TfidfVectorizer()
            docs = [text_input]
            X_tfidf = tfidf.fit_transform(docs)
            df_tfidf = pd.DataFrame(
                X_tfidf.toarray(), columns=tfidf.get_feature_names_out()
            )
            st.dataframe(df_tfidf)

        with col_right:
            st.subheader("Explanation: TF-IDF Vectorization")
            st.write(
                """
            **TF-IDF (Term Frequency-Inverse Document Frequency)** refines raw counts by weighing word importance:
            - **Term Frequency (TF):** Measures how often a word appears in the text.
            - **Inverse Document Frequency (IDF):** Reduces the weight of words that appear across many documents.
            - **Result:** A matrix where higher values indicate words that are both frequent in the document and unique across documents.
            - **Why it helps:** It highlights distinctive words, making it very useful in text analysis.
            """
            )

# Final Summary Section across the full page width
st.header("Summary")
st.success(
    """
In this demo, we learned how to:
- **Tokenize** text into individual tokens.
- **Vectorize** text using:
  - **Count Vectorization** for raw word counts.
  - **TF-IDF Vectorization** for weighted word importance.

We can experiment with different texts and see how the interactive outputs change. These foundational techniques are essential for many natural language processing tasks.
"""
)
