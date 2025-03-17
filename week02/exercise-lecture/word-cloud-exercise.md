# **Group Coding Exercise: Exploring Text Processing and Word Importance**

## **Objective**
In this exercise, you'll work in small groups to analyze how different text-processing techniques (n-grams, stemming, lemmatization, and TF-IDF) influence the most important words in a dataset. You'll then visualize the results using word clouds and share your findings via Mentimeter.

## **Setup Instructions**
1. Install required libraries if needed:
   ```bash
   pip install nltk sklearn wordcloud matplotlib
   ```
2. Import necessary libraries:
   ```python
   import nltk
   from nltk.tokenize import word_tokenize
   from nltk.corpus import stopwords
   from nltk.stem import PorterStemmer, WordNetLemmatizer
   from sklearn.feature_extraction.text import TfidfVectorizer
   from wordcloud import WordCloud
   import matplotlib.pyplot as plt
   import numpy as np
   ```

## **Step 1: Group Assignments**
Each group will focus on a different text-processing technique.

- **Group 1:** Tokenization & stopword removal
- **Group 2:** Stemming
- **Group 3:** Lemmatization
- **Group 4:** N-grams (bigrams or trigrams)

## **Step 2: Load a Sample Dataset**
Use a small dataset like product reviews, news headlines, or tweets. Here’s an example:
```python
sample_text = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural Language Processing is amazing!",
    "I love exploring text analytics and data science.",
    "TF-IDF helps identify important words in documents."
]
```

## **Step 3: Text Processing Techniques**
### **Group 1: Tokenization & Stopword Removal**
```python
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    return ' '.join([word for word in words if word.isalnum() and word not in stop_words])

processed_text = [preprocess_text(text) for text in sample_text]
```

### **Group 2: Stemming**
```python
stemmer = PorterStemmer()

def stem_text(text):
    words = word_tokenize(text.lower())
    return ' '.join([stemmer.stem(word) for word in words if word.isalnum()])

stemmed_text = [stem_text(text) for text in sample_text]
```

### **Group 3: Lemmatization**
```python
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(word) for word in words if word.isalnum()])

lemmatized_text = [lemmatize_text(text) for text in sample_text]
```

### **Group 4: N-Grams & TF-IDF**
```python
vectorizer = TfidfVectorizer(ngram_range=(2,2))  # Change to (3,3) for trigrams
tfidf_matrix = vectorizer.fit_transform(sample_text)
feature_names = vectorizer.get_feature_names_out()
importance = np.array(tfidf_matrix.sum(axis=0)).flatten()
top_ngrams = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:10]
```

## **Step 4: Generate Word Clouds**
```python
def generate_wordcloud(text_list, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_list))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

# Example usage:
generate_wordcloud(processed_text, "Tokenization & Stopword Removal")
generate_wordcloud(stemmed_text, "Stemming")
generate_wordcloud(lemmatized_text, "Lemmatization")
```

## **Step 5: Share & Discuss in Mentimeter**
1. Save your generated word cloud as an image.
   ```python
   wordcloud.to_file("wordcloud.png")
   ```
2. Upload it to **Mentimeter** (or any collaborative platform).
3. Compare different word clouds and discuss:
   - How does preprocessing change the key words?
   - Do n-grams reveal different insights than single words?
   - What do TF-IDF scores tell us about word importance?

## **Summary**
This activity helps you understand how preprocessing affects text analysis and visualization. Each group will contribute insights based on their approach, making this an engaging and interactive learning experience!

