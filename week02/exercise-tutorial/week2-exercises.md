# Week 2: Stemming, lemmatization, and word cloud exercise

## **Objective**

In this exercise, you will preprocess textual data by applying stemming and lemmatization. You will then visualize word frequencies using a word cloud.

## **Instructions**

### **1. Load the dataset**

Use the dataset from Week 1 (`articles.tar.gz` or `articles.zip`). Extract the files and load them into Python using the code below:

```python
# Define dataset path and the source you want to read from
dataset_path = 'articles/'
source_name = 'Vox'  # Change to desired source if needed, like BBC or The Guardian

# Correct the glob pattern to find files in the specified source folder across all dates
newspaperfiles = glob(os.path.join(dataset_path, f'*/{source_name}/*'))

# Initialize a list to hold documents
documents = []

# Read files and handle encoding errors if necessary
for filename in newspaperfiles:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    except Exception as e:
        print(f"Error reading {filename}: {e}")

print(f"Loaded {len(documents)} articles from {source_name}.")
```

---

### **2. Apply stemming and lemmatization**

- Implement stemming using `nltk.PorterStemmer()`.
- Implement lemmatization using `nltk.WordNetLemmatizer()`.
- Compare the outputs of stemming and lemmatization.

**Example:**

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

text_tokens = word_tokenize(text_sample.lower())
stemmed_words = [stemmer.stem(word) for word in text_tokens]
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in text_tokens]

print("Stemmed words:", stemmed_words[:20])
print("Lemmatized words:", lemmatized_words[:20])
```

---

### **3. Generate a Word Cloud**

- After processing the text, generate a word cloud to visualize frequent words.
- Use the `WordCloud` library.

**Example:**

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text_processed = " ".join(lemmatized_words)  # Use lemmatized words for better readability
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_processed)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

---

### **4. Analysis and Discussion**

- How does stemming affect the words in your dataset?
- How does lemmatization compare to stemming in terms of readability?
- How do different preprocessing choices affect the word cloud visualization?
- How do word clouds differ across outlets?


---

### **Hints & resources**

- [NLTK Documentation](https://www.nltk.org/)
- [WordCloud Library](https://github.com/amueller/word_cloud)

