# Tutorial exercises week 1: Working with textual data

# Instructions

In this assignment, you will work with textual data, focusing on dataset structure, processing, and basic analysis techniques. You will inspect the dataset, discuss research questions, and implement fundamental preprocessing steps such as tokenization, stopword removal, and stemming.

## Questions 

### 1. Get the Data

Download `articles.tar.gz` or `articles.zip` from Canvas (under Week 1). Unpack the dataset and inspect the contents.

Hint: On Windows, you can use built-in extraction tools or `tar -xvzf articles.tar.gz `on macOS/Linux.

###  2. Inspect the structure of the dataset

What information do the following elements provide about the dataset?
-   Folder (directory) names
-   Folder structure/hierarchy
-   File names
-   File contents

How can you programmatically inspect these aspects of the dataset?

*Hint*: Consider using `os.listdir()` to check the folder contents and glob for pattern-based file selection.

### 3. Discuss strategies for working with this dataset

Considering the dataset's size and structure:

-   What research questions could you explore using this dataset?
-   What strategies could you use to process and analyze this dataset efficiently?


### 4. Read some (or all) data

Load a sample of the dataset and display the first few lines of text.

How would you handle reading a large number of files efficiently?

*Hint*: The `glob` module can help find files:

```python
dataset_path = 'articles'

folders = os.listdir(dataset_path)
print("Folders:", folders)

# Check the contents of each folder to get an overview.
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        print(f"Folder: {folder}")
        print("Contents:", os.listdir(folder_path))
```

### 5. Tokenization

-   What is tokenization, and why is it useful in text processing?
-   Implement a basic tokenization process using Python.

*Hint*: You can use `nltk.word_tokenize()` or Python's built-in `split()` method.

### 6. Stopword removal

-   Why is it beneficial to remove stopwords in text analysis?
-   Implement a method to remove stopwords from a tokenized text sample.

*Hint*: NLTK provides a built-in list of stopwords in multiple languages.
