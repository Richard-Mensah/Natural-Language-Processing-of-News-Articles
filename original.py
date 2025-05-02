# -------------------- Import Libraries --------------------
import nltk
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from anytree import Node, RenderTree
import string

# -------------------- Download NLTK Resources --------------------
nltk.download('punkt')  # Tokenizer
nltk.download('stopwords')  # Stopwords for filtering
nltk.download('averaged_perceptron_tagger')  # POS tagger

# -------------------- Load and Save Corpus --------------------
corpus = nltk.corpus.abc.raw(['rural.txt', 'science.txt'])

# Save the combined corpus to a text file
with open("ScienceRule.txt", "w", encoding="utf-8") as file:
    file.write(corpus)


# -------------------- Text Preprocessing Function --------------------
def preprocess_text(text):
     """
    Cleans and tokenizes the raw input text by:
    - Splitting it into words
    - Lowercasing all words
    - Removing punctuation and stopwords
    - Keeping only alphabetic words

    This step prepares the text for further analysis by reducing noise.

    Args:
        text (str): The raw combined text from rural and science articles.

    Returns:
        list: A list of clean, lowercase, meaningful words.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# -------------------- Analyze Word Frequency --------------------
def analyze_text(tokens):
    """
    Counts how often each word appears in the tokenized corpus.

    Helps identify the most frequently discussed topics.

    Args:
        tokens (list): Preprocessed list of tokens.

    Returns:
        Counter: Dictionary-like object with word counts.
    """
    return Counter(tokens)

# -------------------- POS Tagging --------------------
def pos_tagging(tokens):
     """
    Assigns a part-of-speech (POS) tag to each word such as noun, verb, adjective.

    Useful for deeper grammatical or semantic analysis.

    Args:
        tokens (list): Preprocessed word tokens.

    Returns:
        list: List of tuples where each tuple is (word, POS tag).
    """
    return nltk.pos_tag(tokens)

# -------------------- Preprocess and Analyze Text --------------------
tokens = preprocess_text(corpus)
frequency = analyze_text(tokens)
pos_tags = pos_tagging(tokens)

# -------------------- Display Top 20 Word Frequencies --------------------
print("\n🔠 Top 20 Most Common Words:")
for word, count in frequency.most_common(20):
    print(f"{word}: {count}")

# -------------------- Create DataFrame --------------------
df = pd.DataFrame(frequency.most_common(20), columns=['Word', 'Frequency'])

# Map POS tags to top words
pos_dict = dict(pos_tags)
df['POS'] = df['Word'].map(pos_dict)
df = df.sort_values('Frequency', ascending=False)

print("\n📊 Word Frequency and POS DataFrame:")
print(df)

# -------------------- Tree Visualization --------------------
grouped = df.groupby('POS')
root = Node("Word Frequency Tree")

# Build tree structure
for pos, group in grouped:
    pos_node = Node(pos, parent=root)
    for _, row in group.iterrows():
        Node(f"{row['Word']} ({row['Frequency']})", parent=pos_node)

# Display tree
print("\n🌳 Word Frequency Tree Structure:")
for pre, fill, node in RenderTree(root):
    print(f"{pre}{node.name}")

# -------------------- Visualizations --------------------
plt.figure(figsize=(15, 10))

# Bar Chart
plt.subplot(2, 2, 1)
plt.bar(df['Word'], df['Frequency'], color='skyblue')
plt.title('Top 20 Words - Bar Chart')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

# Line Graph
plt.subplot(2, 2, 2)
plt.plot(df['Word'], df['Frequency'], marker='o', color='orange')
plt.title('Top 20 Words - Line Graph')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(True)

# Word Cloud
plt.subplot(2, 2, 3)
wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(dict(frequency.most_common(20)))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud')
plt.axis('off')

# Horizontal Bar Chart
plt.subplot(2, 2, 4)
plt.barh(df['Word'], df['Frequency'], color='green')
plt.title('Top 20 Words - Horizontal Bar')
plt.xlabel('Frequency')
plt.ylabel('Words')

plt.tight_layout()
plt.show()

# -------------------- Save Data --------------------
df.to_csv('word_frequencies_with_pos.csv', index=False)

# -------------------- Noun and Verb Analysis --------------------
noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

# Extract top nouns and verbs
nouns = [word for word, tag in pos_tags if tag in noun_tags]
verbs = [word for word, tag in pos_tags if tag in verb_tags]

df_nouns = pd.DataFrame(Counter(nouns).most_common(10), columns=['Noun', 'Frequency'])
df_verbs = pd.DataFrame(Counter(verbs).most_common(10), columns=['Verb', 'Frequency'])

print("\n🔤 Top 10 Nouns:")
print(df_nouns)

print("\n🔤 Top 10 Verbs:")
print(df_verbs)

# Save to CSV
df_nouns.to_csv("top_10_nouns.csv", index=False)
df_verbs.to_csv("top_10_verbs.csv", index=False)

# -------------------- POS Subplots --------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Verbs Plot
axes[0].bar(df_verbs['Verb'], df_verbs['Frequency'], color='orange')
axes[0].set_title('Top 10 Verbs')
axes[0].set_xlabel('Verb')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='x', rotation=45)

# Nouns Plot
axes[1].bar(df_nouns['Noun'], df_nouns['Frequency'], color='green')
axes[1].set_title('Top 10 Nouns')
axes[1].set_xlabel('Noun')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Parts of Speech Frequency Comparison', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

