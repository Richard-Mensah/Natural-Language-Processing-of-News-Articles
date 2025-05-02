# -------------------- Import Required Libraries --------------------
# These libraries help with text processing, visualization, and data analysis.
import nltk
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from anytree import Node, RenderTree
import string

# -------------------- Download NLTK Resources --------------------
# These are necessary tools from the NLTK library for tokenization, stopword filtering, and part-of-speech tagging.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# -------------------- Load and Save Corpus --------------------
# I am combining two text files from the NLTK ABC corpus: one on rural topics and one on science.
# I save this combined text into a single file for reference or reuse.
corpus = nltk.corpus.abc.raw(['rural.txt', 'science.txt'])
with open("ScienceRule.txt", "w", encoding="utf-8") as file:
    file.write(corpus)

# -------------------- Text Preprocessing Function --------------------
def preprocess_text(text):
    """
    This function prepares the raw text for analysis by:
    - Tokenizing the text into words.
    - Converting all words to lowercase for uniformity.
    - Removing punctuation and non-alphabetic characters.
    - Removing English stopwords (e.g., 'the', 'and', 'is') to keep only meaningful words.
    
    This process ensures that we focus only on relevant words for frequency analysis and POS tagging.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# -------------------- Word Frequency Analysis Function --------------------
def analyze_text(tokens):
    """
    This function counts how many times each word appears in the text using a Counter.
    The result helps identify the most common or important terms in the document.
    """
    return Counter(tokens)

# -------------------- POS Tagging Function --------------------
def pos_tagging(tokens):
    """
    This function assigns a part-of-speech (POS) tag to each word (e.g., noun, verb, adjective).
    POS tagging helps us understand the grammatical role of each word and perform deeper linguistic analysis.
    """
    return nltk.pos_tag(tokens)

# -------------------- Preprocess and Analyze Text --------------------
tokens = preprocess_text(corpus)
frequency = analyze_text(tokens)
pos_tags = pos_tagging(tokens)

# -------------------- Display Top 20 Most Frequent Words --------------------
print("\n🔠 Top 20 Most Common Words:")
for word, count in frequency.most_common(20):
    print(f"{word}: {count}")

# -------------------- Create DataFrame from Frequency Data --------------------
# This DataFrame helps structure the top 20 words with their frequency and grammatical roles (POS).
df = pd.DataFrame(frequency.most_common(20), columns=['Word', 'Frequency'])
pos_dict = dict(pos_tags)
df['POS'] = df['Word'].map(pos_dict)
df = df.sort_values('Frequency', ascending=False)

print("\n📊 Word Frequency and POS DataFrame:")
print(df)

# -------------------- Build Tree View for Word Groups --------------------
# Here I use the 'anytree' library to create a tree structure of words grouped by POS.
# It shows the hierarchy: Root -> POS category -> individual words with frequency.
grouped = df.groupby('POS')
root = Node("Word Frequency Tree")

for pos, group in grouped:
    pos_node = Node(pos, parent=root)
    for _, row in group.iterrows():
        Node(f"{row['Word']} ({row['Frequency']})", parent=pos_node)

print("\n🌳 Word Frequency Tree Structure:")
for pre, fill, node in RenderTree(root):
    print(f"{pre}{node.name}")

# -------------------- Visualizing Word Frequencies --------------------
# I use different types of charts to visualize the top 20 most frequent words:
# - Bar chart
# - Line chart
# - Word cloud
# - Horizontal bar chart
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

# -------------------- Save Word Frequency Data --------------------
# Save the top 20 words with POS tags to a CSV file.
df.to_csv('word_freq_with_pos.csv', index=False)

# -------------------- Analyze Nouns and Verbs --------------------
# I separate the top nouns and verbs using POS tags to understand the types of words used.
noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

nouns = [word for word, tag in pos_tags if tag in noun_tags]
verbs = [word for word, tag in pos_tags if tag in verb_tags]

df_nouns = pd.DataFrame(Counter(nouns).most_common(10), columns=['Noun', 'Frequency'])
df_verbs = pd.DataFrame(Counter(verbs).most_common(10), columns=['Verb', 'Frequency'])

print("\n🔤 Top 10 Nouns:")
print(df_nouns)

print("\n🔤 Top 10 Verbs:")
print(df_verbs)

# Save nouns and verbs into separate CSV files for further analysis
df_nouns.to_csv("top_10_nouns.csv", index=False)
df_verbs.to_csv("top_10_verbs.csv", index=False)

# -------------------- Visualize Nouns and Verbs --------------------
# Two bar charts for comparing the top 10 nouns and verbs in the text.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Verbs Bar Chart
axes[0].bar(df_verbs['Verb'], df_verbs['Frequency'], color='orange')
axes[0].set_title('Top 10 Verbs')
axes[0].set_xlabel('Verb')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='x', rotation=45)

# Nouns Bar Chart
axes[1].bar(df_nouns['Noun'], df_nouns['Frequency'], color='green')
axes[1].set_title('Top 10 Nouns')
axes[1].set_xlabel('Noun')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='x', rotation=45)

# Overall title and layout
plt.suptitle('POS Frequency Comparison', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
