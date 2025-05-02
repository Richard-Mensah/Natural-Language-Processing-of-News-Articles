import nltk
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
from anytree import Node, RenderTree

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')  # For POS tagging

# Load the corpus
corpus = nltk.corpus.abc.raw(['rural.txt', 'science.txt'])

# Save corpus to file
with open("ScienceRule.txt", "w", encoding="utf-8") as f:
    f.write(corpus)

# Text preprocessing function
def preprocess_text(text):
    """Cleans and preprocesses the text."""
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    
    # Removing punctuation and non-alphabetic words
    tokens = [word for word in tokens if word.isalpha()]
    
    # Removing stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Text analysis function
def analyze_text(tokens):
    """Analyzes the text and returns frequency distribution."""
    return Counter(tokens)

# POS tagging function
def pos_tagging(tokens):
    """Tags parts of speech for tokens."""
    return nltk.pos_tag(tokens)

# Preprocess and analyze the text
tokens = preprocess_text(corpus)
frequency = analyze_text(tokens)
pos_tags = pos_tagging(tokens)

# Display the 20 most common words
print("\nTop 20 Most Common Words:")
for word, count in frequency.most_common(20):
    print(f"{word}: {count}")

# Create dictionary and DataFrame from frequency data
word_freq_dict = dict(frequency.most_common(20))  # Top 20 words
df = pd.DataFrame(list(word_freq_dict.items()), columns=['Word', 'Frequency'])

# Add POS tags for the top 20 words
pos_dict = dict(pos_tags)
df['POS'] = df['Word'].map(pos_dict)

# Sort DataFrame by frequency
df = df.sort_values('Frequency', ascending=False)

print("\nWord Frequency and POS DataFrame:")
print(df)


from anytree import Node, RenderTree
import pandas as pd

# Your existing DataFrame creation logic
word_freq_dict = dict(frequency.most_common(20))  # Top 20 words
df = pd.DataFrame(list(word_freq_dict.items()), columns=['Word', 'Frequency'])

# Assume pos_tags is a list of tuples like: [('word1', 'NN'), ('word2', 'VB'), ...]
pos_dict = dict(pos_tags)
df['POS'] = df['Word'].map(pos_dict)

# Sort by frequency
df = df.sort_values('Frequency', ascending=False)

# ------------------ Tree Visualization ------------------
# Group words by POS
grouped = df.groupby('POS')

# Create root node
root = Node("Word Frequency Tree")

# Add child nodes under each POS
for pos, group in grouped:
    pos_node = Node(pos, parent=root)
    for _, row in group.iterrows():
        Node(f"{row['Word']} ({row['Frequency']})", parent=pos_node)

# Render the tree
for pre, fill, node in RenderTree(root):
    print(f"{pre}{node.name}")


# 2. Visualizations
plt.figure(figsize=(15, 10))

# Visualization 1: Bar Chart
plt.subplot(2, 2, 1)
plt.bar(df['Word'], df['Frequency'], color='skyblue')
plt.title('Top 20 Words - Bar Chart')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# Visualization 2: Line Graph
plt.subplot(2, 2, 2)
plt.plot(df['Word'], df['Frequency'], marker='o', color='orange', linestyle='-')
plt.title('Top 20 Words - Line Graph')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)

# Visualization 3: Word Cloud
plt.subplot(2, 2, 3)
wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq_dict)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud')
plt.axis('off')

# Visualization 4: Horizontal Bar Chart
plt.subplot(2, 2, 4)
plt.barh(df['Word'], df['Frequency'], color='green')
plt.title('Top 20 Words - Horizontal Bar')
plt.xlabel('Frequency')
plt.ylabel('Words')

plt.tight_layout()
plt.show()

# Save DataFrame to CSV
df.to_csv('word_frequencies_with_pos.csv', index=False)


# Define noun POS tags
noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}

# Filter only nouns
nouns = [word for word, tag in pos_tags if tag in noun_tags]
noun_freq = Counter(nouns)

# Convert to DataFrame
df_nouns = pd.DataFrame(noun_freq.most_common(10), columns=['Noun', 'Frequency'])
print("\nTop 10 Nouns:")
print(df_nouns)

# Define verb POS tags
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

# Filter only verbs
verbs = [word for word, tag in pos_tags if tag in verb_tags]
verb_freq = Counter(verbs)

# Convert to DataFrame
df_verbs = pd.DataFrame(verb_freq.most_common(10), columns=['Verb', 'Frequency'])
print("\nTop 10 Verbs:")
print(df_verbs)

#covert to cvs
df_nouns.to_csv("top_20_nouns.csv", index=False)
df_verbs.to_csv("top_20_verbs.csv", index=False)

# Create subplots for nouns and verbs
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Verbs
axes[0].bar(df_verbs['Verb'], df_verbs['Frequency'], color='orange', label='Verbs')
axes[0].set_title('Top 10 Verbs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Verb', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()

# Add value labels to verb bars
for i, v in enumerate(df_verbs['Frequency']):
    axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

# Subplot 2: Nouns
axes[1].bar(df_nouns['Noun'], df_nouns['Frequency'], color='green', label='Nouns')
axes[1].set_title('Top 10 Nouns', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Noun', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()

# Add value labels to noun bars
for i, v in enumerate(df_nouns['Frequency']):
    axes[1].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

plt.suptitle('Parts of Speech Frequency Comparison', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()