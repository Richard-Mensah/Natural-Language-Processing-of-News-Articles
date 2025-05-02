import nltk
from collections import Counter
from nltk.corpus import abc
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the corpus
corpus = abc.raw(['rural.txt', "science.txt"])  # Using this as corpus

with open("ScienceRule.txt" , "w", encoding ="uTF-8") as f:
    f.write(corpus)

def preprocess_text(text):
    """Cleans and preprocesses the text."""
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    # Removing stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def analyze_text(tokens):
    """Analyzes the text and returns frequency distribution."""
    return Counter(tokens)

# Preprocess and analyze the text
tokens = preprocess_text(corpus)
frequency = analyze_text(tokens)

# Print the 20 most common words
print("10 Most Common Words:")
for word, count in frequency.most_common(20):
    print(f"{word}: {count}")
    dic = {"word":frequency.get(word)}


# 1. Create dictionary and DataFrame from frequency data
word_freq_dict = dict(frequency.most_common(20))  # Top 20 words
df = pd.DataFrame(list(word_freq_dict.items()), columns=['Word', 'Frequency'])

# Sort DataFrame by frequency for better visualization
df = df.sort_values('Frequency', ascending=False)

print("\nWord Frequency DataFrame:")
print(df.head())

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

# Visualization 4: Horizontal Bar Chart (Bonus)
plt.subplot(2, 2, 4)
plt.barh(df['Word'], df['Frequency'], color='green')
plt.title('Top 20 Words - Horizontal Bar')
plt.xlabel('Frequency')
plt.ylabel('Words')

plt.tight_layout()
plt.show()

# Save DataFrame to CSV
df.to_csv('word_frequencies.csv', index=False)