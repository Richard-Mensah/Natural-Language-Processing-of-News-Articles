# -------------------- Import Required Libraries --------------------
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import warnings
warnings.filterwarnings("ignore")

# -------------------- Download NLTK Resources --------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- Load Your Corpus --------------------
with open("ScienceRule.txt", "r", encoding="utf-8") as file:
    text = file.read()

# -------------------- Preprocessing Function --------------------
def preprocess_text(text):
    """
    Tokenizes, removes stopwords, non-alphabetic tokens, and applies lemmatization.
    """
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

# -------------------- Prepare Data for Topic Modeling --------------------
processed_tokens = preprocess_text(text)

# Create a list of token lists for LDA (since LDA expects a list of documents, not a single long text)
# We'll split the tokens into smaller chunks (e.g., 100 words per "document")
chunk_size = 100
chunks = [processed_tokens[i:i + chunk_size] for i in range(0, len(processed_tokens), chunk_size)]

# -------------------- Create Dictionary and Corpus --------------------
dictionary = corpora.Dictionary(chunks)
corpus = [dictionary.doc2bow(chunk) for chunk in chunks]

# -------------------- Train LDA Model --------------------
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10, random_state=42)

# -------------------- Display the Topics --------------------
print("\n🔍 Top Topics Discovered:")
topics = lda_model.print_topics(num_words=10)
for idx, topic in topics:
    print(f"\n🧠 Topic #{idx + 1}:")
    print(topic)

# -------------------- Visualize with pyLDAvis --------------------
pyLDAvis.enable_notebook()
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, 'topic_modeling_visualization.html')
print("\n📄 HTML interactive visualization saved as 'topic_modeling_visualization.html'.")

# -------------------- Word Cloud for Each Topic --------------------
for idx, topic in lda_model.show_topics(num_topics=5, formatted=False):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(topic))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for Topic #{idx + 1}", fontsize=14)
    plt.show()
