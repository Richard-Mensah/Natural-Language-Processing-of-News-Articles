# Natural-Language-Processing-of-News-Articles
Mini-Project: Natural Language Processing of News Articles: Frequency, POS, and Visual Interpretation

Author: Richard Mensah
Username: RM24ZKD
School ID: 500699458
Date: 2nd May, 2024

Australian Broadcasting Commission 2006
http://www.abc.net.au/

Contents:
* Rural News    http://www.abc.net.au/rural/news/
* Science News  http://www.abc.net.au/science/news/

📌 Project Overview
This project applies Natural Language Processing (NLP) techniques to analyze news articles from the ABC (Australian Broadcasting Corporation) corpus. It focuses on:
✔ Word Frequency Analysis (identifying most common words)
✔ Part-of-Speech (POS) Tagging (grammatical classification)
✔ Data Visualization (bar charts, word clouds, tree structures)

The dataset combines Rural News and Science News articles, providing a mix of environmental, agricultural, and scientific narratives.

### 📂 Dataset
- Source: NLTK ABC Corpus (rural.txt + science.txt)

Combined File: ScienceRule.txt

# ## Key Stats:
- Total Words: 8,752
- Unique Words: 3,459
- Characters: 47,510

### 🛠️ NLP Techniques Used
Technique	Purpose	Tools/Libraries
Tokenization	Split text into words	nltk.word_tokenize()
Stopword Removal	Filter common words (e.g., "the", "and")	nltk.corpus.stopwords
POS Tagging	Label words by grammatical role (noun, verb, etc.)	nltk.pos_tag()
Frequency Analysis	Count word occurrences	collections.Counter
Tree Visualization	Hierarchical grouping by POS	anytree
Word Cloud	Visualize frequent words	WordCloud

### Bar/Line Charts	Compare word frequencies	matplotlib
📊 Key Results
Top 20 Most Frequent Words
Word	Frequency	POS Tag
says	7482	VBZ (verb)
said	2196	VBD (past-tense verb)
new	2173	JJ (adjective)
australia	2023	NNP (proper noun)
could	1500	MD (modal verb)
Top 10 Nouns & Verbs
Nouns	Frequency	Verbs	Frequency
people	1281	says	7482
water	1249	said	2196
years	1206	say	1222
farmers	1153	found	977
research	1144	used	670

### 📈 Visualizations
Bar Chart (Top 20 Words)

Line Graph (Frequency Trends)

Word Cloud (Dominant Terms)

POS Tree Structure (Hierarchical Grouping)

Nouns vs. Verbs Comparison

Word Cloud Example (Replace with actual image)

### 📂 Files & Outputs
File	Description
ScienceRule.txt	Combined ABC News Corpus
word_freq_with_pos.csv	Top 20 Words + POS Tags
top_10_nouns.csv	Most Frequent Nouns
top_10_verbs.csv	Most Frequent Verbs
*.png	Saved Visualizations
🚀 How to Run the Code
Install Dependencies:

bash
pip install nltk pandas matplotlib wordcloud anytree
Download NLTK Data:

python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
Run the Script:

bash
python nlp_news_analysis.py
🔍 Insights & Conclusion
Dominant Verbs: "Says," "said," "found" → Reporting-focused language

Key Nouns: "Water," "farmers," "research" → Themes: Environment & Science

POS Trends: High use of verbs (reporting actions) and nouns (topics)

Future Improvements:

Lemmatization (merge "says," "said" → "say")

Bigrams/Trigrams (identify common phrases)

Sentiment Analysis (tone of articles)

📜 License
This project is open-source under the MIT License.

🎯 Developed by Richard Mensah | RM24ZKD | 2024

🔗 Appendix
📄 Full Code: See nlp_news_analysis.py (Link to GitHub repo if available)
📊 Sample Outputs:

word_freq_with_pos.csv

top_10_nouns.csv

top_10_verbs.csv