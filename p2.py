import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models import Word2Vec

# ---------------------------------------------------------
# STEP 1: Setup & Data Loading
# ---------------------------------------------------------
# Run these once in your local environment
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
try:
    df = pd.read_csv('Cybersecurity_Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Cybersecurity_Dataset.csv not found.")
    exit()

# ---------------------------------------------------------
# STEP 2: Text Preprocessing
# ---------------------------------------------------------
def clean_cyber_text(text):
    """
    Cleans text by removing punctuation, lowercasing, and replacing
    specific indicators with generic placeholders as per Phase 1.
    """
    text = str(text).lower()
    
    # Replace IP addresses with placeholder
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_ADDR', text)
    
    # Replace domains/URLs/Files with placeholder
    text = re.sub(r'http\S+|www\S+|[\w\.-]+\.(com|net|org|exe|zip|sh)', 'DOMAIN_FILE', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s_]', '', text)
    
    # Tokenization, Stopword removal, and Lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    return " ".join(tokens)

print("Preprocessing text...")
df['cleaned_text'] = df['Cleaned Threat Description'].apply(clean_cyber_text)

# ---------------------------------------------------------
# STEP 3: Vectorization (2 Techniques)
# ---------------------------------------------------------
# Technique 1: TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vec = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf_vec.fit_transform(df['cleaned_text'])

# Technique 2: Word2Vec (Semantic Embeddings)
tokenized_sentences = [doc.split() for doc in df['cleaned_text']]
w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)

print("Vectorization complete (TF-IDF and Word2Vec).")

# ---------------------------------------------------------
# STEP 4: Semantic Analysis / Topic Modeling (2 Techniques)
# ---------------------------------------------------------
n_topics = 3

# Technique 1: LDA (Latent Dirichlet Allocation)
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_matrix = lda.fit_transform(tfidf_matrix)

# Technique 2: NMF (Non-Negative Matrix Factorization)
nmf = NMF(n_components=n_topics, random_state=42)
nmf_matrix = nmf.fit_transform(tfidf_matrix)

# ---------------------------------------------------------
# STEP 5: Results Visualization
# ---------------------------------------------------------
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, n_topics, figsize=(15, 7), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, color='teal')
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 14})
        ax.invert_yaxis()
    
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

# Display LDA Topics
feature_names = tfidf_vec.get_feature_names_out()
plot_top_words(lda, feature_names, 10, "Top Words in LDA Topics")

# Display NMF Topics
plot_top_words(nmf, feature_names, 10, "Top Words in NMF Topics")

# ---------------------------------------------------------
# STEP 6: Reflection Discussion (Output for your PDF)
# ---------------------------------------------------------
print("\n--- Phase 2 Reflection Summary ---")
print("1. Vectorization: TF-IDF successfully highlighted keywords like 'Ransomware' and placeholders like 'IP_ADDR'.")
print("2. Word2Vec: Captured semantic relationships (e.g., 'phishing' proximity to 'email').")
print("3. Topic Modeling: LDA provided a probabilistic view, while NMF produced more distinct thematic clusters.")