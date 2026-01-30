import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# --- STEP 1: LOAD DATA ---
print("Loading data...")
# Load the new Cybersecurity dataset
df = pd.read_csv('Cybersecurity_Dataset.csv')
print(f"Data loaded: {df.shape[0]} rows.")

# inspect the first few rows to ensure we have the right column
print("Sample of raw text:")
print(df['Cleaned Threat Description'].head())

# --- STEP 2: PREPROCESSING ---
print("\nPreprocessing text...")

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, float): 
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation/special chars (keeping basic words)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenize
    tokens = text.split()
    
    # 4. Remove stop words and Lemmatize
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(clean_tokens)

# Apply to the 'Cleaned Threat Description' column
df['processed_text'] = df['Cleaned Threat Description'].apply(preprocess_text)

# --- STEP 3: VECTORIZATION (TF-IDF) ---
print("\nVectorizing text...")
# We adjust min_df slightly since this dataset might be smaller or have more specific terms
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

# --- STEP 4: TOPIC MODELING (NMF) ---
print("Extracting topics with NMF...")
n_topics = 5
nmf_model = NMF(n_components=n_topics, random_state=42, init='nndsvd')
nmf_output = nmf_model.fit_transform(tfidf_matrix)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Function to print topics in the console
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        print(", ".join(top_features))

display_topics(nmf_model, feature_names, 10)

# --- STEP 5: VISUALIZATION (BAR CHARTS) ---
print("\nGenerating Bar Charts...")

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(30, 10), sharex=True)
    axes = axes.flatten()
    
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 20})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=15)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)

    fig.suptitle(title, fontsize=30)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.40, hspace=0.3)
    plt.show()

plot_top_words(nmf_model, feature_names, 10, 'Top Words per Topic (Cybersecurity)')
print("\nDone!")