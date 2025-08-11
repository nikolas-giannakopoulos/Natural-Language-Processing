import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Necessary setup
packages = ['sentence-transformers', 'gensim', 'seaborn', 'nltk']

for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Download NLTK data
nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for resource in nltk_data:
    nltk.download(resource, quiet=True)


# Texts
texts = {
    "original": {
        "text1": "Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication",
        "text2": "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again."
    },
    "reconstructed": {
        "naive": {
            "text1": "Thank you for your message to show our words to the doctor, during his next contract review, to all of us.",
            "text2": "We should be grateful, i mean all of us, for the acceptance and efforts until the springer link came finally last week, i think."
        },
        "t5": {
            "text1": "Thank you for forwarding our message to the doctor regarding his upcoming contract review. I greatly appreciate the professor's full support for our Springer proceedings publication.",
            "text2": "We should all be grateful for the acceptance and efforts leading to the Springer link's arrival last week. Please remind me if the doctor plans to edit the acknowledgments before resubmission."
        },
        "bert": {
            "text1": "Thank you for your message showing our words to the doctor for his next contract review. I received the professor's message a couple of days ago and very much appreciate the full support for our Springer proceedings publication.",
            "text2": "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link finally came last week. Also, please kindly remind me if the doctor still plans to edit the acknowledgments section before sending again."
        }
    }
}

# Custom NLP Preprocessing Pipeline
class TextProcessor:

    # Constructor
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
    # Comperhensive Text Preprocessing
    def clean_text(self, text, keep_stopwords=False):
        tokens = word_tokenize(text.lower())
        if keep_stopwords:
            cleaned = [self.lemmatizer.lemmatize(t) for t in tokens if t.isalnum()]
        else:
            cleaned = [self.lemmatizer.lemmatize(t) for t in tokens 
                      if t.isalnum() and t not in self.stop_words]
        
        return cleaned
    
    # Extract Vocabulary from Collection of Texts  
    def get_vocab(self, text_list):
        vocab = set()
        for txt in text_list:
            vocab.update(self.clean_text(txt))

        return sorted(list(vocab))
    
    # Analyze Sentence Structure
    def analyze_structure(self, text):
        sentences = sent_tokenize(text)
        return {
            'sentence_count': len(sentences),
            'avg_length': np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
            'sentences': sentences
        }

# Comprehensive Embeddings Analyzer with Multiple Models
class EmbeddingAnalyzer:

    # Constructor
    def __init__(self):
        self.models = {}
        self.processor = TextProcessor()
        self._load_models()
        
    # Load Embedding Models
    def _load_models(self):
        # Load sentence transformer - most reliable for our use case
        try:
            self.models['sbert'] = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded sentence transformer")
        except Exception as e:
            print(f"Failed to load sentence transformer: {e}")
        
        # Try to load GloVe - useful for word-level analysis
        try:
            self.models['glove'] = api.load('glove-twitter-25')
            print("Loaded GloVe embeddings")
        except:
            print("GloVe embeddings not available")
        
        self._train_custom_w2v()
    
    # Create Word2Vec & FastText Embedding
    def _train_custom_w2v(self):
        # Collect all text for training
        all_text = []
        all_text.extend(texts["original"].values())
        for method in texts["reconstructed"].values():
            all_text.extend(method.values())
        
        # Tokenize for training
        corpus = [self.processor.clean_text(txt, keep_stopwords=True) for txt in all_text]
        corpus = [tokens for tokens in corpus if len(tokens) > 0]
        
        if corpus:
            try:
                self.models['w2v'] = Word2Vec(corpus, vector_size=100, window=5, 
                                           min_count=1, sg=1, epochs=10)
            except Exception as e:
                print(f"W2V training failed: {e}")
            
            try:
                self.models['fasttext'] = FastText(corpus, vector_size=100, window=5,
                                                 min_count=1, sg=1, epochs=10)
            except Exception as e:
                print(f"FastText training failed: {e}")
    
    # Get embedding for a single word
    def get_word_vector(self, word, model='glove'):
        word = word.lower()
        if model == 'glove' and 'glove' in self.models:
            try:
                return self.models['glove'][word]
            except KeyError:
                return None
        elif model == 'w2v' and 'w2v' in self.models:
            try:
                return self.models['w2v'].wv[word]
            except KeyError:
                return None
        elif model == 'fasttext' and 'fasttext' in self.models:
            try:
                return self.models['fasttext'].wv[word]
            except KeyError:
                return None
            
        return None
    
    # Get Embedding For Entire Text
    def get_text_vector(self, text, method='sbert'):
        if method == 'sbert' and 'sbert' in self.models:
            return self.models['sbert'].encode(text)
        elif method in ['glove', 'w2v', 'fasttext']:
            tokens = self.processor.clean_text(text)
            vectors = []
            for token in tokens:
                vec = self.get_word_vector(token, method)
                if vec is not None:
                    vectors.append(vec)
            
            return np.mean(vectors, axis=0) if vectors else None
        
        return None
    
    # Compute cosine similarities
    def calculate_similarities(self):
        results = {}
        available_methods = [m for m in ['sbert', 'glove', 'w2v', 'fasttext'] if m in self.models]
        for method in available_methods:
            results[method] = {}
            for text_id in texts["original"].keys():
                orig_text = texts["original"][text_id]
                orig_vec = self.get_text_vector(orig_text, method)
                if orig_vec is not None:
                    sims = {}
                    for recon_method, recon_texts in texts["reconstructed"].items():
                        recon_text = recon_texts[text_id]
                        recon_vec = self.get_text_vector(recon_text, method)
                        if recon_vec is not None:
                            sim = cosine_similarity(orig_vec.reshape(1, -1), 
                                                  recon_vec.reshape(1, -1))[0, 0]
                            sims[recon_method] = sim
                    results[method][text_id] = sims
        
        return results

# Visualize text embeddings
def plot_embeddings_2d(analyzer, embedding_method='sbert'):
    if embedding_method not in analyzer.models:
        print(f"Method {embedding_method} not available")
        return
    
    # Collect all text embeddings
    embeddings = []
    labels = []
    
    # Original texts
    for text_id, text in texts["original"].items():
        vec = analyzer.get_text_vector(text, embedding_method)
        if vec is not None:
            embeddings.append(vec)
            labels.append(f"Original_{text_id}")
    
    # Reconstructed texts
    for method, method_texts in texts["reconstructed"].items():
        for text_id, text in method_texts.items():
            vec = analyzer.get_text_vector(text, embedding_method)
            if vec is not None:
                embeddings.append(vec)
                labels.append(f"{method}_{text_id}")
    
    if len(embeddings) < 2:
        print(f"Not enough embeddings for visualization ({len(embeddings)} found)")
        return
    
    embeddings = np.array(embeddings)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA visualization
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax1.scatter(embeddings_pca[i, 0], embeddings_pca[i, 1], 
                   c=[color], s=100, label=label, alpha=0.7)
        ax1.annotate(label, (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title(f'PCA Visualization - {embedding_method.upper()}')
    ax1.grid(True, alpha=0.3)
    
    # t-SNE visualization
    if len(embeddings) >= 4:
        perplexity = min(5, len(embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax2.scatter(embeddings_tsne[i, 0], embeddings_tsne[i, 1], 
                       c=[color], s=100, label=label, alpha=0.7)
            ax2.annotate(label, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_title(f't-SNE Visualization - {embedding_method.upper()}')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Not enough points\nfor t-SNE', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('t-SNE (Insufficient Data)')
    
    plt.tight_layout()
    plt.show()

def visualize_word_embeddings(analyzer, words=None, embedding_method='w2v'):
    if embedding_method not in analyzer.models:
        print(f"Method {embedding_method} not available")
        return
    
    if words is None:
        # Get common words from our texts
        all_text = []
        all_text.extend(texts["original"].values())
        for method_texts in texts["reconstructed"].values():
            all_text.extend(method_texts.values())
        
        vocab = analyzer.processor.get_vocab(all_text)
        words = vocab[:15]  # Top 15 words
    
    # Collect word embeddings
    embeddings = []
    valid_words = []
    
    for word in words:
        vec = analyzer.get_word_vector(word, embedding_method)
        if vec is not None:
            embeddings.append(vec)
            valid_words.append(word)
    
    if len(embeddings) < 2:
        print(f"Not enough word embeddings for visualization ({len(embeddings)} found)")
        return
    
    embeddings = np.array(embeddings)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA visualization
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_words)))
    
    for i, (word, color) in enumerate(zip(valid_words, colors)):
        ax1.scatter(embeddings_pca[i, 0], embeddings_pca[i, 1], 
                   c=[color], s=100, alpha=0.7)
        ax1.annotate(word, (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title(f'Word Embeddings PCA - {embedding_method.upper()}')
    ax1.grid(True, alpha=0.3)
    
    # t-SNE visualization
    if len(embeddings) >= 4:
        perplexity = min(5, len(embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        for i, (word, color) in enumerate(zip(valid_words, colors)):
            ax2.scatter(embeddings_tsne[i, 0], embeddings_tsne[i, 1], 
                       c=[color], s=100, alpha=0.7)
            ax2.annotate(word, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_title(f'Word Embeddings t-SNE - {embedding_method.upper()}')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Not enough points\nfor t-SNE', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('t-SNE (Insufficient Data)')
    
    plt.tight_layout()
    plt.show()

def plot_results(similarity_data):
    # Convert to DataFrame for easier plotting
    plot_data = []
    for method, texts_data in similarity_data.items():
        for text_id, sims in texts_data.items():
            for recon_method, sim_score in sims.items():
                plot_data.append({
                    'embedding': method,
                    'text': text_id,
                    'reconstruction': recon_method,
                    'similarity': sim_score
                })
    
    if not plot_data:
        print("No similarity data to plot")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Create heatmap
    pivot = df.pivot_table(values='similarity', 
                          index=['embedding', 'text'], 
                          columns='reconstruction')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Text Reconstruction Quality by Embedding Method')
    plt.tight_layout()
    plt.show()
    
    return df

def tfidf_comparison():
    # TF-IDF Based Similarity Analysis
    all_texts = []
    labels = []
    
    # Add original texts
    for text_id, text in texts["original"].items():
        all_texts.append(text)
        labels.append(f"original_{text_id}")
    
    # Add reconstructed texts
    for method, method_texts in texts["reconstructed"].items():
        for text_id, text in method_texts.items():
            all_texts.append(text)
            labels.append(f"{method}_{text_id}")
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarities, annot=True, xticklabels=labels, yticklabels=labels,
                cmap='coolwarm', fmt='.2f')
    plt.title('TF-IDF Similarity Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return similarities

def analyze_structure():
    processor = TextProcessor()
    
    print("Text Structure Analysis:")
    
    for text_id in texts["original"].keys():
        orig = texts["original"][text_id]
        orig_stats = processor.analyze_structure(orig)
        
        print(f"\n{text_id}:")
        print(f"Original: {orig_stats['sentence_count']} sentences, {orig_stats['avg_length']:.1f} avg words")
        
        for method, method_texts in texts["reconstructed"].items():
            recon_stats = processor.analyze_structure(method_texts[text_id])
            print(f"{method:8}: {recon_stats['sentence_count']} sentences, {recon_stats['avg_length']:.1f} avg words")

# Main Comprehensive Analysis Function
def run_analysis():    
    # Structure analysis
    analyze_structure()
    
    # Embedding similarity analysis
    analyzer = EmbeddingAnalyzer()
    similarities = analyzer.calculate_similarities()
    
    print("\nSimilarity Results:")
    
    for method, data in similarities.items():
        print(f"\n{method.upper()} embeddings:")
        for text_id, sims in data.items():
            print(f"  {text_id}:")
            for recon_method, score in sims.items():
                print(f"    {recon_method:8}: {score:.4f}")
    
    # Create visualizations
    df = plot_results(similarities)
    tfidf_sim = tfidf_comparison()
    
    # Text embeddings visualization
    available_methods = [m for m in ['sbert', 'glove', 'w2v', 'fasttext'] if m in analyzer.models]
    for method in available_methods:
        print(f"Visualizing {method} text embeddings")
        plot_embeddings_2d(analyzer, method)
    
    # Word embeddings visualization
    word_methods = [m for m in ['w2v', 'fasttext', 'glove'] if m in analyzer.models]
    for method in word_methods:
        print(f"Visualizing {method} word embeddings")
        visualize_word_embeddings(analyzer, embedding_method=method)
    
    # Find best method
    if similarities and 'sbert' in similarities:
        method_scores = {}
        sbert_data = similarities['sbert']
        
        for recon_method in texts["reconstructed"].keys():
            scores = []
            for text_data in sbert_data.values():
                if recon_method in text_data:
                    scores.append(text_data[recon_method])
            
            if scores:
                method_scores[recon_method] = np.mean(scores)
        
        if method_scores:
            best = max(method_scores, key=method_scores.get)
            print(f"\nBest reconstruction method: {best} (avg similarity: {method_scores[best]:.3f})")
    
    return similarities, analyzer

if __name__ == "__main__":
    results = run_analysis()