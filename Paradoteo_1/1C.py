import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from jiwer import wer  # για Word Error Rate
from bert_score import score as bert_score  # για BERTScore
from sentence_transformers import SentenceTransformer, util  # για SBERT Cosine Similarity
from sklearn.feature_extraction.text import TfidfVectorizer  # για TF-IDF
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 

# Download required NLTK data
nltk_data = ['punkt_tab', 'punkt', 'stopwords', 'wordnet']
for resource in nltk_data:
    try:
        nltk.download(resource, quiet=True)
        print(f"Downloaded {resource}")
    except Exception as e:
        print(f"Warning: Could not download {resource}: {e}")


# Αρχικές προτάσεις
default_sentence_1 = "Thank you for forwarding our remarks to the doctor for his upcoming contract review on behalf of all of us."
default_sentence_2 = "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think."

# Naive
naive_sentence_1 = "Thank you for your message to show our words to the doctor, during his next contract review, to all of us."
naive_sentence_2 = "We should be grateful, i mean all of us, for the acceptance and efforts until the springer link came finally last week, i think."

# Vamsi/T5_Paraphrase_Paws
t5_sentence_1 = "Thank you for your message to show our words to the doctor, as his next contract checking, a couple of days ago."
t5_sentence_2 = "but we should be grateful for the acceptance and efforts until the Springer link finally came last week , I think ."

# eugenesiow/bart-paraphrase
bart_sentence_1 = "Thank you for your message to show our words to the doctor, as his next contract checking, a couple of days ago."
bart_sentence_2 = "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link finally came last week, if the doctor still plans to edit the acknowledgments section before sending it again, I believe."

# prithivida/grammar_error_correcter_v1
grammar_sentence_1 = "Thank you for your message to show our words to the doctor, as his next contract check, to all of us."
grammar_sentence_2 = "We should be grateful, I mean all of us, for the new submission — the one we were waiting for since last autumn"

def ensure_list(x):
    return x if isinstance(x, list) else [x]

# Precision, Recall, F1 βάσει tokens with fallback tokenization
def token_prf(cand: str, ref: str):
    try:
        cand_toks = nltk.word_tokenize(cand.lower())
        ref_toks = nltk.word_tokenize(ref.lower())
    except LookupError:
        # Fallback tokenization if NLTK fails
        print("Warning: NLTK tokenizer not available, using simple split")
        cand_toks = cand.lower().split()
        ref_toks = ref.lower().split()

    if not cand_toks or not ref_toks:
        return 0.0, 0.0, 0.0
    
    matches = sum(1 for t in cand_toks if t in ref_toks)
    P = matches / len(cand_toks)
    R = matches / len(ref_toks)
    if((P + R) > 0):
        F1 = 2 * P * R / (P + R)
    else:
        F1 = 0.0

    return P, R, F1

# Word Error Rate | χαμηλότερο = καλύτερο
def wer_score(cand: str, ref: str):
    return wer(ref.lower(), cand.lower())

# SBERT
sbert = SentenceTransformer('all-MiniLM-L6-v2')

def sbert_cosine(cand: str, ref: str):
    emb_c = sbert.encode(cand, convert_to_tensor=True)
    emb_r = sbert.encode(ref, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb_c, emb_r).item()

# Prepare all texts for TF-IDF
all_texts = [
    default_sentence_1, default_sentence_2, 
    naive_sentence_1, naive_sentence_2,
    t5_sentence_1, t5_sentence_2,
    bart_sentence_1, bart_sentence_2,
    grammar_sentence_1, grammar_sentence_2
]

print("Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer().fit(all_texts)

# TF-IDF cosine similarity
def tfidf_cosine(cand: str, ref: str):
    v = tfidf.transform([cand, ref])
    return cosine_similarity(v[0], v[1])[0, 0]

# Define pipelines
pipelines = [
    ('naive_reconstructed', [naive_sentence_1], [naive_sentence_2]),
    ('t5_paraphraser', [t5_sentence_1], [t5_sentence_2]),
    ('bart_paraphrase', [bart_sentence_1], [bart_sentence_2]),
    ('grammar_correcter', [grammar_sentence_1], [grammar_sentence_2]),
]

records = []

print("Starting evaluation...")

# Για κάθε pipeline και πρόταση
for name, gen1, gen2 in tqdm(pipelines, desc="Evaluating pipelines"):
    print(f"Processing {name}...")
    
    # Πρόταση 1
    try:
        p, r, f1 = token_prf(gen1[0], default_sentence_1)
        w = wer_score(gen1[0], default_sentence_1)
        _, _, fb1 = bert_score([gen1[0]], [default_sentence_1], lang='en', rescale_with_baseline=True)
        sb = sbert_cosine(gen1[0], default_sentence_1)
        tc = tfidf_cosine(gen1[0], default_sentence_1)
        
        records.append({
            'Pipeline': f"{name}_1",
            'Tok_Prec': p, 'Tok_Rec': r, 'Tok_F1': f1,
            'WER': w, 'BERT_F1': fb1.item(),
            'SBERT_Cos': sb, 'TFIDF_Cos': tc,
        })
    except Exception as e:
        print(f"Error processing {name}_1: {e}")
        continue

    # Πρόταση 2
    try:
        p, r, f1 = token_prf(gen2[0], default_sentence_2)
        w = wer_score(gen2[0], default_sentence_2)
        _, _, fb2 = bert_score([gen2[0]], [default_sentence_2], lang='en', rescale_with_baseline=True)
        sb = sbert_cosine(gen2[0], default_sentence_2)
        tc = tfidf_cosine(gen2[0], default_sentence_2)
        
        records.append({
            'Pipeline': f"{name}_2",
            'Tok_Prec': p, 'Tok_Rec': r, 'Tok_F1': f1,
            'WER': w, 'BERT_F1': fb2.item(),
            'SBERT_Cos': sb, 'TFIDF_Cos': tc,
        })
    except Exception as e:
        print(f"Error processing {name}_2: {e}")
        continue

# Μετατροπή σε DataFrame
if records:
    df = pd.DataFrame(records)

    # Μέση τιμή ανά pipeline (group by pipeline name without _1/_2 suffix)
    df['Pipeline_Base'] = df['Pipeline'].str.replace('_[12]$', '', regex=True)
    
    # Select only numeric columns for aggregation
    numeric_cols = ['Tok_Prec', 'Tok_Rec', 'Tok_F1', 'WER', 'BERT_F1', 'SBERT_Cos', 'TFIDF_Cos']
    summary = df.groupby('Pipeline_Base')[numeric_cols].mean().reset_index()

    print("\nΑποτελέσματα:")
    print(summary.set_index('Pipeline_Base').round(3))

    # Visualization
    metrics = ['Tok_F1', 'WER', 'BERT_F1', 'SBERT_Cos', 'TFIDF_Cos']
    x = np.arange(len(summary))
    width = 0.15 

    plt.figure(figsize=(14, 8))
    for i, m in enumerate(metrics):
        plt.bar(x + i * width, summary[m], width, label=m)

    plt.xticks(x + width * (len(metrics) - 1) / 2, summary['Pipeline_Base'], rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Σύγκριση Μεθόδων Ανακατασκευής Κειμένου')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Additional detailed results table
    print("\nΛεπτομερή Αποτελέσματα:")
    print(df.set_index('Pipeline').round(3))

    # Best performing methods per metric
    print("\nΚαλύτερες Μέθοδοι ανά Μετρική:")
    for metric in ['Tok_F1', 'BERT_F1', 'SBERT_Cos', 'TFIDF_Cos']:
        best_method = summary.loc[summary[metric].idxmax(), 'Pipeline_Base']
        best_score = summary[metric].max()
        print(f"{metric}: {best_method} ({best_score:.3f})")
    
    # For WER (lower is better)
    best_method_wer = summary.loc[summary['WER'].idxmin(), 'Pipeline_Base']
    best_score_wer = summary['WER'].min()
    print(f"WER: {best_method_wer} ({best_score_wer:.3f})")
else:
    print("No results to display - all evaluations failed!")