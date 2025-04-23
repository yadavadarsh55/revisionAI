from pypdf import PdfReader
import spacy
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from keybert import KeyBERT
import hdbscan

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in (reader.pages):
        text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)          # remove extra whitespace
    text = re.sub(r'(?<=\w)\.(?=\w)', '. ', text)  # add space after periods if missing
    text = text.strip()
    return text

def text_into_sent(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if (len(sent.text.strip()) > 10)]

def sentence_embeddigns(embedder, sentences):
    return embedder.encode(sentences)

def cluster_sentences(embeddings, sentences, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(sentences[idx])
    return clusters

def optimal_clusters_sentences(embeddings, sentences):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2,min_samples=1, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1: 
            continue
        clusters.setdefault(label, []).append(sentences[idx])
    return clusters

def get_cluster_title(sentences, embedder, top_n=2):
    if not sentences:
        return ["Untitled"]
    kw_model = KeyBERT(embedder)
    cluster_text = " ".join(sentences)
    keywords = kw_model.extract_keywords(cluster_text, top_n=top_n, stop_words='english')
    return [kw[0] for kw in keywords] if keywords else ["Untitled"]

def clustered_outputs(clustered_sentences, embedder):
    final_output = []
    for cluster_id, sentences in clustered_sentences.items():
        keywords = get_cluster_title(sentences, embedder)
        title = ", ".join(keywords).title()
        final_output.append({
            "topic": title,
            "sentences": sentences
        })
    return final_output

def get_topics(pdf_file, embedder):
    text = extract_text_from_pdf(pdf_file)
    text = clean_text(text)
    sentences = text_into_sent(text=text)
    sentence_embedding = sentence_embeddigns(embedder, sentences)
    clustered = optimal_clusters_sentences(embeddings=sentence_embedding, sentences=sentences)
    final_sentences = clustered_outputs(clustered, embedder)
    return final_sentences

if(__name__ == "__main__"):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    final_output = get_topics("sample_notes.pdf")
    print(final_output)
    
