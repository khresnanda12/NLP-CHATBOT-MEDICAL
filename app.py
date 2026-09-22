import re
import random

import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MedBot Alodokter", page_icon="🏥")

DATA_PATH = "alodokter_2000.csv"


# ---------------------------------------------------------------
# 1. Resource berat: di-cache supaya hanya dimuat sekali
# ---------------------------------------------------------------
@st.cache_resource
def siapkan_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        nltk.download(pkg, quiet=True)
    return True


@st.cache_resource
def muat_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None  # fallback ke TF-IDF


siapkan_nltk()
SBERT_MODEL = muat_sbert()
USE_SBERT = SBERT_MODEL is not None


# ---------------------------------------------------------------
# 2. Preprocessing (sama seperti notebook)
# ---------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
indonesian_stopwords = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'atau', 'juga', 'dalam', 'tidak', 'akan', 'ada',
    'saya', 'kamu', 'anda', 'ia', 'mereka', 'kami', 'kita', 'bisa',
    'sudah', 'bila', 'jika', 'maka', 'oleh', 'karena', 'apa',
    'bagaimana', 'berapa', 'kapan', 'dimana', 'siapa', 'apakah', 'cara',
    'lebih', 'sangat', 'dapat', 'nya', 'pun', 'lagi', 'belum',
    'telah', 'namun', 'tapi', 'serta', 'meski', 'agar', 'supaya', 'hal',
    'the', 'is', 'are', 'was', 'what', 'how', 'why', 'when', 'where'
}
all_stopwords = indonesian_stopwords | set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


# ---------------------------------------------------------------
# 3. Dataset (sama seperti Langkah 3 di notebook)
# ---------------------------------------------------------------
def ambil_kategori(tag):
    if pd.isna(tag):
        return "umum"
    teks = re.sub(r"[\[\]'\"]", " ", str(tag))
    bagian = [b for b in re.split(r"[,\s]+", teks) if b]
    return bagian[0].replace("-", " ") if bagian else "umum"


@st.cache_data
def muat_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["title", "question", "answer"]).drop_duplicates(subset="question").reset_index(drop=True)
    df["category"] = df["tag"].apply(ambil_kategori) if "tag" in df.columns else "umum"
    df["question_asli"] = df["question"]
    df["question"] = df["title"].str.strip() + ". " + df["question_asli"].str.strip()
    df["processed_question"] = df["question"].apply(preprocess_text)
    return df


# ---------------------------------------------------------------
# 4. Engine chatbot (sama seperti Langkah 5, termasuk perbaikan keyword boost)
# ---------------------------------------------------------------
class MedicalChatbotEngineV3:
    def __init__(self, dataframe, threshold=0.35, top_k=10):
        self.df = dataframe
        self.threshold = threshold if USE_SBERT else 0.15
        self.top_k = top_k
        self._build_index()
        self._define_rules()

    def _build_index(self):
        if USE_SBERT:
            self.sbert_embeddings = SBERT_MODEL.encode(self.df['question'].tolist(), convert_to_tensor=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_question'])

    def _define_rules(self):
        self.rules = {
            'emergency': {
                'patterns': [r'(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan)'],
                'responses': ["🚨 DARURAT! Hubungi 119 atau segera ke IGD!"]
            },
            'greeting': {
                'patterns': [r'\b(halo|hai|hi|hello)\b'],
                'responses': ["👋 Halo! Ada yang bisa saya bantu?"]
            }
        }

    def _check_rules(self, text):
        for data in self.rules.values():
            for pattern in data['patterns']:
                if re.search(pattern, text.lower()):
                    return random.choice(data['responses'])
        return None

    def _search_sbert(self, query):
        from sentence_transformers import util
        emb = SBERT_MODEL.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(emb, self.sbert_embeddings)[0].cpu().numpy()
        top = np.argsort(-scores)[:self.top_k]
        return [(idx, float(scores[idx])) for idx in top]

    def _search_tfidf(self, query):
        vec = self.vectorizer.transform([preprocess_text(query)])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top = np.argsort(scores)[::-1][:self.top_k]
        return [(idx, float(scores[idx])) for idx in top]

    def get_response(self, user_input, history):
        if not user_input.strip():
            return "Silakan ketik pertanyaan."

        rule = self._check_rules(user_input)
        if rule:
            return rule

        # konteks dari input sebelumnya (disimpan per pengguna di session_state)
        query = user_input

        if USE_SBERT:
            results, method = self._search_sbert(query), "SBERT"
        else:
            results, method = self._search_tfidf(query), "TF-IDF"

        if results[0][1] < self.threshold:
            return "🤔 Tidak menemukan jawaban yang cukup relevan."

        kata_umum = {"mengatasi", "mengobati", "menghilangkan", "gejala", "penyebab", "obat"}
        boosted = []
        for idx, score in results:
            text = self.df.iloc[idx]['question']
            bonus = sum(1 for word in preprocess_text(user_input).split()
                        if word not in kata_umum and word in text.lower())
            boosted.append((idx, score + 0.05 * bonus))

        best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]
        row = self.df.iloc[best_idx]
        history.append(user_input)

        return (f"**Kategori:** {row['category']} · {method}\n\n"
                f"**Topik terkait:** {row['title']}\n\n"
                f"{row['answer']}\n\n"
                f"---\n⚠️ Untuk kondisi serius, konsultasikan ke dokter.")


@st.cache_resource
def buat_bot():
    return MedicalChatbotEngineV3(muat_data())


# ---------------------------------------------------------------
# 5. Tampilan Streamlit
# ---------------------------------------------------------------
st.title("🏥 MedBot Alodokter")
st.caption("Chatbot tanya-jawab kesehatan berbasis 2.000 data forum Alodokter. "
           "Hanya untuk edukasi, bukan pengganti dokter. Darurat medis: hubungi 119.")

with st.spinner("Menyiapkan model dan data (pertama kali bisa 1–2 menit)..."):
    bot = buat_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Tentang")
    st.write(f"Metode pencarian: **{'Sentence-BERT' if USE_SBERT else 'TF-IDF'}**")
    st.write(f"Jumlah data: **{len(bot.df)}** pasangan tanya-jawab")
    st.write("Sumber data: [agufsamudra/alodokter-qna](https://huggingface.co/datasets/agufsamudra/alodokter-qna)")
    if st.button("🗑️ Mulai percakapan baru"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.subheader("Contoh pertanyaan")
    for contoh in ["Anak demam tinggi sejak semalam", "Sakit kepala sebelah", "Jerawat muncul saat haid"]:
        if st.button(contoh):
            st.session_state.pending = contoh

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ketik pertanyaan kesehatan...")
if st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    jawaban = bot.get_response(prompt, st.session_state.history)
    st.session_state.messages.append({"role": "assistant", "content": jawaban})
    with st.chat_message("assistant"):
        st.markdown(jawaban)
