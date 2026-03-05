import streamlit as st
import pandas as pd
import kagglehub
import os
from sklearn.feature_extraction.text import TfidfVectorizer # <--- BERUBAH: Pakai TF-IDF sesuai proposal
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 CineMatch")
st.write("Project Data Science: Content-Based Filtering menggunakan TF-IDF & Cosine Similarity")

# --- 2. LOAD DATA AUTOMATICALLY ---
@st.cache_data
def load_data_from_kaggle():
    try:
        with st.spinner('Sedang mendownload dataset dari Kaggle... (ini hanya sekali)'):
            # Download dataset MovieLens Latest Small
            path = kagglehub.dataset_download("grouplens/movielens-latest-small")
            
            # Cari file movies.csv di dalam folder hasil download
            file_path = os.path.join(path, "movies.csv")
            
            # Baca file CSV
            df = pd.read_csv(file_path)
            return df
    except Exception as e:
        st.error(f"Terjadi error saat download data: {e}")
        return None

# Panggil fungsi load data
movies = load_data_from_kaggle()

# Jika data gagal dimuat, hentikan aplikasi
if movies is None:
    st.stop()

# --- 3. BUILD MODEL (Content-Based) ---
# PERBAIKAN: Menggunakan TfidfVectorizer sesuai janji di Proposal Bab 3.4
# TF-IDF lebih baik daripada hitungan biasa karena memberikan bobot lebih pada genre yang unik.
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'), token_pattern=None)

try:
    # Mengubah teks genre menjadi matriks angka
    genre_matrix = tfidf.fit_transform(movies['genres'])
except ValueError:
    st.error("Format data genre tidak sesuai.")
    st.stop()

# Menghitung kemiripan (Cosine Similarity)
# Ini sesuai dengan rumus matematika di Proposal Bab 2.2.4
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# --- 4. RECOMMENDATION FUNCTION ---
def get_recommendations(movie_title, num_recommendations=5):
    try:
        # Cari index film berdasarkan judul
        idx = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return []

    # Ambil skor kemiripan film tersebut dengan semua film lain
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Urutkan dari yang paling mirip (score tertinggi) ke terendah
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Ambil top N film (skip index 0 karena itu film itu sendiri)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    movie_indices = [i[0] for i in sim_scores]
    
    # Kembalikan judul dan genre
    return movies.iloc[movie_indices][['title', 'genres']]

# --- 5. USER INTERFACE ---
# Dropdown menu
selected_movie = st.selectbox(
    "Pilih film yang kamu suka:",
    movies['title'].values
)

# Tombol Action
if st.button("Cari Rekomendasi"):
    # Tampilkan loading sebentar biar terasa canggih
    with st.spinner('Sedang menghitung kemiripan...'):
        recommendations = get_recommendations(selected_movie)
        
    if len(recommendations) > 0:
        st.success(f"Rekomendasi film mirip **{selected_movie}**:")
        # Tampilkan hasil
        for index, row in recommendations.iterrows():
            with st.container():
                st.subheader(row['title'])
                st.caption(f"Genre: {row['genres']}")
                st.markdown("---")
    else:
        st.warning("Maaf, tidak ada rekomendasi.")