from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import streamlit as st
from streamlit_CONST import STOPS
from streamlit_functions import find_closest, get_allergens, get_results

df = pd.read_csv("./data/02_intermediate/foodflix.csv", index_col=0)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                     min_df=0, stop_words=STOPS)

tfidf_matrix = tf.fit_transform(df['content'])

st.set_page_config(
    page_title="Foodflix", page_icon="./assets/f.jpg", layout='wide', initial_sidebar_state='expanded'
)

# --------------- #
# Title component #
# --------------- #

st.title("Moteur de recommandation basé sur le contenu")

# ----------------- #
# Sidebar component #
# ----------------- #

st.sidebar.image("./assets/foodlix.png", output_format='PNG')
user_input = st.sidebar.text_input('Que recherchez vous?').lower()

if user_input:
    allergens_filter = st.sidebar.multiselect(
        'Filtre allergènes', get_allergens(df["allergens"])
    )
# -------------- #
# Body component #
# -------------- #


if user_input:
    found = find_closest(tf, tfidf_matrix, user_input)
    results = get_results(df, found)

    for i in range(len(results)):
        st.header(results[i][0])
        st.subheader(f"Marque : {results[i][1]}")
        col1, col2 = st.beta_columns(2)
        with col1:
            st.text(f"Nutri-Score {results[i][2]}")
            st.text(f"Allergènes : {results[i][3]}")
            st.text("Ingrédients :")
            for el in results[i][4].replace(";", ",").replace("-", ",").split(','):
                st.text(f"- {el.strip()}")
        with col2:
            st.text("Valeurs énergétiques :")
            st.dataframe(results[i][5])
        st.markdown("_______")
        i+=1
