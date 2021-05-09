import pandas as pd
import streamlit as st
from streamlit_CONST import WARN_INPUT_NOT_FOUND
from streamlit_functions import find_closest, get_allergens, get_results, fit_model, find_fuzzy
from fuzzywuzzy import fuzz

df = pd.read_csv("./data/02_intermediate/foodflix.csv", index_col=0)

st.set_page_config(
    page_title="Foodflix",
    page_icon="./assets/f.jpg",
    layout='wide',
    initial_sidebar_state='expanded'
)

# --------------- #
# Title component #
# --------------- #

st.title("Moteur de recommandation basé sur le contenu")

# ----------------- #
# Sidebar component #
# ----------------- #

st.sidebar.image("./assets/foodlix.png", output_format='PNG')

method = st.sidebar.radio("Sélection de la méthode : ", ("TF-IDF", "CountVectorizer", "BERT"))

model, X = fit_model(df, method)

user_input = st.sidebar.text_input('Que recherchez vous?').lower().capitalize()
short = st.sidebar.checkbox("Affichage simplifié", value=True)

if user_input and not short:
    allergens_filter = st.sidebar.multiselect(
        'Filtre allergènes', get_allergens(df["allergens"])
    )
# -------------- #
# Body component #
# -------------- #
if user_input:
    results_are_shown = True
    empty = st.empty()
    if df["product_name"].to_string().find(user_input) == -1 or df["brands"].to_string().find(user_input) == -1:
        with empty.beta_container():
            results_are_shown = False
            fuzzies = find_fuzzy(user_input, df["product_name"].to_list())
            st.warning(WARN_INPUT_NOT_FOUND.format(user_input))
            choices = [fuzzy[0] for fuzzy in fuzzies]
            choices.insert(0, "")
            choices = list(set(choices))
            choices.sort()
            radio = st.radio("", choices)
            if radio != "":
                user_input = radio
                results_are_shown = True
                empty.empty()
                print(user_input, type(user_input))

    if results_are_shown:
        found = find_closest(model, X, user_input)
        results = get_results(df, found, short)

        for _, el in enumerate(results):

            if short:
                st.subheader(f"{el[0]} - {el[1]}")
                st.markdown(f"*_Nutri-Score_* {el[2]}")
            else:
                st.header(el[0])
                st.subheader(f"Marque : {el[1]}")
                col1, col2 = st.beta_columns(2)
                with col1:
                    st.markdown(f"**_Nutri-Score {el[2]}_**")
                    st.markdown(f"**_Allergènes: {el[3]}_**")
                    for el1 in el[4].split(","):
                        st.text(f"- {el1.strip()}")
                with col2:
                    st.text("Valeurs énergétiques :")
                    st.dataframe(el[5])
            st.markdown("_______")
