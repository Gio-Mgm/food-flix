from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import sys
import streamlit as st
from streamlit_functions import find_closest

df = pd.read_csv("./data/02_intermediate/foodflix.csv", index_col=0)
stops = "nan aux de à alors au aucuns aussi autre avant avec avoir bon car ce cela ces ceux chaque ci comme comment dans des du dedans dehors depuis devrait doit donc dos début elle elles en encore essai est et eu fait faites fois font hors ici il ils je juste la le les leur là ma maintenant mais mes mien moins mon mot même ni nommés notre nous ou où par parce pas peut peu plupart pour pourquoi quand que quel quelle quelles quels qui sa sans ses seulement si sien son sont sous soyez sujet sur ta tandis tellement tels tes ton tous tout trop très tu voient vont votre vous vu ça étaient état étions été être".split(" ")

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                     min_df=0, stop_words=stops)
tfidf_matrix = tf.fit_transform(df['content'])

st.title("Foodflix")

user_input = st.sidebar.text_input('Que recherchez vous?').lower()

allergens = pd.Series(", ".join(df["allergens"]).split(','))
a = allergens.value_counts()
a = a[a > 100].index[1:]

allergens_filter = st.sidebar.multiselect(
    'Filtre allergènes', list(a)
)

if user_input:
    finds = find_closest(tf, tfidf_matrix, df, user_input)
    for i in finds:
        vals = df.iloc[i][[
            "energy_100g", "fat_100g", "saturated-fat_100g",
            "carbohydrates_100g", "sugars_100g", "fiber_100g",
            "proteins_100g", "salt_100g"
        ]]
        rename = {
            "energy_100g": "énergie (en kj)",
            "fat_100g": "lipides",
            "saturated-fat_100g": "dont saturés",
            "carbohydrates_100g": "glucides",
            "sugars_100g": "dont sucres",
            "fiber_100g": "fibres",
            "proteins_100g": "protéines",
            "salt_100g": "sel",
        }
        vals = vals.rename(index=rename)
        name = df.iloc[i].product_name
        brand = df.iloc[i].brands
        nutriscore = df.iloc[i].nutrition_grade_fr
        allergens = df.iloc[i].allergens
        ingredients = df.iloc[i].ingredients_text

        st.markdown(f"# {name}")
        st.markdown(f"## Marque : {brand}")
        f"Nutri-Score {nutriscore}"
        f"Allergènes : {allergens}"
        f"Ingrédients : {ingredients}"

        "Valeurs énergétiques :"
        vals
        st.markdown("_______")
