from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import sys
import streamlit as st

df = pd.read_csv("./data/02_intermediate/foodflix.csv")

stops = "nan aux de à alors au aucuns aussi autre avant avec avoir bon car ce cela ces ceux chaque ci comme comment dans des du dedans dehors depuis devrait doit donc dos début elle elles en encore essai est et eu fait faites fois font hors ici il ils je juste la le les leur là ma maintenant mais mes mien moins mon mot même ni nommés notre nous ou où par parce pas peut peu plupart pour pourquoi quand que quel quelle quelles quels qui sa sans ses seulement si sien son sont sous soyez sujet sur ta tandis tellement tels tes ton tous tout trop très tu voient vont votre vous vu ça étaient état étions été être".split(" ")

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                     min_df=0, stop_words=stops)
tfidf_matrix = tf.fit_transform(df['content'])

st.title("Foodflix")

def find_closest():
    input_matrix = tf.transform(["chocolat"])

    cosine_similarities = linear_kernel(input_matrix, tfidf_matrix)
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    similar_items = [(cosine_similarities[0][i], df['Unnamed: 0']
                    [i], df['product_name'][i]) for i in similar_indices]

def find_item(search):
    return df[df['content'].str.contains(search)].index.tolist()


user_input = st.sidebar.text_input('Que recherchez vous?')
allergens_filter = st.sidebar.multiselect('Filtre allergènes', ["Lait", "Oeuf", "Soja"])
if user_input:
    found = find_item(user_input)

    res = df.loc[df.index.isin(found)]
    for i in range(len(found)):
        vals = res.iloc[i][[
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
            "fibers_100g": "fibres",
            "proteins_100g": "protéines",
            "salt_100g": "sel",
        }
        vals = vals.rename(index=rename)
        name = res.iloc[i].product_name
        brand = res.iloc[i].brands
        nutriscore = res.iloc[i].nutrition_grade_fr
        allergens = res.iloc[i].allergens
        ingredients = res.iloc[i].ingredients_text

        st.markdown(f"# {name}")
        st.markdown(f"## marque : {brand}")
        f"Nutri-Score {nutriscore.upper()}"
        f"allergènes : {allergens}"
        f"ingrédients : {ingredients}"

        "Valeurs énergétiques :"
        vals
        st.markdown("_______")
