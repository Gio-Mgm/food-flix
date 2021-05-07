import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel

def find_closest(tf, tfidf_matrix, query):
    input_matrix = tf.transform(query.split())

    cosine_similarities = linear_kernel(input_matrix, tfidf_matrix)
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    return [i for i in similar_indices]


def get_allergens(series):
    s = pd.Series(", ".join(series).split(','))
    a = s.value_counts()
    return list(a[a > 100].index[1:])

def get_results(df, found):
    results = []
    for i in found:
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
        results.append([name, brand, nutriscore, allergens, ingredients, vals])
    return results
