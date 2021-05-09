import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from streamlit_CONST import STOPS
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process

def fit_model(df, method):
    if method == "TF-IDF":
        model = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                min_df=0, stop_words=STOPS)
    elif method == "CountVectorizer":
        model = CountVectorizer(analyzer='word', ngram_range=(1, 2),
                            min_df=0, stop_words=STOPS)
    else:
        return "BERT"
    X = model.fit_transform(df['content'])
    return [model, X]

def find_closest(model, X, query):
    x = model.transform([query])
    cosine_similarities = linear_kernel(x, X)
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    return [i for i in similar_indices]

def find_fuzzy(x, series):
    return process.extractBests(x, series, limit=10)

def get_allergens(series):
    s = pd.Series(", ".join(series).split(','))
    a = s.value_counts()
    return list(a[a > 100].index[1:])

def get_results(df, found, short):
    results = []

    for i in found:
        name = df.iloc[i].product_name
        brand = df.iloc[i].brands
        nutriscore = df.iloc[i].nutrition_grade_fr
        if short:
            results.append([name, brand, nutriscore])
        else:
            allergens = df.iloc[i].allergens
            ingredients = df.iloc[i].ingredients_text
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
            results.append([name, brand, nutriscore, allergens, ingredients, vals])
    return results
