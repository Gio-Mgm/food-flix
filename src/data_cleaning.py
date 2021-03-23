import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# Data import
df = pd.read_csv("data/01_raw/en.openfoodfacts.org.products.tsv", sep="\t")

# ---- Data selection ---- #

#  Set df["countries"] to "France" for all France related product
df["countries"].str.replace(r".*(fr).*", "France", case=False, regex=True)

# Selection of rows about France and where product_name have a value
df = df[(df["countries"] == "France")]

# Selection of revelant columns
df = df[
    [
        'product_name',
        'brands',
        'categories',
        'ingredients_text',
        'allergens',
        'nutrition_score_fr',
        'nutrition_grade',
        'energy_100g',
        'fat_100g',
        'saturated-fat_100g',
        'carbohydrates_100g',
        'sugars_100g',
        'fiber_100g',
        'proteins_100g',
        'salt_100g'
    ]
]

# ---- Missing values treatment ---- #


# ---- Séléction des données ---- #

# ---- Séléction des données ---- #


# Data export

df.to_csv("../data/02_intermediate/foodflix.csv")
