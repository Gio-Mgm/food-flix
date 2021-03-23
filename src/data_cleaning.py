import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# Data import
df = pd.read_csv("data/01_raw/en.openfoodfacts.org.products.tsv",
                 sep="\t", low_memory=False)

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
        'nutrition-score-fr_100g',
        'nutrition_grade_fr',
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

# Drop lines without product_name
df = df[df.product_name.notna()]

# ---- Duplicates Treatment ---- #

df.drop_duplicates(inplace=True)

# ---- Outliers treatment ---- #

# Max energy_100g can't exceed 3700Kj
df = df[df["energy_100g"] <= 3700]

# Max values per 100g can't exceed 100g

cols = df[[
    'fat_100g',
    'saturated-fat_100g',
    'carbohydrates_100g',
    'sugars_100g',
    'fiber_100g',
    'proteins_100g',
    'salt_100g',
]]

for col in cols:
    df = df[df[col] <= 100]

# Data export

df.to_csv("data/02_intermediate/foodflix.csv")
print("Done !")