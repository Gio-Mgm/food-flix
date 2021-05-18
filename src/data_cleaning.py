import pandas as pd
import numpy as np
from functions import reduce_mem_usage

# Data import
df = pd.read_csv("data/01_raw/en.openfoodfacts.org.products.tsv",
                 sep="\t", low_memory=False)


#--------------------------#
#----- Data selection -----#
#--------------------------#


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

#-----------------------------------#
#----- Missing values treatment ----#
#-----------------------------------#


# Drop lines without product_name
df = df[df.product_name.notna()]

#--------------------------------#
# ---- Duplicates Treatment ---- #
#--------------------------------#


df.drop_duplicates(inplace=True)


#----------------------------#
#---- Outliers treatment ----#
#----------------------------#


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

#-----------------------#
#---- Format values ----#
#-----------------------#

df['nutrition_grade_fr'] = np.where(
    df['nutrition_grade_fr'].str.isalpha(),
    df['nutrition_grade_fr'].str.upper(),
    df['nutrition_grade_fr'])


df.fillna("Non Renseigné", axis=1, inplace=True)
df["product_name"] = df["product_name"].str.strip().str.replace(
    "-", " ").str.lower().str.title()

df["brands"] = df["brands"].str.split(",", n=1, expand=True)
df["brands"] = df["brands"].str.strip().str.replace(
    "-", " ").str.lower().str.title()
df.drop_duplicates(subset=["product_name", "brands"],
                   keep='first', inplace=True)

df["allergens"] = df["allergens"].apply(
    lambda x: ", ".join(set(str(x).lower().split(', ')))
)

df['content'] = df[["product_name", "brands"]].astype(str).apply(lambda x: ' // '.join(x).lower(), axis=1)
df['content'].fillna('Null', inplace=True)

df["ingredients_text"] = df["ingredients_text"].apply(
    lambda x:
        x.replace(";", ",")
         .replace("&quot,", "")
         .replace("_", "")
         .replace("•", ",")
         .replace("-", ","))

#-----------------------#
#----- Data export -----#
#-----------------------#

df, NAlist = reduce_mem_usage(df)


df.to_csv("data/02_intermediate/foodflix.csv")
print("Done !")
