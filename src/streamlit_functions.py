from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

def find_closest(tf, tfidf_matrix, df, query):
    input_matrix = tf.transform(query.split())

    cosine_similarities = linear_kernel(input_matrix, tfidf_matrix)
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    return [i for i in similar_indices]
