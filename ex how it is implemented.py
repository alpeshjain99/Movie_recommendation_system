from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text =['India Jaipur India','Jaipur Jaipur India']

cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

count_matrix = count_matrix.toarray()

similarity_score = cosine_similarity(count_matrix)
