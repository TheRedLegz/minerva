from modules.services import DatabaseConnection
from modules.vectorizer import generate_dbvector_matrix
from modules.som import SOM
import math


db = DatabaseConnection('mongodb://localhost:27017')


data = list(db.get_vectors())
uniq = list(db.get_features())
uniq = [a['name'] for a in uniq]

tfidf = generate_dbvector_matrix(data, uniq)

size = [4,4]
iterations = 3000
rate = 0.1

db.save_settings(size, iterations, rate)

row = size[0]
col = size[1]

checkpoint = math.floor(iterations/10)


def save_snapshot(matrix, step):
    if step%checkpoint == 0:
        db.save_snapshot(matrix, step, 1)
        
    
som = SOM(tfidf, rate, (row, col), iterations, save_snapshot)