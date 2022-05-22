from modules.services import DatabaseConnection

db = DatabaseConnection('mongodb://localhost:27017')
table = db.conn['features']

cl = db.get_clean_tweets()

words = []

for a in cl:
    for gram in a['grams']:
        if gram not in words:
            words.append(gram)
            
            insert = {
                'name': gram
            }

            table.replace_one({ 'name': gram }, insert, upsert=True)


