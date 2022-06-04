from modules.services import DatabaseConnection
import json
import requests

# data = [a for a in lt2 if 'language' in a['data'] and a['data']['language'] != 'en']
# text = [a['data']['full_text'] for a in data]

# with open('grams.json', 'w') as f:
#     text = json.dumps({
#         "something": "anything"
#     })

#     f.write(text)



db = DatabaseConnection('mongodb://localhost:27017')
lt = list(db.get_clean_tweets())
grams = [a['grams'] for a in lt]

URL = 'https://translation.googleapis.com/language/translate/v2/detect'
PARAMS = {
    "key": "AIzaSyC1GEMqJDIjaIsX0s_0E3hPIkw2XqOhbM4"
}

newdocs = []

for a in lt:
    newgrams = []
    doc = a['grams']

    for g in doc:
        data = {
            "q": g,
            "target": "en"
        }
        r = requests.post(url=URL, params=PARAMS, data=data)
        res = json.loads(r.text)
        
        detections = res['data']['detections'][0]
        highest = detections[0]

        if len(detections) > 1:
            for d in detections:
                if d['confidence'] > highest['confidence']:
                    highest = d
        
        if highest['language'] == 'en':
            newgrams.append(g)

    print(len(doc))
    a['grams'] = newgrams
    a['_id'] = str(a['_id'])
    print(len(newgrams))
    print('')
    
    newdocs.append(a)


with open('grams.json', 'w') as f:
    text = json.dumps(newdocs)
    f.write(text)