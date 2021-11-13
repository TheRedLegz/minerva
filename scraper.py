import twint

c = twint.Config()
c.Near = 'Philippines'

# change keywords later
c.Search = '"online classes" OR "online class" OR "e-class" OR "online learning" OR "eclass" OR "face to face" OR "face-to-face" OR "lms" OR "distance learning" OR "online education" -filter:replies' 

c.Until = "2021-11-01"
c.Count = True
c.Filter_retweets = True
c.Store_json = True
c.Stats = True
c.Output = 'file2.json'

twint.run.Search(c)