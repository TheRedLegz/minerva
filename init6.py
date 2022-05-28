from modules.services import DatabaseConnection
db = DatabaseConnection('mongodb://localhost:27017')
from modules.lib import init_save_clean,filter_non_eng

raw = db.get_full_raw_tweets()
filt = filter_non_eng(raw)

init_save_clean(filt)