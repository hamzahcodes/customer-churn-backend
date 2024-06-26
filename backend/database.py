import pymongo

connection_string = ''
client = pymongo.MongoClient(connection_string)
db = client['customer']
collection = db['users']
