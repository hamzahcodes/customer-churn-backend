import pymongo

connection_string = 'mongodb+srv://hamzah:65290866@cluster0.t9ild.mongodb.net/test'
client = pymongo.MongoClient(connection_string)
db = client['customer']
collection = db['users']
