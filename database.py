import datetime
import pprint

from pymongo import MongoClient

client = MongoClient()
client = MongoClient("localhost", 27017)

db = client["vra-dataset"]


def insert_doc(data):
    print(data)
    try:
        our_data = {
            "name": data.name,
            "index_name": data.index_name,
            "index": data.index
        }
        post_id = db.posts.insert_one(our_data).inserted_id
        print("post_id ",post_id)
        return True
    except Exception as e:
        return False



pprint.pprint(db.posts.find_one({"index_name": "first"}))