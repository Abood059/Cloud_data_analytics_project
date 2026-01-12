from pymongo import MongoClient
import datetime
import ssl

class MongoHandler:
    def __init__(self, uri, db_name):
        self.client = MongoClient(
            uri, 
            tls=True,
            tlsAllowInvalidCertificates=True 
        )
        self.db = self.client[db_name]

    def save_stats(self, stats_data):
        collection = self.db["data_statistics"]
        stats_data["timestamp"] = datetime.datetime.utcnow()
        return collection.insert_one(stats_data)

    def save_benchmark_result(self, result):
        collection = self.db["ml_results"]
        result["timestamp"] = datetime.datetime.utcnow()
        return collection.insert_one(result)