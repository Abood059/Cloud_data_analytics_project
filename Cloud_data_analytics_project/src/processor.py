import os
import time
import gc
import pandas as pd
import datetime
import zipfile
import random
import kagglehub
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture, LDA

class MongoHandler:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
        self.db = self.client[db_name]

    def check_data_exists(self, collection_name):
        return self.db[collection_name].count_documents({})

    def save_benchmark_result(self, result):
        self.db["ml_results"].insert_one({**result, "timestamp": datetime.datetime.utcnow()})

class SparkProcessor:
    def __init__(self, mongo_uri):
        self.mongo_uri = mongo_uri
        self.handler = MongoHandler(self.mongo_uri, "SparkBenchmarkDB")
        self.dataset_id = "yasserh/instacart-online-grocery-basket-analysis-dataset"
        self.local_data_dir = os.path.join(os.getcwd(), "data") 
        self.input_cols = []

    def prepare_data(self, source_type, file_obj=None, n_rows=500000):
        collection = self.handler.db["active_training_data"]
        self.local_data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(self.local_data_dir, exist_ok=True)
        
        df_pd = None

        if source_type == "server":
            # تحديد اسم الملف الرئيسي المطلوب بناءً على صورتك
            target_file = "order_products__prior.csv" 
            csv_path = os.path.join(self.local_data_dir, target_file)
            
            # 1. إذا لم يكن الملف المطلوب موجوداً، نقوم بالتحميل
            if not os.path.exists(csv_path):
                print(f"[*] {target_file} not found. Downloading...")
                downloaded_path = kagglehub.dataset_download(self.dataset_id)
                for item in os.listdir(downloaded_path):
                    s = os.path.join(downloaded_path, item)
                    d = os.path.join(self.local_data_dir, item)
                    if os.path.isfile(s): shutil.copy2(s, d)

            # 2. التأكد مرة أخرى من وجود الملف بعد التحميل وقراءته
            if os.path.exists(csv_path):
                print(f"[*] Reading target file: {csv_path}")
                df_pd = pd.read_csv(csv_path, nrows=n_rows)
            else:
                # محاولة احتياطية: إذا لم يجد الملف المحدد، يقرأ أول ملف متاح
                all_csvs = [f for f in os.listdir(self.local_data_dir) if f.endswith('.csv')]
                if all_csvs:
                    csv_path = os.path.join(self.local_data_dir, all_csvs[0])
                    df_pd = pd.read_csv(csv_path, nrows=n_rows)
                else:
                    raise Exception("No CSV files found in data folder!")

        else:
            # معالجة الملف المرفوع يدوياً
            if file_obj is None: raise Exception("No file uploaded")
            # ... (نفس كود القراءة للملف المرفوع) ...
            df_pd = pd.read_csv(file_obj.name) # مثال بسيط

        # 3. المعالجة النهائية والإرجاع (حساس جداً لمنع خطأ NoneType)
        if df_pd is not None:
            # تنظيف الأسماء
            df_pd.columns = [str(c).strip().replace(' ', '_').replace('(', '').replace(')', '') for c in df_pd.columns]
            df_pd = df_pd.dropna()
            
            # تخزين في MongoDB
            collection.delete_many({})
            collection.insert_many(df_pd.to_dict(orient='records'))

            # اكتشاف الأعمدة الإحصائية
            self._auto_detect_cols(df_pd)
            
            # التأكد من إرجاع القيمتين المطلوبة لـ Gradio
            return df_pd.describe().reset_index(), self.input_cols
        
        raise Exception("Dataframe is empty or could not be loaded")

    def get_historical_results(self):
        results = list(self.handler.db["ml_results"].find({}, {'_id': 0}).sort("timestamp", -1).limit(10))
        if not results:
            return pd.DataFrame(columns=["algo", "Nodes", "Time (s)", "Speedup", "Efficiency", "rows"])
        return pd.DataFrame(results)

    def _auto_detect_cols(self, df):
        self.input_cols = [c for c in df.select_dtypes(include=['number']).columns 
                          if c.lower() not in ['_id', 'id'] and "noise" not in c.lower()]
        if len(self.input_cols) > 4: self.input_cols = self.input_cols[:4]

    def run_ml_benchmark(self, algo_name, n_nodes):
        spark = SparkSession.builder \
            .master(f"local[{n_nodes}]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.lang=ALL-UNNAMED") \
            .getOrCreate()
        
        try:
            data_list = list(self.handler.db["active_training_data"].find({}, {'_id': 0}))
            df_spark = spark.createDataFrame(pd.DataFrame(data_list))

            r_seed = random.randint(1, 20000000)
            df_spark = df_spark.withColumn("compute_noise", F.rand(seed=r_seed)).repartition(n_nodes)

            assembler = VectorAssembler(inputCols=self.input_cols, outputCol="features", handleInvalid="skip")
            data = assembler.transform(df_spark).select("features")
            data.count() 

            if "K-Means" in algo_name:
                algo = KMeans().setK(40).setMaxIter(80).setSeed(r_seed)
            elif "Bisecting K-Means" in algo_name:
                algo = BisectingKMeans().setK(30).setMaxIter(60).setSeed(r_seed)
            elif "Gaussian Mixture" in algo_name:
                algo = GaussianMixture().setK(10).setMaxIter(20).setSeed(r_seed)
            elif "LDA" in algo_name:
                data = data.sample(0.5)
                algo = LDA().setK(3).setMaxIter(2).setSeed(r_seed).setOptimizer("online")

            start = time.time()
            algo.fit(data) 
            duration = time.time() - start
            
            return duration
        finally:
            spark.stop()
            gc.collect()