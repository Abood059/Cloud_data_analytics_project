import os
import zipfile

class DataIngestor:
    def __init__(self, dataset_id, download_path):
        self.dataset_id = dataset_id
        self.download_path = download_path

    def download_data(self):
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)
        
        target = os.path.join(self.download_path, "order_products__prior.csv")
        if os.path.exists(target):
            print("[*] Data already exists.")
            return

        print(f"[*] Downloading {self.dataset_id}...")
        os.system(f"kaggle datasets download -d {self.dataset_id} -p {self.download_path}")
        
        zip_file = os.path.join(self.download_path, f"{self.dataset_id.split('/')[-1]}.zip")
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(self.download_path)