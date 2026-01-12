import yaml
from processor import SparkProcessor
from database import MongoHandler

def main():
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    db_handler = MongoHandler(config['mongodb']['uri'], config['mongodb']['db_name'])
    processor = SparkProcessor(config["app_name"])

    print("--- Step 1: Saving Descriptive Statistics to Cloud ---")
    stats = processor.load_data(config["data_path"], config["sample_rows"])
    db_handler.save_stats(stats)
    print(f"Successfully saved stats for {stats['total_rows']} rows.")

    print("\n--- Step 2: Running Distributed Benchmarks (LDA First) ---")
    ml_jobs = ["LDA", "KMeans", "BisectingKMeans", "GMM"]
    nodes_to_test = [1, 2, 4, 8]

    for job in ml_jobs:
        t1_time = None
        for n in nodes_to_test:
            print(f"\n>>> Testing {job} on {n} Nodes...")
            duration = processor.run_ml_benchmark(job, n)
            
            if n == 1: t1_time = duration
            
            speedup = round(t1_time / duration, 2) if duration > 0 else 1
            efficiency = round(speedup / n, 2)

            res_doc = {
                "algorithm": job,
                "nodes": n,
                "execution_time": round(duration, 2),
                "speedup": speedup,
                "efficiency": efficiency
            }
            db_handler.save_benchmark_result(res_doc)
            print(f"Success: {job} | Nodes: {n} | Time: {res_doc['execution_time']}s | Speedup: {speedup}")

if __name__ == "__main__":
    main()