import gradio as gr
import plotly.express as px
import pandas as pd
import yaml
from processor import SparkProcessor

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

MONGO_URI = config['mongodb']['uri']
TARGET_ROWS = config.get('sample_rows', 2000000)
proc = SparkProcessor(MONGO_URI)

def refresh_history():
    return proc.get_historical_results()

def process_flow(source_type, file, algo):
    try:
        yield " Data is being prepared and the warehouse is being updated...", None, None, None, None
        
        stats_raw, cols = proc.prepare_data(source_type, file, n_rows=TARGET_ROWS)
       
        stats_df = stats_raw[stats_raw['index'].isin(['mean', 'stddev', 'std', 'min', 'max'])].rename(columns={'index': 'Stat'})
        
        results = []
        nodes_list = [1, 2, 4, 8]
        t1 = 0
        
        for n in nodes_list:
            yield f"Processing in progress on {n} nodes... ", stats_df, None, None, None
            duration = proc.run_ml_benchmark(algo, n)
            
            if n == 1: t1 = duration
            speedup = round(t1 / duration, 2) if duration > 0 else 1
            efficiency = f"{round((speedup/n)*100, 1)}%"
            
            res_row = {"algo": algo, "Nodes": n, "Time (s)": round(duration, 4), "Speedup": speedup, "Efficiency": efficiency, "rows": TARGET_ROWS}
            results.append(res_row)
            proc.handler.save_benchmark_result(res_row)

        perf_df = pd.DataFrame(results)
        fig = px.line(perf_df, x="Nodes", y="Time (s)", markers=True, title=f"Performance: {algo}")
        
        history_df = proc.get_historical_results()
        yield " The test is complete and the results are stored!", stats_df, perf_df, fig, history_df
        
    except Exception as e:
        yield f"❌ Erorr: {str(e)}", None, None, None, None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"Cloud ML Benchmark (Target: {TARGET_ROWS} Rows)")
    
    with gr.Tab("Analysis and processing"):
        with gr.Row():
            algo_opt = gr.Dropdown(["K-Means", "Bisecting K-Means", "Gaussian Mixture", "LDA"], label="Algorithm", value="K-Means")
            server_btn = gr.Button("Run Server Data", variant="primary")
            file_input = gr.File(label="Upload a file(CSV, Excel, ODS)", file_types=[".csv", ".xlsx", ".ods"])
            upload_btn = gr.Button("Run Uploaded File")
        
        status = gr.Textbox(label="Status")
        
        with gr.Row():
            out_stats = gr.Dataframe(label="Statistical data")
            out_perf = gr.Dataframe(label=" Current test results ")
        
        out_plot = gr.Plot(label="Performance curve")

    with gr.Tab("Recording results from the database"):
        refresh_btn = gr.Button("Updating the log from MongoDB")
        history_table = gr.Dataframe(label="Latest stored results", value=proc.get_historical_results())

    server_btn.click(process_flow, [gr.State("server"), gr.State(None), algo_opt], [status, out_stats, out_perf, out_plot, history_table])
    upload_btn.click(process_flow, [gr.State("upload"), file_input, algo_opt], [status, out_stats, out_perf, out_plot, history_table])
    refresh_btn.click(refresh_history, None, history_table)

if __name__ == "__main__":
    demo.launch(share=True)