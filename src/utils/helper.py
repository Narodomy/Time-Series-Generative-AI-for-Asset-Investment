import torch
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

def inspect(data, name: str = ""):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    _min = np.min(data)
    _max = np.max(data)
    _mean = np.mean(data)
    _std = np.std(data)
    
    print(f"--- Inspecting: {name} ---")
    print("-" * 36)
    print(f"Shape: {data.shape}")
    print(f"Min:   {_min:.4f}")
    print(f"Max:   {_max:.4f}")
    print(f"Mean:  {_mean:.4f}")
    print(f"Std:   {_std:.4f}")
    print("-" * 36)
    return _min, _max, _mean, _std


def inverse_log_returns(r_log: np.ndarray) -> np.ndarray:
    # R_simple = e^(R_log) - 1
    r_simple = np.exp(r_log) - 1

    return r_simple



def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def save_as_html(report: dict, filename: str):
    """
    Requires
    {
        "metrics": {"Mean": 0.5, "Sharpe": 1.2, ...},
        "plots": {"Cumulative Return": "base64...", "Drawdown": "base64..."}
    }
    """
    metrics_html = ""
    if "metrics" in report and report["metrics"]:
        rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in report["metrics"].items()])
        metrics_html = f"""
        <h3>Metrics</h3>
        <table border="1" style="border-collapse: collapse; width: 50%;">
            <tr><th>Metric</th><th>Value</th></tr>
            {rows}
        </table>
        """

    plots_html = ""
    if "plots" in report and report["plots"]:
        for title, img_str in report["plots"].items():
            plots_html += f"""
            <div style="margin-bottom: 20px;">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{img_str}" style="max-width: 100%; border: 1px solid #ccc;">
            </div>
            """
            
    html = f"""
    <html>
    <head>
        <title>Simulation Report</title>
        <style>body {{ font-family: sans-serif; padding: 20px; }} h3 {{ color: #444; }}</style>
    </head>
    <body>
        <h1>Inspection Report</h1>
        <p><strong>ID:</strong> {report.get('id', 'N/A')}</p>
        <hr>
        {metrics_html}
        <hr>
        {plots_html}
    </body>
    </html>
    """
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)