import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf

def save_current_plot(filename, save_dir):
    """Function that helps to save current graph."""
    if save_dir:
        # แปลงเป็น Path object กันเหนียว
        save_path = Path(save_dir) / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

def plot_loss_comparison(history_dict, save_dir=None):
    """พล็อตกราฟ Loss รวมของทุกโมเดล"""
    plt.figure(figsize=(10, 6))
    for name, loss_data in history_dict.items():
        plt.plot(loss_data, label=name)
    
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        save_current_plot("loss_comparison.png", save_dir)
    
    plt.show()

def plot_series(x, label="Series", color="blue", title="Series"):
    plt.figure(figsize=(10, 5))
    plt.plot(x, label=label, color=color, alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_time_series(x_real, x_fake, model_name, feature_idx=0, save_dir=None):
    plt.figure(figsize=(10, 5))
    plt.plot(x_real[0, :, feature_idx], label="Real", color="blue", alpha=0.7)
    plt.plot(x_fake[0, :, feature_idx], label="Synthetic", color="red", alpha=0.7)
    plt.title(f"Time Series: Real vs {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        save_current_plot(f"{model_name}_timeseries.png", save_dir)
    
    plt.show()

def plot_projection(past, future, col=0, title="Series Projection"):
    plt.figure(figsize=(10, 5))
    """ Slice only column """
    y_past = past[:, col]
    y_future = future[:, col]

    """ Auto-generate X axis """
    x_past = np.arange(len(y_past))
    x_future = np.arange(len(y_past), len(y_past) + len(y_future))

    """ Plotting """
    plt.plot(x_past, y_past, label="Past", color="#1f77b4")

    plot_x_future = np.concatenate(([x_past[-1]], x_future))
    plot_y_future = np.concatenate(([y_past[-1]], y_future))

    plt.plot(plot_x_future, plot_y_future, label="Projection", color="#ff7f0e")
    # """ Connection line """
    # plt.plot([x_past[-1], x_future[0]], [y_past[-1], y_future[0]], color='gray', linestyle=':', alpha=0.5)
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)

def plot_distribution(x_real, x_fake, model_name, feature_idx=0, save_dir=None):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(x_real[:, :, feature_idx].flatten(), color="blue", label="Real", fill=True, alpha=0.3)
    sns.kdeplot(x_fake[:, :, feature_idx].flatten(), color="red", label="Synthetic", fill=True, alpha=0.3)
    plt.title(f"Distribution (KDE): Real vs {model_name}")
    plt.legend()
    
    if save_dir:
        save_current_plot(f"{model_name}_distribution.png", save_dir)
    
    plt.show()

def plot_pca(x_real, x_fake, model_name, save_dir=None):
    real_flat = x_real.reshape(x_real.shape[0], -1)
    fake_flat = x_fake.reshape(x_fake.shape[0], -1)
    
    pca = PCA(n_components=2)
    pca.fit(real_flat)
    r_pca = pca.transform(real_flat)
    f_pca = pca.transform(fake_flat)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(r_pca[:, 0], r_pca[:, 1], label="Real", alpha=0.5, color="blue")
    plt.scatter(f_pca[:, 0], f_pca[:, 1], label="Synthetic", alpha=0.5, color="red")
    plt.title(f"PCA: Real vs {model_name}")
    plt.legend()
    
    if save_dir:
        save_current_plot(f"{model_name}_pca.png", save_dir)
    
    plt.show()

def _compute_avg_acf(data, steps=20, feature_idx=0):
    acfs = []
    for i in range(len(data)):
        try:
            stat = acf(data[i, :, feature_idx], nlags=steps, fft=True)
            acfs.append(stat)
        except:
            pass # if there have errors
    return np.mean(acfs, axis=0) if acfs else np.zeros(steps+1)

def plot_acf(x_real, x_fake, model_name, save_dir=None):
    real_acf = _compute_avg_acf(x_real)
    fake_acf = _compute_avg_acf(x_fake)
    
    plt.figure(figsize=(8, 4))
    plt.plot(real_acf, label="Real ACF", marker='o', color="blue")
    plt.plot(fake_acf, label="Synthetic ACF", marker='x', linestyle='--', color="red")
    plt.title(f"ACF: Real vs {model_name}")
    plt.legend()
    plt.grid(True)
    
    if save_dir:
        save_current_plot(f"{model_name}_acf.png", save_dir)
        
    plt.show()

def visualize_all(x_real, x_fake, model_name, save_dir=None):
    """เรียกทีเดียว Save ครบทุกรูป โดยแปะชื่อ Model ไว้หน้าไฟล์"""
    print(f"Visualizing & Saving for {model_name}...")
    
    plot_time_series(x_real, x_fake, model_name, save_dir=save_dir)
    plot_distribution(x_real, x_fake, model_name, save_dir=save_dir)
    plot_pca(x_real, x_fake, model_name, save_dir=save_dir)
    plot_acf(x_real, x_fake, model_name, save_dir=save_dir)


def plot_monte_carlo(history, scenarios, feature_idx=0):
    """
    history: ข้อมูลอดีต (48, 2)
    scenarios: ผล Monte Carlo (50, 16, 2)
    """
    plt.figure(figsize=(10, 5))
    
    # 1. เตรียมข้อมูล
    hist_data = history[:, feature_idx]
    
    # สร้างแกน X
    hist_x = range(len(hist_data))
    pred_x = range(len(hist_data), len(hist_data) + scenarios.shape[1])
    
    # เตรียมพิกัดจุดเชื่อม (จุดสุดท้ายของ History)
    last_hist_x = hist_x[-1]
    last_hist_y = hist_data[-1]
    first_pred_x = pred_x[0] # จุดเริ่มของ Forecast
    
    # 2. วาดเส้น Monte Carlo และเส้นเชื่อม
    for i in range(scenarios.shape[0]):
        # 2.1 วาดเส้น Forecast (สีแดงจางๆ เหมือนเดิม)
        plt.plot(pred_x, scenarios[i, :, feature_idx], color='red', alpha=0.3)
        
        # 2.2 [ส่วนที่เพิ่ม] วาดเส้นเชื่อมสีเทา (Bridge)
        # ลากจาก (จุดท้าย History) -> (จุดแรกของ Forecast เส้นนี้)
        first_scen_y = scenarios[i, 0, feature_idx]
        plt.plot([last_hist_x, first_pred_x], [last_hist_y, first_scen_y], color='red', alpha=0.1)
        
    # 3. วาดเส้น History (ของจริง)
    plt.plot(hist_x, hist_data, label="History", color='blue', linewidth=2)
    
    # (Optional) วาดเส้นค่าเฉลี่ย
    mean_scenario = np.mean(scenarios[:, :, feature_idx], axis=0)
    plt.plot(pred_x, mean_scenario, label="Mean Forecast", color='darkred', linestyle='--', linewidth=2)
    
    # เชื่อมเส้นค่าเฉลี่ยด้วย (เพื่อให้ดูเนียนเหมือนกัน)
    plt.plot([last_hist_x, first_pred_x], [last_hist_y, mean_scenario[0]], color='gray', linestyle='--', alpha=0.5)

    plt.title(f"Monte Carlo Simulation (Feature {feature_idx})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_comparison(dataloader, model, diffusion, scaler, dataset, device="cuda"):
    """
    ฟังก์ชันเปรียบเทียบ Real vs Fake (รองรับ Dataset แบบ Dictionary)
    """
    model.eval()
    
    # -------------------------------------------------------
    # 1. 📥 ดึงข้อมูลจริง (Real)
    # -------------------------------------------------------
    # ดึงมา 1 Batch
    batch = next(iter(dataloader)) 
    
    # ดึงข้อมูลดิบ (x_raw) มาเลย! ไม่ต้อง Inverse Transform ให้ยุ่งยาก
    # shape: [B, L, C] -> เลือกตัวอย่างแรก [0] -> [L, C]
    real_sample_raw = batch['x_raw'][0].cpu().numpy()

    # -------------------------------------------------------
    # 2. 🤖 เสกข้อมูลปลอม (Fake)
    # -------------------------------------------------------
    print(f"Generating synthetic data...")
    
    # สร้าง 1 ตัวอย่าง
    # fake_scaled shape: [1, L, C] (ค่าจะเป็น -1 ถึง 1)
    fake_scaled = diffusion.sample(model, n=1) 
    
    # ดึงออกมาเป็น numpy [L, C]
    fake_sample_scaled = fake_scaled[0].cpu().numpy()

    # -------------------------------------------------------
    # 3. 🔄 แปลงร่างข้อมูลปลอม (Inverse Scale)
    # -------------------------------------------------------
    
    # ต้องเช็คว่า Scaler ถูก Fit มาหรือยัง
    if scaler is not None:
        fake_sample_raw = scaler.inverse_transform(fake_sample_scaled)
    else:
        # กรณีไม่ได้ Scale (scale=False) ก็ใช้ค่าเดิมเลย
        fake_sample_raw = fake_sample_scaled

    # -------------------------------------------------------
    # 4. 📈 พล็อตกราฟ (Dynamic Layout)
    # -------------------------------------------------------
    features = dataset.features # ["Open", "Close", ...]
    num_features = len(features)
    
    # สร้าง Subplots แนวตั้ง
    fig, axs = plt.subplots(num_features, 1, figsize=(12, 4 * num_features), sharex=True)
    
    # กรณีมี Feature เดียว (เช่นเทรนแค่ Close) ให้แปลง axs เป็น list
    if num_features == 1:
        axs = [axs]

    for i, name in enumerate(features):
        ax = axs[i]
        
        # กราฟจริง (สีน้ำเงิน) - จาก x_raw
        ax.plot(real_sample_raw[:, i], label='Real Data (Raw)', color='dodgerblue', alpha=0.8, linewidth=2)
        
        # กราฟปลอม (สีส้มแดง) - จาก AI Gen
        ax.plot(fake_sample_raw[:, i], label='AI Generated', color='orangered', linestyle='--', alpha=0.9, linewidth=2)
        
        ax.set_title(f"Feature: {name}", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
    model.train() # อย่าลืมสับสวิตช์กลับเป็น Train เผื่อเรียกใช้ระหว่างเทรน


# ==========================================
# Helper Functions
# ==========================================

def _get_display_info(dataset, feature):
    """ช่วยเช็คว่าเปิดโหมด Log Return ไหม ถ้าใช่ให้เปลี่ยนชื่อและคืนค่า Flag"""
    use_log = getattr(dataset, 'use_log_return', False)
    log_feats = getattr(dataset, 'log_return_features', None)
    
    # ถ้า log_return_features เป็น None แล้ว use_log=True แปลว่าทำทุกตัว หรือทำ Close เป็น Default
    # แต่ใน code Dataset คุณ default คือ ['Close'] ดังนั้นเช็คให้ชัวร์
    if use_log:
        if log_feats is None:
            is_log = (feature == 'Close') # Default behavior
        else:
            is_log = (feature in log_feats)
    else:
        is_log = False
    
    name = f"{feature} (Log Return)" if is_log else feature
    return name, is_log

def _prepare_single_data(dataset, feature):
    """ดึงข้อมูลดิบและวันที่จาก TimeSeriesDataset"""
    if feature not in dataset.features:
        print(f"Error: Feature {feature} not found.")
        return None, None
        
    feat_idx = dataset.features.index(feature)
    values = dataset.raw_values[:, feat_idx]
    
    if hasattr(dataset, 'dates'):
        dates = pd.to_datetime(dataset.dates)
    else:
        dates = np.arange(len(values))
        
    return dates, values

# ==========================================
# 1. Single Asset Functions (TimeSeriesDataset)
# ==========================================

def viz_single_timeline(dataset, start_idx, feature):
    """Plot ยาวจนจบ (Start -> End)"""
    dates, values = _prepare_single_data(dataset, feature)
    if dates is None: return

    # ตัดข้อมูลตั้งแต่ start_idx จนจบ
    plot_dates = dates[start_idx:]
    plot_values = values[start_idx:]
    
    label_name, is_log = _get_display_info(dataset, feature)

    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, plot_values, label=label_name, linewidth=1.5)
    
    plt.title(f"Timeline View: {label_name} (Start Idx: {start_idx})")
    plt.ylabel(label_name)
    plt.grid(True, alpha=0.3)
    if is_log: plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    
    # Format Date
    if len(plot_dates) > 0 and isinstance(plot_dates[0], pd.Timestamp):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
    plt.legend()
    plt.show()

def viz_single_window(dataset, start_idx, feature):
    """Plot เฉพาะใน Window (Start -> Start + Window)"""
    dates, values = _prepare_single_data(dataset, feature)
    if dates is None: return
    
    window_size = dataset.window_size
    # Logic Forward: เริ่มที่ start_idx ไปอีก window_size
    end_idx = start_idx + window_size
    
    # ป้องกัน Index เกิน
    actual_end_idx = min(end_idx, len(values))
    
    plot_dates = dates[start_idx : actual_end_idx]
    plot_values = values[start_idx : actual_end_idx]
    
    label_name, is_log = _get_display_info(dataset, feature)

    plt.figure(figsize=(10, 5))
    plt.plot(plot_dates, plot_values, label=label_name, marker='.', markersize=5)
    
    # Fix แกน X ให้โชว์วันที่แบบ Forward
    if len(plot_dates) > 0 and isinstance(plot_dates[0], pd.Timestamp):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

    plt.title(f"Window View ({window_size} steps): {label_name} | Start Idx: {start_idx}")
    plt.ylabel(label_name)
    plt.grid(True, alpha=0.3)
    if is_log: plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.show()


# ==========================================
# 2. Portfolio/Group Functions (PortfolioDataset)
# ==========================================

def viz_group_timeline(portfolio_ds, start_idx, feature):
    """Plot ทุกตัวพร้อมกัน แบบ Timeline ยาว (Start Date -> End)"""
    # 1. หาวันที่เริ่มต้นจาก Basket Index
    try:
        basket_data = portfolio_ds.security_basket_dataset[start_idx]
        start_date = pd.to_datetime(basket_data['date'])
    except IndexError:
        print("Error: Start Index out of range")
        return

    label_name, is_log = _get_display_info(portfolio_ds, feature)
    
    plt.figure(figsize=(14, 7))
    count = 0
    
    # 2. วนลูปทุก Asset
    for symbol, ts_ds in portfolio_ds.asset_datasets.items():
        if feature not in ts_ds.features: continue
            
        dates, values = _prepare_single_data(ts_ds, feature)
        
        # Filter เอาเฉพาะวันที่ >= start_date (เดินหน้า)
        mask = dates >= start_date
        if not np.any(mask): continue
            
        plt.plot(dates[mask], values[mask], label=symbol, alpha=0.6, linewidth=1)
        count += 1
        
    plt.title(f"Group Timeline: {label_name} | Start Date: {start_date.date()} | N={count}")
    plt.ylabel(label_name)
    plt.grid(True, alpha=0.3)
    if is_log: plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    if count <= 15: plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def viz_group_window(portfolio_ds, index, feature):
    """Plot ทุกตัวพร้อมกัน เฉพาะ Window (Forward Looking)"""
    
    # 1. หาวันที่ "เริ่มต้น" (Anchor Date)
    try:
        basket_data = portfolio_ds.security_basket_dataset[index]
        start_date = pd.to_datetime(basket_data['date'])
    except IndexError:
        print("Error: Start Index out of range")
        return

    window_size = portfolio_ds.window_size
    label_name, is_log = _get_display_info(portfolio_ds, feature)
    
    plt.figure(figsize=(12, 6))
    count = 0
    
    # 2. วนลูป Asset
    for symbol, ts_ds in portfolio_ds.asset_datasets.items():
        if feature not in ts_ds.features: continue
        
        # หาตำแหน่งวันที่ "เริ่มต้น" ใน Asset นั้นๆ
        if start_date not in portfolio_ds.asset_date_maps[symbol]:
            continue
            
        row_idx = portfolio_ds.asset_date_maps[symbol][start_date]
        
        # --- [CRITICAL FIX] Forward Looking Logic ---
        # Start = row_idx
        # End   = row_idx + window_size
        s_idx = row_idx
        e_idx = row_idx + window_size
        
        dates, values = _prepare_single_data(ts_ds, feature)
        
        # เช็คว่ามีข้อมูลพอไหม (ถ้าไม่พอ ตัดเท่าที่มี)
        if s_idx >= len(dates): continue
        actual_e_idx = min(e_idx, len(dates))

        plot_dates = dates[s_idx : actual_e_idx]
        plot_values = values[s_idx : actual_e_idx]
        
        plt.plot(plot_dates, plot_values, label=symbol, alpha=0.8, marker='.')
        count += 1

    plt.title(f"Group Window View ({window_size} days): {label_name} | Start Date: {start_date.date()}")
    plt.ylabel(label_name)
    plt.grid(True, alpha=0.3)
    if is_log: plt.axhline(0, color='black', linestyle='--')
    
    # Format Date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    if count <= 15: plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()