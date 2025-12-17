import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
