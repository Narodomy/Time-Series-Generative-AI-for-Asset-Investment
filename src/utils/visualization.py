import matplotlib.pyplot as plt
import torch
import numpy as np

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