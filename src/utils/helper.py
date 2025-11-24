import os
import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def save_time_series_plots(sampled_data, path, num_plots=4, channels_to_plot=[0, 1], figsize=(12, 8)):
    """
    บันทึก Time Series ตัวอย่างที่ Diffusion Model สร้างขึ้นมาเป็นรูปกราฟ PNG.

    Args:
        sampled_data (torch.Tensor): Tensor ของ Time Series ที่ Sampling มา, รูปร่าง [B, C, L]
        path (str): Path เต็มของไฟล์ที่จะบันทึก (เช่น "results/run_name/epoch_0.png")
        num_plots (int): จำนวน Time Series ตัวอย่างจาก Batch ที่จะนำมาพล็อต (สูงสุดไม่เกิน B)
        channels_to_plot (list): รายชื่อ Index ของ Feature/Channel ที่ต้องการพล็อต
                                 เช่น [0, 1] จะพล็อต Feature ที่ 0 และ 1
        figsize (tuple): ขนาดของ Figure (กว้าง, สูง)
    """
    
    # 1. ย้ายข้อมูลไป CPU และแปลงเป็น NumPy array
    #    (matplotlib ไม่ทำงานกับ CUDA tensor)
    data_np = sampled_data.cpu().numpy() # [B, C, L]
    
    # 2. ตรวจสอบให้แน่ใจว่าจำนวน plots ไม่เกินขนาด Batch
    num_plots = min(num_plots, data_np.shape[0])
    
    # 3. สร้าง Figure และ Axes สำหรับ Subplots
    #    (เราจะพล็อต num_plots แถว x len(channels_to_plot) คอลัมน์)
    fig, axes = plt.subplots(num_plots, len(channels_to_plot), figsize=figsize, squeeze=False)
    
    # 4. Loop เพื่อพล็อตแต่ละ Time Series และแต่ละ Channel
    for i in range(num_plots): # Loop ผ่านแต่ละตัวอย่างใน Batch
        for j, channel_idx in enumerate(channels_to_plot): # Loop ผ่านแต่ละ Channel ที่ต้องการ
            
            # 4a. เลือก subplot ที่ถูกต้อง
            ax = axes[i, j]
            
            # 4b. พล็อต Time Series ของตัวอย่างที่ i, Channel ที่ channel_idx
            ax.plot(data_np[i, channel_idx, :]) # data_np[Batch_idx, Channel_idx, Time_steps]
            
            # 4c. ตั้งชื่อกราฟ
            ax.set_title(f"Sample {i+1}, Feature {channel_idx}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle='--', alpha=0.6) # เพิ่ม Grid
            
    # 5. ปรับ Layout และบันทึก Figure
    plt.tight_layout() # ปรับระยะห่างของ Subplot ให้สวยงาม
    plt.savefig(path)  # บันทึกเป็นไฟล์ PNG
    plt.close(fig)     # ปิด Figure เพื่อเคลียร์ Memory
    
    # logging.info(f"Saved {num_plots} time series plots to {path}")

def save_loss_plot(loss_history, path, total_epochs, y_axis_max, y_axis_min, figsize=(10, 5)):
    """
    พล็อต "ประวัติ" (History) ของ Loss...
    ... "พร้อม" (With) "Fix" (Fixed) "แกน" (Axes) ... สำหรับ "วิดีโอ" (Videos)!
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. พล็อต "ข้อมูล" (Data)
    #    (เรา "ยัง" (Still) พล็อต "แค่" (Only) 'loss_history' (ที่ "โต" (Grows) ขึ้นเรื่อยๆ))
    ax.plot(loss_history, marker='o', linestyle='--') 
    
    # 2. แต่งกราฟ (Decorate)
    ax.set_title("Average Epoch Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- [THE FIX - ที่คุณ "เตือน" (Warned) ผม!] ---
    
    # 3. "Fix" (Fix) "แกน X" (X-axis)
    #    (เราจะ "ปรับ" (Adjust) "ขอบ" (Limits) ให้ "เข้ากับ" (Fit) "จำนวน" (Number) "Epochs ที่มี" (Available)!)
    
    # "จำนวน" (Number) Epoch ที่ "พล็อต" (Plotted) จริงๆ... คือ 'len(loss_history)'
    current_epochs = len(loss_history) - 1 # (เพราะ Epoch เริ่มที่ 0)
    
    # "ถ้า" (If) มี "น้อยกว่า" (Less than) 5 Epoch... เราจะ "ขยาย" (Expand) แกน X ... "ให้เห็น" (To show) "ได้ถึง" (Up to) 5 Epochs
    # "ถ้า" (If) มี "มากกว่า" (More than) 5 Epoch... เราจะ "ขยาย" (Expand) แกน X ... "ให้เห็น" (To show) "ได้ถึง" (Up to) 'total_epochs'
    
    # max_x_display = max(5, total_epochs) # (คุณอาจจะอยากได้แบบนี้... ถ้าอยาก "เห็น" (See) ถึง 500 เลย)
    # แต่คุณอยาก "ซูม" (Zoom in)... งั้นเรา "จะ" (Will) "ขยาย" (Expand) "แกน" (Axis) ... "ทีละนิด" (Little by little) ครับ!
    
    # "ขอบเขต" (Limit) "สูงสุด" (Max) ของ "แกน X" (X-axis) ... "จะ" (Will) "ค่อยๆ" (Gradually) "เพิ่ม" (Increase) ขึ้นเรื่อยๆ
    # ... โดยที่ "อย่างน้อย" (At least) 5 Epochs ... และ "อย่างมาก" (At most) 'total_epochs'
    dynamic_x_max = max(5, current_epochs + 1) # (ให้เห็นถึง Epoch ปัจจุบัน + 1)
    
    ax.set_xlim(-0.5, dynamic_x_max + 0.5)
    
    # 4. "Fix" (Fix) "แกน Y" (Y-axis)
    ax.set_yscale('log') # (Log scale "ยัง" (Still) "ดี" (Good))
    ax.set_ylim(y_axis_min, y_axis_max) # <-- "ท่าไม้ตาย" (THE FIX)!
    
    # ---------------------------------------------

    # 5. เซฟ!
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig) # (สำคัญ! เคลียร์ Memory)