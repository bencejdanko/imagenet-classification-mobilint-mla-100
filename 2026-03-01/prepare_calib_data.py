import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm

def prepare_calib_data(data_dir, output_dir, num_samples_per_class=5):
    """
    Prepares calibration data for Qubee compiler.
    Resizes, center crops, and saves images as .npy files in HWC layout.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Updated to direct Resize to 240x240 as requested
    preprocess = transforms.Compose([
        transforms.Resize((240, 240)),
    ])

    calib_paths = []
    
    class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Found {len(class_dirs)} classes. Collecting {num_samples_per_class} samples from each.")
    
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(data_dir, class_name)
        img_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Take a subset of images
        samples = img_files[:num_samples_per_class]
        
        for img_name in samples:
            img_path = os.path.join(class_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img_pre = preprocess(img)
                    
                    # Convert to numpy array and scale to [0, 1] (similar to transforms.ToTensor())
                    img_np = np.array(img_pre).astype(np.float32) / 255.0
                    
                    # Save as .npy
                    save_name = f"{class_name}_{Path(img_name).stem}.npy"
                    save_path = os.path.join(output_dir, save_name)
                    np.save(save_path, img_np)
                    calib_paths.append(os.path.abspath(save_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Create calib_data.txt
    txt_path = "calib_data.txt"
    with open(txt_path, "w") as f:
        for p in calib_paths:
            f.write(f"{p}\n")
            
    print(f"Done. Prepared {len(calib_paths)} calibration samples.")
    print(f"Calibration list saved to {txt_path}")

if __name__ == "__main__":
    DATA_DIR = "data/imagenet_train20a"
    OUTPUT_DIR = "calib_npy"
    prepare_calib_data(DATA_DIR, OUTPUT_DIR)
