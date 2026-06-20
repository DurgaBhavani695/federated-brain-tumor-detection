import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def generate_mri_slice(has_tumor=False, img_size=128):
    """
    Generates a synthetic 2D MRI brain slice.
    
    Parameters:
        has_tumor (bool): Whether to include a tumor in the brain slice.
        img_size (int): Resolution of the square image (default: 128x128).
        
    Returns:
        PIL.Image: Grayscale image of the simulated MRI slice.
    """
    # Create black background
    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    
    # Coordinates of center
    cx, cy = img_size // 2, img_size // 2
    
    # 1. Draw Skull (outer bone ellipse)
    skull_rx = int(img_size * 0.40)
    skull_ry = int(img_size * 0.46)
    draw.ellipse([cx - skull_rx, cy - skull_ry, cx + skull_rx, cy + skull_ry], fill=20, outline=120, width=2)
    
    # 2. Draw Brain Matter (cerebrum) - overlapping softer ellipses
    # Left hemisphere
    draw.ellipse([cx - int(skull_rx*0.9), cy - int(skull_ry*0.9), cx, cy + int(skull_ry*0.85)], fill=65)
    # Right hemisphere
    draw.ellipse([cx, cy - int(skull_rx*0.9), cx + int(skull_rx*0.9), cy + int(skull_ry*0.85)], fill=65)
    
    # Add minor details / lobes (frontal, occipital)
    draw.ellipse([cx - int(skull_rx*0.7), cy - int(skull_ry*0.85), cx + int(skull_rx*0.7), cy - int(skull_ry*0.3)], fill=80)
    draw.ellipse([cx - int(skull_rx*0.75), cy + int(skull_ry*0.3), cx + int(skull_rx*0.75), cy + int(skull_ry*0.8)], fill=55)
    
    # 3. Draw Ventricles (dark butterfly-shaped cavities in the center)
    vent_color = 15
    draw.ellipse([cx - 15, cy - 8, cx - 2, cy + 8], fill=vent_color)
    draw.ellipse([cx + 2, cy - 8, cx + 15, cy + 8], fill=vent_color)
    # Ventricle horns
    draw.ellipse([cx - 10, cy - 18, cx - 4, cy - 6], fill=vent_color)
    draw.ellipse([cx + 4, cy - 18, cx + 10, cy - 6], fill=vent_color)
    
    # Convert to numpy array to add detailed textures and tumor
    arr = np.array(img, dtype=np.float32)
    
    # Add subtle brain texture / gyri using noise
    y, x = np.ogrid[:img_size, :img_size]
    # Simple low-frequency sinusoids for brain tissue patterns
    texture = np.sin(x / 3.0) * np.cos(y / 3.0) * 8.0
    # Apply texture only where there is brain matter (pixel intensity > 25)
    brain_mask = arr > 25
    arr[brain_mask] += texture[brain_mask]
    
    # 4. Draw Tumor (if requested)
    if has_tumor:
        # Choose a random location inside the brain matter
        # Keep away from the extreme edges
        t_theta = random.uniform(0, 2 * np.pi)
        t_r = random.uniform(skull_rx * 0.15, skull_rx * 0.6)
        tx = int(cx + t_r * np.cos(t_theta))
        ty = int(cy + t_r * np.sin(t_theta))
        
        # Tumor radius (varying size)
        t_rx = random.randint(8, 16)
        t_ry = random.randint(8, 16)
        
        # Create a temporary mask for the tumor and its edema
        tumor_img = Image.new('L', (img_size, img_size), 0)
        tdraw = ImageDraw.Draw(tumor_img)
        
        # Draw Edema (swelling around tumor) - dim halo
        edema_rx = t_rx + random.randint(6, 12)
        edema_ry = t_ry + random.randint(6, 12)
        tdraw.ellipse([tx - edema_rx, ty - edema_ry, tx + edema_rx, ty + edema_ry], fill=40)
        
        # Draw Tumor Core - bright irregular mass
        tdraw.ellipse([tx - t_rx, ty - t_ry, tx + t_rx, ty + t_ry], fill=160)
        
        # Blur the tumor image to make it look organic
        tumor_img = tumor_img.filter(ImageFilter.GaussianBlur(radius=2.5))
        tumor_arr = np.array(tumor_img, dtype=np.float32)
        
        # Combine tumor with the brain
        # Max combination to ensure it sits on top of brain matter nicely
        arr = np.maximum(arr, tumor_arr)
        
    # Clip pixel values to valid range
    arr = np.clip(arr, 0, 255)
    
    # 5. Add MRI noise (Gaussian noise)
    noise_sigma = random.uniform(2.0, 5.0)
    noise = np.random.normal(0, noise_sigma, arr.shape)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)

def setup_federated_dataset(base_dir="data", mode="iid", num_train_per_client=100, num_val_per_client=20, num_test=50):
    """
    Sets up the directory structure and populates it with synthetic train, val, and test MRI slices.
    
    Parameters:
        base_dir (str): Base directory to store the data.
        mode (str): Data distribution mode - "iid" or "non_iid".
        num_train_per_client (int): Number of training images per client.
        num_val_per_client (int): Number of validation images per client.
        num_test (int): Number of global test images.
    """
    clients = ["hospital_a", "hospital_b", "hospital_c"]
    
    # Clean previous data if any
    import shutil
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    os.makedirs(os.path.join(base_dir, "test", "normal"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", "tumor"), exist_ok=True)
    
    # Generate test set (always 50% normal, 50% tumor)
    for i in range(num_test // 2):
        img_norm = generate_mri_slice(has_tumor=False)
        img_norm.save(os.path.join(base_dir, "test", "normal", f"test_norm_{i}.png"))
        
        img_t = generate_mri_slice(has_tumor=True)
        img_t.save(os.path.join(base_dir, "test", "tumor", f"test_tumor_{i}.png"))
        
    # Define tumor ratio per client based on distribution mode
    if mode == "iid":
        # Every client gets a balanced distribution
        ratios = {
            "hospital_a": 0.50,
            "hospital_b": 0.50,
            "hospital_c": 0.50
        }
    else:  # non_iid
        # Highly skewed distributions
        ratios = {
            "hospital_a": 0.15,  # Mostly Normal
            "hospital_b": 0.50,  # Balanced
            "hospital_c": 0.85   # Mostly Tumor
        }
        
    for client in clients:
        # Create directories
        for split in ["train", "val"]:
            os.makedirs(os.path.join(base_dir, client, split, "normal"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, client, split, "tumor"), exist_ok=True)
            
        ratio = ratios[client]
        
        # Generate training set
        num_tumor_train = int(num_train_per_client * ratio)
        num_normal_train = num_train_per_client - num_tumor_train
        
        for i in range(num_normal_train):
            img = generate_mri_slice(has_tumor=False)
            img.save(os.path.join(base_dir, client, "train", "normal", f"{client}_train_norm_{i}.png"))
            
        for i in range(num_tumor_train):
            img = generate_mri_slice(has_tumor=True)
            img.save(os.path.join(base_dir, client, "train", "tumor", f"{client}_train_tumor_{i}.png"))
            
        # Generate validation set (balanced validation to assess client capability fairly)
        num_tumor_val = num_val_per_client // 2
        num_normal_val = num_val_per_client - num_tumor_val
        
        for i in range(num_normal_val):
            img = generate_mri_slice(has_tumor=False)
            img.save(os.path.join(base_dir, client, "val", "normal", f"{client}_val_norm_{i}.png"))
            
        for i in range(num_tumor_val):
            img = generate_mri_slice(has_tumor=True)
            img.save(os.path.join(base_dir, client, "val", "tumor", f"{client}_val_tumor_{i}.png"))
            
    print(f"Dataset setup completed successfully with mode={mode}.")

if __name__ == "__main__":
    # Test generation
    setup_federated_dataset(base_dir="temp_test_data", mode="non_iid", num_train_per_client=10, num_val_per_client=4, num_test=6)
    print("Done generating test dataset.")
    import shutil
    shutil.rmtree("temp_test_data")
