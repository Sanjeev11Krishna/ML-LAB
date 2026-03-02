import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import string

print("Generating synthetic A-Z handwritten character dataset...")

# Generate 10,000 samples (enough for testing)
num_samples = 10000
image_size = 28 * 28  # 784 pixels

# Create lists to store data
data = []
labels = []

# Use different fonts to create variation
fonts_sizes = [20, 24, 28, 32]

for i in range(num_samples):
    # Random letter A-Z
    letter = random.choice(string.ascii_uppercase)
    labels.append(ord(letter) - 65)  # Convert to 0-25
    
    # Create blank image
    img = Image.new('L', (28, 28), color=255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font_size = random.choice(fonts_sizes)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Add some random position variation
    x_offset = random.randint(-2, 2)
    y_offset = random.randint(-2, 2)
    
    # Draw the character
    draw.text((14 + x_offset, 14 + y_offset), letter, fill=0, anchor="mm", font=font)
    
    # Convert to numpy array and flatten
    img_array = np.array(img).flatten()
    data.append(img_array)
    
    if (i + 1) % 1000 == 0:
        print(f"Generated {i + 1}/{num_samples} samples...")

# Create DataFrame
df = pd.DataFrame(data)
df.insert(0, 'label', labels)

# Save to CSV
output_file = "A_Z Handwritten Data.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset saved to: {output_file}")
print(f"Shape: {df.shape}")
print(f"Sample labels: {labels[:10]}")
