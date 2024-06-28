import os
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Define paths
input_folder = '/Users/srivantv/Downloads/archive (2)/matiz black'
output_folder = '/Users/srivantv/Downloads/archive (2)/matiz black_masks'
os.makedirs(output_folder, exist_ok=True)

# Load the SAM model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam_checkpoint = "/Users/srivantv/Downloads/archive (2)/matiz black_masks_weights.pth"  # Update this path to the location of your checkpoint
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Load image
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate masks
        masks = mask_generator.generate(image)

        # Save and visualize masks
        for i, mask in enumerate(masks):
            mask_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask_{i}.png")
            cv2.imwrite(mask_path, mask.astype('uint8') * 255)
            
            # Visualize the mask
            plt.figure(figsize=(10, 5))

            # Original image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Mask
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Generated Mask")
            plt.axis('off')

            plt.show()

print("Masks have been successfully generated, saved, and visualized.")
