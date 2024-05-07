import os
import numpy as np
import imageio.v2 as imageio

def compute_fixation_density(image_dir, fixation_dir):
    image_files = os.listdir(image_dir)
    fixation_files = os.listdir(fixation_dir)

    # Ensure image and fixation files are sorted (assuming sorted order corresponds to matching pairs)
    image_files.sort()
    fixation_files.sort()

    # Initialize cumulative fixation density map
    cumulative_density_map = None

    for img_file, fix_file in zip(image_files, fixation_files):
        # Load image and fixation map
        img_path = os.path.join(image_dir, img_file)
        fix_path = os.path.join(fixation_dir, fix_file)

        image = imageio.imread(img_path)
        fixation_map = imageio.imread(fix_path)

        # Normalize fixation map (assuming fixation_map is in range [0, 255])
        normalized_fixation_map = fixation_map.astype(np.float32) / 255.0

        # Accumulate fixation map to cumulative density map
        if cumulative_density_map is None:
            cumulative_density_map = normalized_fixation_map
        else:
            cumulative_density_map += normalized_fixation_map

    # Normalize cumulative density map by the number of images
    num_images = len(image_files)
    average_density_map = cumulative_density_map / num_images

    return average_density_map

def save_center_bias_density(center_bias_density, save_path):
    # Save center bias density to .npy file
    np.save(save_path, center_bias_density)

# Example usage:
image_dir = 'data/training/images/'
fixation_dir = 'data/training/fixations/'
save_path = 'data/center_bias_density.npy'

# Compute fixation density map for all training images
center_bias_density_map = compute_fixation_density(image_dir, fixation_dir)

# Save center bias density map to .npy file
save_center_bias_density(center_bias_density_map, save_path)