# The center bias density has been obtained by averaging the eye fixation densities of all training images.

import os
import numpy as np
from PIL import Image

def convert_to_density_map(fixation_image):
    # Convert the fixation image to a density map by normalizing pixel values
    fixation_map = np.array(fixation_image)
    density_map = fixation_map / 255.0  # Normalize to [0, 1]
    return density_map

def aggregate_density_maps(density_maps):
    # Aggregate density maps by summing them
    aggregated_density_map = np.sum(density_maps, axis=0)
    aggregated_density_map /= np.sum(aggregated_density_map)  # Normalize
    return aggregated_density_map

# Example usage:
# Specify paths to the folder containing training images and fixation images
training_images_folder = "/Users/cimmykwok/Desktop/CV2/project/data/cv2_training_data/images/train" #"training_images"
fixation_images_folder = "/Users/cimmykwok/Desktop/CV2/project/data/cv2_training_data/fixations/train" #"fixation_images"

# Load training images and fixation images
training_image_paths = sorted(os.listdir(training_images_folder))
fixation_image_paths = sorted(os.listdir(fixation_images_folder))

# Ensure the number of training images matches the number of fixation images
assert len(training_image_paths) == len(fixation_image_paths)

# Initialize list to store density maps for each fixation image
density_maps = []

# Iterate over fixation images
for fixation_image_path in fixation_image_paths:
    # Load fixation image
    fixation_image = Image.open(os.path.join(fixation_images_folder, fixation_image_path))

    # Convert fixation image to density map
    density_map = convert_to_density_map(fixation_image)

    # Add density map to the list
    density_maps.append(density_map)

# Aggregate density maps for all fixation images
aggregated_density_map = aggregate_density_maps(density_maps)

# Save or use the aggregated density map as needed
np.save("center_bias_density.npy", aggregated_density_map)
