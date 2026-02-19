### Usage: script extracts vessels from an image from Brightfield microscopy. 
### Combination of image processing techniques to enhance the vessels 
### Applies a thresholding method to segment them

# Import modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import black_tophat, disk, remove_small_objects, remove_small_holes, skeletonize

# Convert RGB to Vessel Contrast Channel
def extract_vessel_channel(img):
    if img.ndim == 3:
        # Use green channel (strongest Hb contrast)
        green = img[:,:,1]
        return green
    else:
        return img

# Enhance dark structures (vessels) using contrast normalization and inversion
def enhance_dark_structures(img):
    img = img.astype(np.float32)
    
    # Contrast normalization
    img = (img - img.min()) / (img.max() - img.min())
    
    # Invert → dark vessels become bright
    inverted = 1 - img
    
    # Slight Gaussian smoothing
    inverted = cv2.GaussianBlur(inverted, (5,5), 0)
    
    return inverted

# Top-Hat Filtering (Enhance Thin Dark Lines)
def dark_tophat_enhancement(img, radius=8):
    selem = disk(radius)
    enhanced = black_tophat(img, selem)
    return enhanced

# Thresholding (Otsu's method)
def threshold_image(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary

# Cleanup
def clean_binary(binary, min_size=200):
    binary = remove_small_objects(binary, min_size=min_size)
    binary = remove_small_holes(binary, area_threshold=100)
    return binary

# Skeletonization
def get_skeleton(binary):
    return skeletonize(binary)

image_path = "your_image.png"

img = load_image(image_path)

# Step 1: channel extraction
vessel_channel = extract_vessel_channel(img)

# Step 2: invert + normalize
inverted = enhance_dark_structures(vessel_channel)

# Step 3: optional black top-hat
enhanced = dark_tophat_enhancement(inverted, radius=10)

# Step 4: threshold
binary = threshold_image(enhanced)

# Step 5: cleanup
clean = clean_binary(binary, min_size=300)

# Step 6: skeleton
skel = get_skeleton(clean)

# --- Visualization ---
fig, axes = plt.subplots(1,6, figsize=(24,4))

axes[0].imshow(img)
axes[0].set_title("Original")

axes[1].imshow(vessel_channel, cmap="gray")
axes[1].set_title("Green Channel")

axes[2].imshow(inverted, cmap="gray")
axes[2].set_title("Inverted")

axes[3].imshow(enhanced, cmap="gray")
axes[3].set_title("Top-hat Enhanced")

axes[4].imshow(clean, cmap="gray")
axes[4].set_title("Binary Network")

axes[5].imshow(skel, cmap="gray")
axes[5].set_title("Skeleton")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
