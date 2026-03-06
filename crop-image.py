#!/usr/bin/env python3

### Import necessary libraries
import cv2
import os

### Load image and select ROI
# Load image
img_path = os.path.abspath(os.sys.argv[1])
img = cv2.imread(img_path)
print("Reading image from path: ", img_path)

# Dynamically select ROI
x, y, w, h = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)

# Crop the image to the selected ROI
cropped_img = img[y:y+h, x:x+w]

cv2.imshow("CROPPED image", cropped_img)
cv2.destroyAllWindows()

### Save the cropped image
# Parse the original image name and create a new name for the cropped image
base = os.path.basename(img_path)        # e.g. "img2512101930-1.tif"
name, ext = os.path.splitext(base)         # name="img2512101930-1", ext=".tif"

output_name = f"data/{name}_roi.tif"            # "img2512101930-1_roi.tif"
print("Saving cropped image to: ", output_name)

cv2.imwrite(output_name, cropped_img)
