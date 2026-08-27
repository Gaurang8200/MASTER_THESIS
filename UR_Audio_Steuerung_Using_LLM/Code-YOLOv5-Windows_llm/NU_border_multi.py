import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def apply_watershed(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area using dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Marking boundaries in red

    # Extracting border points, excluding edges near the image boundary
    height, width = image.shape[:2]
    margin = 10  # Margin to exclude near the image edges
    border_points = np.column_stack(np.where(markers == -1))
    filtered_border_points = [pt for pt in border_points if margin < pt[0] < height - margin and margin < pt[1] < width - margin]
    
    return image, filtered_border_points

def pick_random_points(points, num_points=500):
    if len(points) < num_points:
        print(f"Warning: Requested {num_points} points, but only {len(points)} are available.")
        num_points = len(points)
    return random.sample(points, num_points)

def get_crop_image_path():
    """
    Lese Crop-Image-Pfad: Versuche zuerst final_object_crop_path.txt,
    dann Fallback zu crop_img_path.txt, dann hardcoded path
    """
    # Try final crop path first (from precision detection filtering)
    final_crop_file = "txt_file/final_object_crop_path.txt"
    if os.path.exists(final_crop_file):
        with open(final_crop_file, "r") as file:
            crop_path = file.read().strip()
            if crop_path and os.path.exists(crop_path):
                print(f"BORDER: Using final crop path: {crop_path}")
                return crop_path
            else:
                print(f"BORDER WARNING: Final crop path does not exist: {crop_path}")
    
    # Fallback to standard crop path
    standard_crop_file = "txt_file/crop_img_path.txt"
    if os.path.exists(standard_crop_file):
        with open(standard_crop_file, "r") as file:
            crop_path = file.read().strip()
            if crop_path and os.path.exists(crop_path):
                print(f"BORDER: Using standard crop path: {crop_path}")
                return crop_path
            else:
                print(f"BORDER WARNING: Standard crop path does not exist: {crop_path}")
    
    # Final fallback to hardcoded path
    hardcoded_path = '/Users/vishaltala/Desktop/Thesis/Codes/Object-Detection-Using-YOLO-v5/yolov5/runs/detect/exp/crops/Cylinder/photo_1.jpg'
    print(f"BORDER: Using hardcoded fallback path: {hardcoded_path}")
    return hardcoded_path

# Load the image from the determined path
image_path = get_crop_image_path()
image = cv2.imread(image_path)

if image is None:
    print(f"BORDER ERROR: Could not load image from: {image_path}")
    exit(1)

# Apply watershed and get border points
result, border_points = apply_watershed(image)

# Pick 500 random points from the border
random_border_points = pick_random_points(border_points, 500)

# Draw the random points on the image
for point in random_border_points:
    cv2.circle(result, tuple(point[::-1]), 5, (0, 255, 255), -1)  # Draw in yellow

plt.subplot()
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Filtered Border Points')
plt.axis('off')
plt.show()

# Print the random points
print("Random Border Points:")
for point in random_border_points:
    formatted_points = ', '.join(['[' + ', '.join(map(str, point)) + ']' for point in random_border_points])
print(f"[{formatted_points}]")
