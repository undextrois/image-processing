from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

class ImageProcessor:
    def __init__(self, image_path):
        """Initialize with an image file path"""
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.array = None
    
    def display_image(self):
        """Display the original image"""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.image)
        plt.axis('off')
        plt.title('Original Image')
        plt.show()
    
    def to_numpy_array(self):
        """Convert the image to a numpy array"""
        self.array = np.array(self.image)
        return self.array
    
    def isolate_blue_channel(self):
        """
        Set RED and GREEN channels to 0, isolating the BLUE channel.
        Uses slicing to modify the array.
        """
        # Convert to numpy array if not already done
        if self.array is None:
            self.to_numpy_array()
        
        # Create a copy to avoid modifying the original
        blue_only = self.array.copy()
        
        # Set RED channel (index 0) to 0
        blue_only[:, :, 0] = 0
        
        # Set GREEN channel (index 1) to 0
        blue_only[:, :, 1] = 0
        
        # BLUE channel (index 2) remains unchanged
        
        return blue_only
    
    def show_blue_channel(self):
        """Display the isolated blue channel using imshow()"""
        blue_only = self.isolate_blue_channel()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(blue_only)
        plt.axis('off')
        plt.title('Isolated Blue Channel')
        plt.show()
    
    def show_all_channels(self):
        """Display original image and isolated blue channel side by side"""
        blue_only = self.isolate_blue_channel()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(self.image)
        axes[0].axis('off')
        axes[0].set_title('Original Image')
        
        axes[1].imshow(blue_only)
        axes[1].axis('off')
        axes[1].set_title('Isolated Blue Channel')
        
        plt.tight_layout()
        plt.show()
    
    def scale_image_with_factors(self, fx=0.5, fy=0.5):
        """
        Scale the image using scaling factors.
        dim=(0,0) means the output size is calculated from fx and fy.
        """
        if self.array is None:
            self.to_numpy_array()
        
        # cv2.resize with dim=(0,0) and scaling factors
        scaled = cv2.resize(self.array, (0, 0), fx=fx, fy=fy)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(scaled)
        plt.axis('off')
        plt.title(f'Scaled Image (fx={fx}, fy={fy})')
        plt.show()
        
        return scaled
    
    def stretch_image(self, dim=(600, 600)):
        """
        Stretch the image to specified dimensions (width, height).
        Uses default interpolation (cv2.INTER_LINEAR).
        """
        if self.array is None:
            self.to_numpy_array()
        
        # cv2.resize expects (width, height)
        stretched = cv2.resize(self.array, dim)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(stretched)
        plt.axis('off')
        plt.title(f'Stretched Image {dim}')
        plt.show()
        
        return stretched
    
    def stretch_image_nearest(self, dim=(600, 600)):
        """
        Stretch the image to specified dimensions using INTER_NEAREST interpolation.
        This creates a pixelated effect.
        """
        if self.array is None:
            self.to_numpy_array()
        
        # cv2.resize with INTER_NEAREST interpolation
        stretched = cv2.resize(self.array, dim, interpolation=cv2.INTER_NEAREST)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(stretched)
        plt.axis('off')
        plt.title(f'Stretched Image {dim} (INTER_NEAREST)')
        plt.show()
        
        return stretched
    
    def detect_and_rotate_blue_box(self, rotation_angle=45):
        """
        Detect the blue box in the image, draw a bounding box around it,
        crop the region, and rotate it without clipping.
        """
        if self.array is None:
            self.to_numpy_array()
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(self.array, cv2.COLOR_RGB2BGR)
        
        # Convert to HSV for better blue detection
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Define range for blue color (adjust these values based on your blue)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No blue box detected!")
            return None
        
        # Get the largest contour (assuming it's the blue box)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw bounding box on original image
        img_with_box = self.array.copy()
        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop the region of interest
        roi = self.array[y:y+h, x:x+w].copy()
        
        # Get dimensions for rotation without clipping
        (h_roi, w_roi) = roi.shape[:2]
        center = (w_roi // 2, h_roi // 2)
        
        # Calculate new dimensions to avoid clipping
        angle_rad = np.radians(rotation_angle)
        new_w = int(abs(w_roi * np.cos(angle_rad)) + abs(h_roi * np.sin(angle_rad)))
        new_h = int(abs(w_roi * np.sin(angle_rad)) + abs(h_roi * np.cos(angle_rad)))
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Adjust translation to account for new dimensions
        M[0, 2] += (new_w - w_roi) / 2
        M[1, 2] += (new_h - h_roi) / 2
        
        # Rotate with new dimensions
        rotated = cv2.warpAffine(roi, M, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        
        # Display results
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(self.array)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Original Image')
        
        axes[0, 1].imshow(img_with_box)
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Detected Blue Box (ROI)')
        
        axes[1, 0].imshow(roi)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Cropped Region')
        
        axes[1, 1].imshow(rotated)
        axes[1, 1].axis('off')
        axes[1, 1].set_title(f'Rotated {rotation_angle}° (No Clipping)')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'bounding_box': (x, y, w, h),
            'cropped_roi': roi,
            'rotated': rotated
        }


# Example usage:
# processor = ImageProcessor('path/to/your/image.jpg')
# processor.display_image()
# array = processor.to_numpy_array()
# print(f"Array shape: {array.shape}")
# processor.show_blue_channel()
# processor.show_all_channels()
#
# # New scaling methods:
# processor.scale_image_with_factors(fx=0.5, fy=0.5)  # Scale to 50%
# processor.stretch_image(dim=(600, 600))  # Stretch with default interpolation
# processor.stretch_image_nearest(dim=(600, 600))  # Stretch with nearest neighbor
