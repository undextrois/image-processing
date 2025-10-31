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


# Example usage:
# processor = ImageProcessor('path/to/your/image.jpg')
# processor.display_image()
# array = processor.to_numpy_array()
# print(f"Array shape: {array.shape}")
# processor.show_blue_channel()
# processor.show_all_channels()
