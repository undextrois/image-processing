import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.image_rgb = None
        self.image_hsv = None
        self.shape = None
        self.size = None
    
    def read_image(self):
        """Read the image from the specified path"""
        try:
            # Read image in BGR format (OpenCV default)
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise ValueError("Could not read the image. Please check the file path.")
            
            # Convert BGR to RGB for display
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            # Convert to HSV color space
            self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            return True
        except Exception as e:
            print(f"Error reading image: {e}")
            return False
    
    def analyze_image(self):
        """Analyze the image shape and size"""
        if self.image is not None:
            self.shape = self.image.shape
            self.size = self.image.size
            return True
        else:
            print("No image loaded. Please call read_image() first.")
            return False
    
    def display_rgb_channels(self):
        """Display individual R, G, B channels"""
        if self.image_rgb is None:
            print("No RGB image available. Please call read_image() first.")
            return
        
        # Split RGB channels
        r_channel = self.image_rgb[:, :, 0]  # Red channel
        g_channel = self.image_rgb[:, :, 1]  # Green channel
        b_channel = self.image_rgb[:, :, 2]  # Blue channel
        
        # Create zero arrays for individual channel visualization
        zeros = np.zeros_like(r_channel)
        
        # Create individual channel images
        red_image = np.stack([r_channel, zeros, zeros], axis=2)
        green_image = np.stack([zeros, g_channel, zeros], axis=2)
        blue_image = np.stack([zeros, zeros, b_channel], axis=2)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Display original image
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Original RGB Image')
        axes[0, 0].axis('off')
        
        # Display red channel
        axes[0, 1].imshow(red_image)
        axes[0, 1].set_title('Red Channel')
        axes[0, 1].axis('off')
        
        # Display green channel
        axes[0, 2].imshow(green_image)
        axes[0, 2].set_title('Green Channel')
        axes[0, 2].axis('off')
        
        # Display blue channel
        axes[1, 0].imshow(blue_image)
        axes[1, 0].set_title('Blue Channel')
        axes[1, 0].axis('off')
        
        # Display grayscale red channel
        axes[1, 1].imshow(r_channel, cmap='Reds')
        axes[1, 1].set_title('Red Channel (Grayscale)')
        axes[1, 1].axis('off')
        
        # Display grayscale green channel
        axes[1, 2].imshow(g_channel, cmap='Greens')
        axes[1, 2].set_title('Green Channel (Grayscale)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print channel statistics
        print("\nRGB Channel Statistics:")
        print(f"Red Channel - Min: {r_channel.min()}, Max: {r_channel.max()}, Mean: {r_channel.mean():.2f}")
        print(f"Green Channel - Min: {g_channel.min()}, Max: {g_channel.max()}, Mean: {g_channel.mean():.2f}")
        print(f"Blue Channel - Min: {b_channel.min()}, Max: {b_channel.max()}, Mean: {b_channel.mean():.2f}")
    
    def display_hsv_channels(self):
        """Display individual H, S, V channels"""
        if self.image_hsv is None:
            print("No HSV image available. Please call read_image() first.")
            return
        
        # Split HSV channels
        h_channel = self.image_hsv[:, :, 0]  # Hue channel (0-179 in OpenCV)
        s_channel = self.image_hsv[:, :, 1]  # Saturation channel (0-255)
        v_channel = self.image_hsv[:, :, 2]  # Value channel (0-255)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Display original image
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Original RGB Image')
        axes[0, 0].axis('off')
        
        # Display hue channel
        axes[0, 1].imshow(h_channel, cmap='hsv')
        axes[0, 1].set_title('Hue Channel (0-179°)')
        axes[0, 1].axis('off')
        
        # Display saturation channel
        axes[0, 2].imshow(s_channel, cmap='gray')
        axes[0, 2].set_title('Saturation Channel')
        axes[0, 2].axis('off')
        
        # Display value channel
        axes[1, 0].imshow(v_channel, cmap='gray')
        axes[1, 0].set_title('Value Channel (Brightness)')
        axes[1, 0].axis('off')
        
        # Display hue histogram
        axes[1, 1].hist(h_channel.flatten(), bins=180, range=[0, 180], color='orange', alpha=0.7)
        axes[1, 1].set_title('Hue Distribution')
        axes[1, 1].set_xlabel('Hue Value (0-179)')
        axes[1, 1].set_ylabel('Frequency')
        
        # Display saturation-value scatter
        axes[1, 2].scatter(s_channel.flatten(), v_channel.flatten(), alpha=0.1, s=1)
        axes[1, 2].set_title('Saturation vs Value')
        axes[1, 2].set_xlabel('Saturation')
        axes[1, 2].set_ylabel('Value')
        
        plt.tight_layout()
        plt.show()
        
        # Print HSV statistics
        print("\nHSV Channel Statistics:")
        print(f"Hue Channel - Min: {h_channel.min()}, Max: {h_channel.max()}, Mean: {h_channel.mean():.2f}")
        print(f"Saturation Channel - Min: {s_channel.min()}, Max: {s_channel.max()}, Mean: {s_channel.mean():.2f}")
        print(f"Value Channel - Min: {v_channel.min()}, Max: {v_channel.max()}, Mean: {v_channel.mean():.2f}")
    
    def display_color_spaces_comparison(self):
        """Display comparison between RGB and HSV color spaces"""
        if self.image_rgb is None or self.image_hsv is None:
            print("Images not available. Please call read_image() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RGB Image
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('RGB Color Space')
        axes[0, 0].axis('off')
        
        # HSV Image (convert back to RGB for display)
        hsv_display = cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2RGB)
        axes[0, 1].imshow(hsv_display)
        axes[0, 1].set_title('HSV Color Space (converted to RGB for display)')
        axes[0, 1].axis('off')
        
        # RGB Histogram
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            axes[1, 0].hist(self.image_rgb[:, :, i].flatten(), bins=50, 
                           color=color, alpha=0.7, label=color.upper())
        axes[1, 0].set_title('RGB Histogram')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # HSV Histogram
        hsv_colors = ['orange', 'gray', 'black']
        hsv_names = ['Hue', 'Saturation', 'Value']
        for i in range(3):
            axes[1, 1].hist(self.image_hsv[:, :, i].flatten(), bins=50, 
                           color=hsv_colors[i], alpha=0.7, label=hsv_names[i])
        axes[1, 1].set_title('HSV Histogram')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def display_info(self):
        """Display the image shape and size"""
        if self.shape is not None and self.size is not None:
            print(f"Image Shape: {self.shape}")
            print(f"Image Size (total pixels): {self.size}")
            
            if len(self.shape) == 3:
                height, width, channels = self.shape
                print(f"Height: {height} pixels")
                print(f"Width: {width} pixels")
                print(f"Channels: {channels}")
            else:
                height, width = self.shape
                print(f"Height: {height} pixels")
                print(f"Width: {width} pixels")
                print("Channels: 1 (Grayscale)")
        else:
            print("No analysis data available. Please call analyze_image() first.")
    
    def process_image(self):
        """Complete processing: read, analyze, and display"""
        if self.read_image():
            if self.analyze_image():
                self.display_info()
                return True
        return False

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "path/to/your/image.jpg"
    
    # Create analyzer instance
    analyzer = ImageAnalyzer(image_path)
    
    # Process the image
    if analyzer.process_image():
        # Display RGB channels
        analyzer.display_rgb_channels()
        
        # Display HSV channels
        analyzer.display_hsv_channels()
        
        # Display color spaces comparison
        analyzer.display_color_spaces_comparison()
