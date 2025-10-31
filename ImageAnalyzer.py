import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.image_rgb = None
        self.image_hsv = None
        self.image_gray = None
        self.shape = None
        self.size = None
        self.processed_images = {}
    
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
            
            # Convert to grayscale
            self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
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
    
    def perform_thresholding(self):
        """Perform thresholding using Gaussian blur with thresholds 5,55"""
        if self.image_gray is None:
            print("No grayscale image available. Please call read_image() first.")
            return None
        
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(self.image_gray, (5, 5), 0)
            
            # Apply threshold with values 5 and 55
            _, thresh1 = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)
            _, thresh2 = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)
            
            # Store the thresholded images
            self.processed_images['gaussian_blurred'] = blurred
            self.processed_images['threshold_5'] = thresh1
            self.processed_images['threshold_55'] = thresh2
            
            print("Thresholding completed successfully!")
            return thresh1, thresh2
            
        except Exception as e:
            print(f"Error in thresholding: {e}")
            return None
    
    def perform_morphological_operations(self, thresh_image):
        """Perform dilation, erosion, opening & closing using 5x5 kernel with 1 iteration"""
        if thresh_image is None:
            print("No threshold image provided.")
            return None
        
        try:
            # Create 5x5 kernel
            kernel = np.ones((5, 5), np.uint8)
            
            # Perform morphological operations
            dilation = cv2.dilate(thresh_image, kernel, iterations=1)
            erosion = cv2.erode(thresh_image, kernel, iterations=1)
            opening = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
            
            # Store the results
            results = {
                'dilation': dilation,
                'erosion': erosion,
                'opening': opening,
                'closing': closing
            }
            
            # Update processed images dictionary
            self.processed_images.update(results)
            
            print("Morphological operations completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error in morphological operations: {e}")
            return None
    
    def perform_canny_edge_detection(self, image, image_name="image"):
        """Apply Canny edge detection with wide and narrow thresholds"""
        if image is None:
            print(f"No {image_name} provided for edge detection.")
            return None
        
        try:
            # Wide threshold range (more edges)
            edges_wide = cv2.Canny(image, 50, 150)
            
            # Narrow threshold range (fewer edges)
            edges_narrow = cv2.Canny(image, 100, 200)
            
            # Store the results
            edge_results = {
                f'edges_wide_{image_name}': edges_wide,
                f'edges_narrow_{image_name}': edges_narrow
            }
            
            # Update processed images dictionary
            self.processed_images.update(edge_results)
            
            print(f"Canny edge detection completed for {image_name}!")
            return edge_results
            
        except Exception as e:
            print(f"Error in Canny edge detection for {image_name}: {e}")
            return None
    
    def display_thresholding_results(self):
        """Display thresholding results"""
        if 'threshold_5' not in self.processed_images or 'threshold_55' not in self.processed_images:
            print("Thresholding results not available. Please call perform_thresholding() first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original grayscale
        axes[0, 0].imshow(self.image_gray, cmap='gray')
        axes[0, 0].set_title('Original Grayscale')
        axes[0, 0].axis('off')
        
        # Gaussian blurred
        axes[0, 1].imshow(self.processed_images['gaussian_blurred'], cmap='gray')
        axes[0, 1].set_title('Gaussian Blurred')
        axes[0, 1].axis('off')
        
        # Threshold 5
        axes[0, 2].imshow(self.processed_images['threshold_5'], cmap='gray')
        axes[0, 2].set_title('Threshold (5)')
        axes[0, 2].axis('off')
        
        # Threshold 55
        axes[1, 0].imshow(self.processed_images['threshold_55'], cmap='gray')
        axes[1, 0].set_title('Threshold (55)')
        axes[1, 0].axis('off')
        
        # Histogram of original
        axes[1, 1].hist(self.image_gray.flatten(), bins=50, color='blue', alpha=0.7)
        axes[1, 1].set_title('Grayscale Histogram')
        axes[1, 1].axvline(x=5, color='red', linestyle='--', label='Thresh=5')
        axes[1, 1].axvline(x=55, color='green', linestyle='--', label='Thresh=55')
        axes[1, 1].legend()
        
        # Histogram of blurred
        axes[1, 2].hist(self.processed_images['gaussian_blurred'].flatten(), bins=50, color='orange', alpha=0.7)
        axes[1, 2].set_title('Blurred Histogram')
        axes[1, 2].axvline(x=5, color='red', linestyle='--', label='Thresh=5')
        axes[1, 2].axvline(x=55, color='green', linestyle='--', label='Thresh=55')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def display_morphological_results(self):
        """Display morphological operation results"""
        required_keys = ['dilation', 'erosion', 'opening', 'closing']
        if not all(key in self.processed_images for key in required_keys):
            print("Morphological results not available. Please call perform_morphological_operations() first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original threshold
        if 'threshold_5' in self.processed_images:
            axes[0, 0].imshow(self.processed_images['threshold_5'], cmap='gray')
            axes[0, 0].set_title('Original Threshold (5)')
            axes[0, 0].axis('off')
        
        # Dilation
        axes[0, 1].imshow(self.processed_images['dilation'], cmap='gray')
        axes[0, 1].set_title('Dilation')
        axes[0, 1].axis('off')
        
        # Erosion
        axes[0, 2].imshow(self.processed_images['erosion'], cmap='gray')
        axes[0, 2].set_title('Erosion')
        axes[0, 2].axis('off')
        
        # Opening
        axes[1, 0].imshow(self.processed_images['opening'], cmap='gray')
        axes[1, 0].set_title('Opening')
        axes[1, 0].axis('off')
        
        # Closing
        axes[1, 1].imshow(self.processed_images['closing'], cmap='gray')
        axes[1, 1].set_title('Closing')
        axes[1, 1].axis('off')
        
        # Kernel visualization
        kernel_viz = np.ones((5, 5)) * 255
        axes[1, 2].imshow(kernel_viz, cmap='gray', vmin=0, vmax=255)
        axes[1, 2].set_title('5x5 Kernel')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def display_canny_results(self, image_name="threshold_5"):
        """Display Canny edge detection results"""
        wide_key = f'edges_wide_{image_name}'
        narrow_key = f'edges_narrow_{image_name}'
        
        if wide_key not in self.processed_images or narrow_key not in self.processed_images:
            print(f"Canny results for {image_name} not available.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if image_name in self.processed_images:
            axes[0].imshow(self.processed_images[image_name], cmap='gray')
        else:
            axes[0].imshow(self.image_gray, cmap='gray')
        axes[0].set_title(f'Original ({image_name})')
        axes[0].axis('off')
        
        # Wide edges
        axes[1].imshow(self.processed_images[wide_key], cmap='gray')
        axes[1].set_title('Canny Edges (Wide: 50-150)')
        axes[1].axis('off')
        
        # Narrow edges
        axes[2].imshow(self.processed_images[narrow_key], cmap='gray')
        axes[2].set_title('Canny Edges (Narrow: 100-200)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print edge statistics
        wide_edges = self.processed_images[wide_key]
        narrow_edges = self.processed_images[narrow_key]
        
        print(f"\nEdge Detection Statistics for {image_name}:")
        print(f"Wide edges (50-150): {np.sum(wide_edges > 0)} edge pixels")
        print(f"Narrow edges (100-200): {np.sum(narrow_edges > 0)} edge pixels")
        print(f"Ratio (wide/narrow): {np.sum(wide_edges > 0) / max(1, np.sum(narrow_edges > 0)):.2f}")
    
    def complete_image_processing_pipeline(self):
        """Complete image processing pipeline with all operations"""
        if not self.read_image():
            return False
        
        self.analyze_image()
        self.display_info()
        
        # Perform thresholding
        thresh5, thresh55 = self.perform_thresholding()
        self.display_thresholding_results()
        
        # Perform morphological operations on threshold 5
        morph_results = self.perform_morphological_operations(thresh5)
        self.display_morphological_results()
        
        # Perform Canny edge detection on original threshold and morphological results
        self.perform_canny_edge_detection(thresh5, "threshold_5")
        self.display_canny_results("threshold_5")
        
        # Also perform Canny on some morphological results
        self.perform_canny_edge_detection(morph_results['opening'], "opening")
        self.display_canny_results("opening")
        
        self.perform_canny_edge_detection(morph_results['closing'], "closing")
        self.display_canny_results("closing")
        
        return True
    
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

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "path/to/your/image.jpg"
    
    # Create analyzer instance
    analyzer = ImageAnalyzer(image_path)
    
    # Run complete processing pipeline
    analyzer.complete_image_processing_pipeline()
    
    # Or run individual methods
    # analyzer.read_image()
    # analyzer.analyze_image()
    # analyzer.display_info()
    # analyzer.perform_thresholding()
    # analyzer.display_thresholding_results()
    # thresh5, thresh55 = analyzer.perform_thresholding()
    # analyzer.perform_morphological_operations(thresh5)
    # analyzer.display_morphological_results()
    # analyzer.perform_canny_edge_detection(thresh5, "threshold_5")
    # analyzer.display_canny_results("threshold_5")
