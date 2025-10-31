import cv2
import matplotlib.pyplot as plt
import numpy as np

class DualImageProcessor:
    def __init__(self, image_path1, image_path2):
        """Initialize with two image file paths"""
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.image1 = None
        self.image2 = None
        self.image1_flipped = None
        self.image2_flipped = None
        self.image1_rect = None
        self.image2_triangle = None
        
    def read_and_display_images(self):
        """Read 2 images and display them"""
        # Read images using OpenCV (reads in BGR)
        img1_bgr = cv2.imread(self.image_path1)
        img2_bgr = cv2.imread(self.image_path2)
        
        # Correct RGB order (BGR to RGB)
        self.image1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        self.image2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
        
        # Display both images
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(self.image1)
        axes[0].axis('off')
        axes[0].set_title('Image 1 (RGB Corrected)')
        
        axes[1].imshow(self.image2)
        axes[1].axis('off')
        axes[1].set_title('Image 2 (RGB Corrected)')
        
        plt.tight_layout()
        plt.show()
        
        return self.image1, self.image2
    
    def flip_and_display(self):
        """Flip the images upside down and display them"""
        if self.image1 is None or self.image2 is None:
            print("Please read images first using read_and_display_images()")
            return
        
        # Flip images upside down (flip code 0 = vertical flip)
        self.image1_flipped = cv2.flip(self.image1, 0)
        self.image2_flipped = cv2.flip(self.image2, 0)
        
        # Display flipped images
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(self.image1_flipped)
        axes[0].axis('off')
        axes[0].set_title('Image 1 (Flipped Upside Down)')
        
        axes[1].imshow(self.image2_flipped)
        axes[1].axis('off')
        axes[1].set_title('Image 2 (Flipped Upside Down)')
        
        plt.tight_layout()
        plt.show()
        
        return self.image1_flipped, self.image2_flipped
    
    def draw_rectangle_on_image1(self, top_left=(50, 50), bottom_right=(200, 200), thickness=3):
        """Draw an empty red rectangle on image 1"""
        if self.image1 is None:
            print("Please read images first using read_and_display_images()")
            return
        
        # Create a copy to avoid modifying original
        self.image1_rect = self.image1.copy()
        
        # Draw red rectangle (color in RGB: red = (255, 0, 0))
        cv2.rectangle(self.image1_rect, top_left, bottom_right, (255, 0, 0), thickness)
        
        # Display
        plt.figure(figsize=(8, 6))
        plt.imshow(self.image1_rect)
        plt.axis('off')
        plt.title('Image 1 with Red Rectangle')
        plt.show()
        
        return self.image1_rect
    
    def draw_triangle_on_image2(self):
        """Draw a filled blue triangle in the middle of image 2"""
        if self.image2 is None:
            print("Please read images first using read_and_display_images()")
            return
        
        # Create a copy to avoid modifying original
        self.image2_triangle = self.image2.copy()
        
        # Get image dimensions
        height, width = self.image2.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Define triangle points (centered, pointing upward)
        triangle_size = min(width, height) // 4
        pts = np.array([
            [center_x, center_y - triangle_size],           # Top point
            [center_x - triangle_size, center_y + triangle_size // 2],  # Bottom left
            [center_x + triangle_size, center_y + triangle_size // 2]   # Bottom right
        ], np.int32)
        
        pts = pts.reshape((-1, 1, 2))
        
        # Draw filled blue triangle (color in RGB: blue = (0, 0, 255))
        cv2.fillPoly(self.image2_triangle, [pts], (0, 0, 255))
        
        # Display
        plt.figure(figsize=(8, 6))
        plt.imshow(self.image2_triangle)
        plt.axis('off')
        plt.title('Image 2 with Blue Filled Triangle')
        plt.show()
        
        return self.image2_triangle
    
    def display_all_versions(self):
        """Display original and all modified versions of both images"""
        if self.image1 is None or self.image2 is None:
            print("Please process images first!")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))
        
        # Row 1: Original images
        axes[0, 0].imshow(self.image1)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Image 1 - Original')
        
        axes[0, 1].imshow(self.image2)
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Image 2 - Original')
        
        # Row 2: Flipped images
        if self.image1_flipped is not None:
            axes[1, 0].imshow(self.image1_flipped)
            axes[1, 0].axis('off')
            axes[1, 0].set_title('Image 1 - Flipped')
        
        if self.image2_flipped is not None:
            axes[1, 1].imshow(self.image2_flipped)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Image 2 - Flipped')
        
        # Row 3: Modified images (rectangle and triangle)
        if self.image1_rect is not None:
            axes[2, 0].imshow(self.image1_rect)
            axes[2, 0].axis('off')
            axes[2, 0].set_title('Image 1 - Red Rectangle')
        
        if self.image2_triangle is not None:
            axes[2, 1].imshow(self.image2_triangle)
            axes[2, 1].axis('off')
            axes[2, 1].set_title('Image 2 - Blue Triangle')
        
        plt.tight_layout()
        plt.show()


# Example usage:
# processor = DualImageProcessor('path/to/image1.jpg', 'path/to/image2.jpg')
# 
# # Step 1: Read and display images with RGB correction
# processor.read_and_display_images()
# 
# # Step 2: Flip images upside down
# processor.flip_and_display()
# 
# # Step 3: Draw red rectangle on image 1
# processor.draw_rectangle_on_image1()
# 
# # Step 4: Draw blue filled triangle on image 2
# processor.draw_triangle_on_image2()
# 
# # Step 5: Display all versions together
# processor.display_all_versions()
