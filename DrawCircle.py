import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_circles_on_click(image_path, circle_radius=30, circle_thickness=2):
    """
    Read an image and add empty red circles when right-clicked.
    
    Parameters:
    - image_path: Path to the image file
    - circle_radius: Radius of the circles to draw
    - circle_thickness: Thickness of the circle outline
    
    Controls:
    - Right Click: Draw a red circle at the clicked position
    - Close the window to finish
    """
    
    # Read the image
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert BGR to RGB for matplotlib
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Keep a copy of the original image
    original_image = image.copy()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    img_display = ax.imshow(image)
    ax.axis('off')
    ax.set_title('Right-Click to Add Red Circles', fontsize=14, pad=10)
    
    # Store circles list for potential undo
    circles = []
    
    def on_click(event):
        nonlocal image
        
        # Check if right mouse button was clicked and click is inside the axes
        if event.button == 3 and event.inaxes == ax:  # button 3 = right click
            x, y = int(event.xdata), int(event.ydata)
            
            # Draw an empty red circle (RGB format: red = (255, 0, 0))
            cv2.circle(image, (x, y), circle_radius, (255, 0, 0), circle_thickness)
            
            # Store circle info
            circles.append((x, y))
            
            # Update the display
            img_display.set_data(image)
            fig.canvas.draw()
            
            print(f"Circle #{len(circles)} drawn at position: ({x}, {y})")
    
    def on_key(event):
        nonlocal image
        
        # Press 'r' to reset
        if event.key == 'r':
            image = original_image.copy()
            circles.clear()
            img_display.set_data(image)
            fig.canvas.draw()
            print("Image reset to original")
        
        # Press 'u' to undo last circle
        elif event.key == 'u' and circles:
            circles.pop()
            image = original_image.copy()
            # Redraw all remaining circles
            for x, y in circles:
                cv2.circle(image, (x, y), circle_radius, (255, 0, 0), circle_thickness)
            img_display.set_data(image)
            fig.canvas.draw()
            print(f"Undo - {len(circles)} circle(s) remaining")
    
    # Connect the event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Display instructions
    print("=" * 60)
    print("Interactive Circle Drawer")
    print("=" * 60)
    print("Controls:")
    print("  - Right Click: Draw a red circle")
    print("  - Press 'r': Reset image (remove all circles)")
    print("  - Press 'u': Undo last circle")
    print("  - Close window: Finish and return result")
    print("=" * 60)
    
    plt.tight_layout()
    plt.show()
    
    return image


# Example usage:
# result_image = draw_circles_on_click('path/to/your/image.jpg')
# 
# # Optional: Display the result
# if result_image is not None:
#     plt.figure(figsize=(10, 8))
#     plt.imshow(result_image)
#     plt.axis('off')
#     plt.title('Final Result')
#     plt.show()
# 
# # Optional: Save the modified image
# # result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
# # cv2.imwrite('output_with_circles.jpg', result_bgr)
