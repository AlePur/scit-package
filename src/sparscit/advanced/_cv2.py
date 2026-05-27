import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

class ImageToolsCV2:
    """Utility class for image processing operations using OpenCV."""
    @staticmethod
    def crop_white(img, threshold=245, padding=10):
        """
        Crop white parts of an image, keeping only the non-white content with optional padding.

        Parameters:
        image_path (str): Path to the input image
        threshold (int): Pixel values above this are considered white (0-255)
        padding (int): Additional padding around the cropped area

        Returns:
        numpy.ndarray: Cropped image
        """
        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create a binary mask where white pixels are 0 and non-white pixels are 1
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Find the coordinates of non-white pixels
        non_white_pixels = cv2.findNonZero(mask)

        # If no non-white pixels found, return the original image
        if non_white_pixels is None:
            print("No non-white content found in the image.")
            return img

        # Get the bounding box of non-white pixels
        x, y, w, h = cv2.boundingRect(non_white_pixels)

        # Add padding (ensure we don't go outside image boundaries)
        height, width = img.shape[:2]
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)

        # Crop the image
        cropped = img[y:y + h, x:x + w]

        return cropped

    @staticmethod
    def fig_to_img(fig, dpi = None):
        import cv2
        canvas = FigureCanvasAgg(fig)
        if dpi is not None:
            canvas.figure.dpi = dpi
        canvas.draw()
        figure_image = np.array(canvas.renderer.buffer_rgba())
        plt.close(fig)

        # Convert from RGBA to BGR (OpenCV format)
        figure_image = cv2.cvtColor(figure_image, cv2.COLOR_RGBA2BGRA)
        return figure_image
    
    @classmethod
    def join_figures(cls, figs: list, dpi=None, crop=True, spacing=20, direction='horizontal'):
        
        # Convert figures to images
        imgs = [cls.fig_to_img(f, dpi) for f in figs]
        
        # Optionally crop white space
        if crop:
            imgs = [cls.crop_white(img) for img in imgs]
        
        # Get dimensions
        heights = [img.shape[0] for img in imgs]
        widths = [img.shape[1] for img in imgs]
        
        if direction == 'horizontal':
            # Calculate combined dimensions
            total_width = sum(widths) + spacing * (len(imgs) - 1)
            max_height = max(heights)
            
            # Create white canvas
            combined = np.ones((max_height, total_width, 4), dtype=np.uint8) * 255
            
            # Paste images horizontally
            x_offset = 0
            for img in imgs:
                h, w = img.shape[:2]
                # Center vertically
                y_offset = (max_height - h) // 2
                combined[y_offset:y_offset + h, x_offset:x_offset + w] = img
                x_offset += w + spacing
                
        elif direction == 'vertical':
            # Calculate combined dimensions
            max_width = max(widths)
            total_height = sum(heights) + spacing * (len(imgs) - 1)
            
            # Create white canvas
            combined = np.ones((total_height, max_width, 4), dtype=np.uint8) * 255
            
            # Paste images vertically
            y_offset = 0
            for img in imgs:
                h, w = img.shape[:2]
                # Center horizontally
                x_offset = (max_width - w) // 2
                combined[y_offset:y_offset + h, x_offset:x_offset + w] = img
                y_offset += h + spacing
        else:
            raise ValueError("direction must be 'horizontal' or 'vertical'")
        
        return combined

    @staticmethod
    def overlay_images(background, overlay, position, alpha=1.0):
        """
        Place 'overlay' on 'background' at 'position' with transparency 'alpha'
        """
        # Get dimensions
        h, w = overlay.shape[:2]

        # Create region of interest (ROI) for the overlay
        # Ensure we don't go outside the background image boundaries
        y1, y2 = position[1], position[1] + h
        x1, x2 = position[0], position[0] + w

        # Crop if overlay would extend beyond background edges
        bg_h, bg_w = background.shape[:2]
        if x2 > bg_w:
            overlay = overlay[:, 0:bg_w - x1]
            x2 = bg_w
        if y2 > bg_h:
            overlay = overlay[0:bg_h - y1, :]
            y2 = bg_h
        if x1 < 0:
            overlay = overlay[:, abs(x1):]
            x1 = 0
        if y1 < 0:
            overlay = overlay[abs(y1):, :]
            y1 = 0

        # Get the ROI
        roi = background[y1:y2, x1:x2]

        # Now overlay, respecting transparency
        # Extract overlay alpha channel
        if overlay.shape[2] == 4:  # If overlay has an alpha channel
            overlay_alpha = overlay[:, :, 3] / 255.0 * alpha
            alpha_3channel = np.dstack((overlay_alpha, overlay_alpha, overlay_alpha))

            # Extract RGB channels
            overlay_rgb = overlay[:, :, 0:3]

            # Blend
            for c in range(0, 3):
                roi[:, :, c] = (1 - alpha_3channel[:, :, c]) * roi[:, :, c] + alpha_3channel[:, :, c] * overlay_rgb[:, :, c]

            # Update alpha channel of background
            roi_alpha = roi[:, :, 3] / 255.0
            new_alpha = np.maximum(roi_alpha, overlay_alpha)
            roi[:, :, 3] = new_alpha * 255
        else:
            import cv2
            # If overlay doesn't have alpha, just blend with constant alpha
            cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0, roi)

        return background