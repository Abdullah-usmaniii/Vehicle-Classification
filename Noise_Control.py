import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = "noiseImages"
OUTPUT_ROOT = "Processed_Images"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# How much to enlarge the image for cleaner, crisper viewing
UPSCALE_FACTOR = 2.5  


# ---------------------------------------------------------------------------
# Core Enhancement Logic
# ---------------------------------------------------------------------------
def intelligently_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """
    Removes noise and sharpens blurry edges naturally without creating
    harsh halos, blinding contrast, or charcoal-like artifacts.
    """
    # 1. Clean Upscaling (Lanczos4 preserves edge sharpness better than linear/cubic)
    h, w = image_bgr.shape[:2]
    new_size = (int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR))
    upscaled = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Convert to YCrCb Color Space (Y = Brightness/Detail, Cr/Cb = Color)
    ycrcb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # 3. Aggressive Chroma Denoising (Kills color noise without softening edges)
    cr_clean = cv2.GaussianBlur(cr, (0, 0), sigmaX=3.0)
    cb_clean = cv2.GaussianBlur(cb, (0, 0), sigmaX=3.0)
    
    # 4. Intelligent Luminance Smoothing (Bilateral filter smooths flat areas, protects edges)
    y_smooth = cv2.bilateralFilter(y, d=7, sigmaColor=40, sigmaSpace=40)
    
    # 5. Halo-Free Unsharp Masking (Crispens blurry transitions gently)
    blur_mask = cv2.GaussianBlur(y_smooth, (0, 0), sigmaX=1.5)
    y_sharp = cv2.addWeighted(y_smooth, 1.25, blur_mask, -0.25, 0)
    
    # 6. Recombine and convert back to BGR
    enhanced_ycrcb = cv2.merge((y_sharp, cr_clean, cb_clean))
    final_bgr = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return final_bgr


# ---------------------------------------------------------------------------
# Interactive Matplotlib Viewer (Prevents UI Crashes)
# ---------------------------------------------------------------------------
class InteractiveViewer:
    def __init__(self, results: list):
        self.results = results
        self.index = 0
        self.total = len(results)
        
        # Create a single 1x2 figure and reserve space at the bottom for buttons
        self.fig, (self.ax_before, self.ax_after) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.subplots_adjust(bottom=0.18)
        
        # Define button positions [left, bottom, width, height] in figure coordinates
        ax_prev = plt.axes([0.35, 0.05, 0.12, 0.06])
        ax_next = plt.axes([0.53, 0.05, 0.12, 0.06])
        
        self.btn_prev = Button(ax_prev, '◀ Previous', color='0.9', hovercolor='0.8')
        self.btn_next = Button(ax_next, 'Next ▶', color='0.9', hovercolor='0.8')
        
        # Connect click and keyboard events
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Initial render
        self.update_display()
        print("\n[UI READY] Use the buttons or Left/Right arrow keys to navigate images.")
        plt.show()

    def update_display(self):
        item = self.results[self.index]
        
        # Clear previous frame from memory
        self.ax_before.clear()
        self.ax_after.clear()
        
        # Render Before
        self.ax_before.imshow(item["original"])
        self.ax_before.set_title(f"Before: {item['filename']}\n({self.index + 1} of {self.total})", fontsize=12, pad=10)
        self.ax_before.axis("off")
        
        # Render After
        self.ax_after.imshow(item["processed"])
        self.ax_after.set_title(f"After: Intelligent Smoothing\n({self.index + 1} of {self.total})", fontsize=12, fontweight='bold', color='darkblue', pad=10)
        self.ax_after.axis("off")
        
        self.fig.canvas.draw_idle()

    def next_image(self, event=None):
        self.index = (self.index + 1) % self.total
        self.update_display()

    def prev_image(self, event=None):
        self.index = (self.index - 1) % self.total
        self.update_display()

    def on_key(self, event):
        # Allow arrow keys, spacebar, and 'A'/'D' for navigation
        if event.key in ['right', 'down', 'd', 'space']:
            self.next_image()
        elif event.key in ['left', 'up', 'a', 'backspace']:
            self.prev_image()


# ---------------------------------------------------------------------------
# Main Execution & File Handling
# ---------------------------------------------------------------------------
def main():
    if not os.path.isdir(INPUT_DIR):
        raise SystemExit(f"Input folder '{INPUT_DIR}' not found. Create it and add images.")

    filenames = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXTENSIONS))
    if not filenames:
        raise SystemExit(f"No images found in '{INPUT_DIR}'.")

    run_folder = os.path.join(OUTPUT_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_folder, exist_ok=True)
    
    results = []

    for filename in filenames:
        input_path = os.path.join(INPUT_DIR, filename)
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            print(f"  [skip] Could not read '{filename}'.")
            continue

        print(f"Enhancing '{filename}'...")
        enhanced_bgr = intelligently_enhance(image_bgr)

        # Save to output directory
        out_path = os.path.join(run_folder, f"enhanced_{filename}")
        cv2.imwrite(out_path, enhanced_bgr)

        results.append({
            "filename": filename,
            "original": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            "processed": cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        })

    print(f"\nDone. Saved {len(results)} image(s) to: {run_folder}")
    
    if results:
        InteractiveViewer(results)


if __name__ == "__main__":
    main()