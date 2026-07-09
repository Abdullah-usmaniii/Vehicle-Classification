from pathlib import Path
from datetime import datetime
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


class YoloModelTester:

    def __init__(self, model_path, images_folder):
        # 1. Load the fine-tuned YOLO model
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)

        # 2. Gather all image paths from the folder
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.image_paths = [
            p
            for p in Path(images_folder).iterdir()
            if p.suffix.lower() in supported_formats
        ]

        if not self.image_paths:
            raise ValueError(f"No images found in folder: {images_folder}")

        self.current_index = 0
        print(
            f"Found {len(self.image_paths)} images. Starting interactiver viewer..."
        )

        # 4. Create a timestamped output subfolder inside Processed_Images (time_date)
        run_timestamp = datetime.now().strftime("%H-%M-%S_%Y-%m-%d")
        processed_root = Path(__file__).parent / "Processed_Images"
        self.output_folder = processed_root / run_timestamp
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Saving processed images to: {self.output_folder}")

        # 5. Run inference on ALL images upfront and save results to disk
        self._process_all_images()

        # 3. Setup the Matplotlib figure and connect keyboard controls
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Display the first image
        self.show_current_image()
        plt.show()

    def _process_all_images(self):
        """Run inference on every image, save annotated image + label txt to disk.
        Stores the RGB annotated arrays in self.annotated_images for the GUI."""
        print("Processing all images — please wait...")
        self.annotated_images = []
        total = len(self.image_paths)
        for i, img_path in enumerate(self.image_paths, start=1):
            print(f"  [{i}/{total}] {img_path.name}", end="\r")

            results = self.model(img_path, verbose=False)
            annotated_img_bgr = results[0].plot()         # BGR numpy array
            annotated_img_rgb = annotated_img_bgr[..., ::-1]  # RGB for Matplotlib
            self.annotated_images.append(annotated_img_rgb)

            # --- Save annotated image ---
            img_suffix = img_path.suffix.lower()
            save_ext = img_suffix if img_suffix in (".jpg", ".jpeg", ".png") else ".jpg"
            stem = img_path.stem
            out_image_path = self.output_folder / f"{stem}_processed{save_ext}"
            cv2.imwrite(str(out_image_path), annotated_img_bgr)

            # --- Save corresponding label text file ---
            out_text_path = self.output_folder / f"{stem}_processed.txt"
            boxes = results[0].boxes
            names = results[0].names
            lines = []
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    lines.append(
                        f"{class_name} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}"
                    )
            out_text_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"\nAll {total} images processed and saved to: {self.output_folder}")

    def show_current_image(self):
        self.ax.clear()

        # Get current image path and its pre-computed annotated array
        img_path = self.image_paths[self.current_index]
        annotated_img = self.annotated_images[self.current_index]  # already RGB

        # Display the image in the Matplotlib axes
        self.ax.imshow(annotated_img)
        self.ax.set_title(
            f"Image {self.current_index + 1}/{len(self.image_paths)}: {img_path.name}\n"
            f"[SPACE / RIGHT ARROW] Next  |  [LEFT ARROW] Previous  |  [Q / ESC] Quit",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        self.ax.axis("off")  # Hide grid axes for a cleaner look
        self.fig.canvas.draw()

    def on_key_press(self, event):
        # Go to Next Image
        if event.key in ["right", " ", "enter"]:
            if self.current_index < len(self.image_paths) - 1:
                self.current_index += 1
                self.show_current_image()
            else:
                print("You have reached the last image in the folder.")

        # Go to Previous Image
        elif event.key in ["left", "backspace"]:
            if self.current_index > 0:
                self.current_index -= 1
                self.show_current_image()

        # Quit Program
        elif event.key in ["q", "escape"]:
            print("Exiting evaluation window.")
            plt.close(self.fig)


# RUNNING THE PROGRAM
if __name__ == "__main__":
    # Change these paths to match your actual files
    MODEL_PATH = "my_model.pt"
    IMAGES_FOLDER = Path(__file__).parent / "Images"

    YoloModelTester(model_path=MODEL_PATH, images_folder=IMAGES_FOLDER)