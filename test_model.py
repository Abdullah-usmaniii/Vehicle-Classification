from pathlib import Path
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

        # 3. Setup the Matplotlib figure and connect keyboard controls
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Display the first image
        self.show_current_image()
        plt.show()

    def show_current_image(self):
        self.ax.clear()

        # Get current image path
        img_path = self.image_paths[self.current_index]

        # Run inference using your fine-tuned model
        # verbose=False keeps the terminal clean
        results = self.model(img_path, verbose=False)

        # Ultralytics results[0].plot() returns a numpy array with bounding boxes drawn
        # Note: It returns BGR color format (OpenCV standard), so we convert it to RGB for Matplotlib
        annotated_img = results[0].plot()[..., ::-1]

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