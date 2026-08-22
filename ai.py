from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

# -----------------------------
# SETTINGS
# -----------------------------

INPUT_FOLDER = Path("BananaDatasets")
OUTPUT_FOLDER = Path("Cleaned_BananaDatasets")

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
}

# Change this if you want a different maximum resolution.
# Images smaller than this are NOT enlarged.
MAX_SIZE = (1024, 1024)


# -----------------------------
# CLEAN IMAGE
# -----------------------------

def clean_image(input_path, output_path):
    try:
        with Image.open(input_path) as image:

            # Check that the file can actually be decoded
            image.load()

            # Correct phone/camera rotation using EXIF data
            image = ImageOps.exif_transpose(image)

            # Convert everything to RGB
            # This removes palette/grayscale/alpha inconsistencies
            image = image.convert("RGB")

            # Resize very large images while maintaining aspect ratio
            image.thumbnail(MAX_SIZE)

            # Make sure destination folder exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save everything as JPEG
            output_path = output_path.with_suffix(".jpg")

            image.save(
                output_path,
                "JPEG",
                quality=90,
                optimize=True
            )

            return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as error:

        print(f"BAD IMAGE: {input_path}")
        print(f"Reason: {error}")

        return False


# -----------------------------
# PROCESS DATASET
# -----------------------------

def process_dataset():

    good_images = 0
    bad_images = 0
    ignored_files = 0

    for class_folder in INPUT_FOLDER.iterdir():

        if not class_folder.is_dir():
            continue

        print(f"\nProcessing: {class_folder.name}")

        for file_path in class_folder.rglob("*"):

            if not file_path.is_file():
                continue

            # Ignore non-image files
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                print(f"Ignored: {file_path}")
                ignored_files += 1
                continue

            # Keep the same class/folder structure
            relative_path = file_path.relative_to(INPUT_FOLDER)

            output_path = OUTPUT_FOLDER / relative_path

            if clean_image(file_path, output_path):
                good_images += 1
                print(f"Cleaned: {file_path}")

            else:
                bad_images += 1

    print("\n----------------------------")
    print("DATASET CLEANING COMPLETE")
    print("----------------------------")
    print(f"Good images:    {good_images}")
    print(f"Bad images:     {bad_images}")
    print(f"Ignored files:  {ignored_files}")
    print(f"Output folder:  {OUTPUT_FOLDER}")


# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    process_dataset()