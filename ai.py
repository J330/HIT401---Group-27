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
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"
}

# Maximum image size.
# Images smaller than this will NOT be enlarged.
MAX_SIZE = (1024, 1024)


# -----------------------------
# CLEAN ONE IMAGE
# -----------------------------

def clean_image(input_path, output_path):
    try:
        with Image.open(input_path) as image:

            # Fully load the image to catch corrupted files
            image.load()

            # Fix rotation from phone/camera EXIF data
            image = ImageOps.exif_transpose(image)

            # Convert all images to RGB
            image = image.convert("RGB")

            # Resize large images while keeping aspect ratio
            image.thumbnail(MAX_SIZE)

            # Make sure output folder exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save everything as JPG
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

        print(f"\nBAD IMAGE: {input_path}")
        print(f"Reason: {error}")

        return False


# -----------------------------
# PROCESS DATASET
# -----------------------------

def process_dataset():

    # Make sure the source folder exists
    if not INPUT_FOLDER.exists():
        print(f"ERROR: Could not find '{INPUT_FOLDER}'")
        print("Make sure this Python file is in the same location as BananaDatasets.")
        return

    good_images = 0
    bad_images = 0
    skipped_images = 0
    ignored_files = 0

    print("Starting dataset cleaning...")
    print(f"Input:  {INPUT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}")

    # Go through Healthy, Sigatoka, and any future class folders
    for class_folder in INPUT_FOLDER.iterdir():

        if not class_folder.is_dir():
            continue

        print(f"\n----------------------------")
        print(f"Processing class: {class_folder.name}")
        print(f"----------------------------")

        for file_path in class_folder.rglob("*"):

            if not file_path.is_file():
                continue

            # Ignore non-image files
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                print(f"Ignored non-image: {file_path.name}")
                ignored_files += 1
                continue

            # Keep the same folder structure
            relative_path = file_path.relative_to(INPUT_FOLDER)

            output_path = OUTPUT_FOLDER / relative_path

            # Everything is saved as JPG
            output_path = output_path.with_suffix(".jpg")

            # -----------------------------------------
            # SKIP IMAGE IF IT WAS ALREADY CLEANED
            # -----------------------------------------

            if output_path.exists():
                print(f"Already cleaned, skipping: {file_path.name}")
                skipped_images += 1
                continue

            # -----------------------------------------
            # CLEAN NEW IMAGE
            # -----------------------------------------

            if clean_image(file_path, output_path):
                good_images += 1
                print(f"Cleaned: {file_path.name}")
            else:
                bad_images += 1

    # -----------------------------
    # FINAL REPORT
    # -----------------------------

    print("\n================================")
    print("DATASET CLEANING COMPLETE")
    print("================================")

    print(f"New images cleaned:     {good_images}")
    print(f"Already cleaned:        {skipped_images}")
    print(f"Bad/corrupted images:   {bad_images}")
    print(f"Non-image files ignored:{ignored_files}")

    print("\nCleaned dataset saved to:")
    print(OUTPUT_FOLDER)


# -----------------------------
# RUN PROGRAM
# -----------------------------

if __name__ == "__main__":
    process_dataset()