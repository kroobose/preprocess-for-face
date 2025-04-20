import os
from pathlib import Path
from typing import List, Tuple # For type hinting

# Try to import Pillow and set a flag
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# List of files to create
# Note: '3.' has been corrected to '3'.
# 'images/4/4_3.jpg' is duplicated as requested in the list.
# The script will create/overwrite this file twice according to the list.
files_to_create: List[str] = [
    "./images/0/0_1.jpg",
    "./images/0/0_2.jpg",
    "./images/0/0_3.jpg",
    "./images/1/1_1.jpg",
    "./images/1/1_2.jpg",
    "./images/2/2_1.jpg",
    "./images/2/2_2.jpg",
    "./images/2/2_3.jpg",
    "./images/3/3_1.jpg", # Corrected '3.' to '3'
    "./images/4/4_1.jpg",
    "./images/4/4_2.jpg",
    "./images/4/4_3.jpg",
    "./images/4/4_4.jpg",
]

# Dummy image properties (size as tuple, color as string)
dummy_image_size: Tuple[int, int] = (10, 10) # Small size (width, height)
dummy_image_color: str = 'gray'             # Color (e.g., 'red', 'blue', 'gray')

def create_dummy_files(file_list: List[str]) -> None:
    """
    Creates dummy JPEG files and their directories based on the provided list.
    Requires the Pillow library to be installed.

    Args:
        file_list (List[str]): A list of relative file paths to create.
    """
    # Check if Pillow is available
    if not PIL_AVAILABLE:
        print("Error: Pillow library is not installed or could not be imported.")
        print("Please install it using: pip install Pillow")
        return

    print("Starting creation of dummy image directories and files...")
    created_count: int = 0
    error_count: int = 0
    duplicate_paths_found = False

    # Check for duplicates beforehand to inform the user
    path_counts = {}
    for p in file_list:
        path_counts[p] = path_counts.get(p, 0) + 1
    duplicates = {path: count for path, count in path_counts.items() if count > 1}
    if duplicates:
        duplicate_paths_found = True
        print("\nWarning: The following file paths are duplicated in the list:")
        for path, count in duplicates.items():
            print(f"  - '{path}' (appears {count} times)")
        print("The file at these paths will be created/overwritten multiple times.")
        print("-" * 30)


    # Process each file path in the list
    for file_path_str in file_list:
        try:
            # Use pathlib for robust path handling
            file_path: Path = Path(file_path_str)

            # Create parent directories if they don't exist.
            # `parents=True` creates necessary parent directories.
            # `exist_ok=True` prevents an error if the directory already exists.
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate a small dummy image using Pillow
            # 'RGB' mode is standard for color images
            img = Image.new('RGB', dummy_image_size, color=dummy_image_color)

            # Save the dummy image as a JPEG file
            # If the file already exists (e.g., due to duplication in the list),
            # it will be overwritten without warning by default.
            img.save(file_path, 'JPEG')
            print(f"  Created/Overwritten: {file_path_str}")
            created_count += 1

        except OSError as e:
            # Handle potential OS-level errors (e.g., permission denied)
            print(f"Error creating directory or file '{file_path_str}': {e}")
            error_count += 1
        except ValueError as e:
             # Handle potential Pillow errors (e.g., invalid color name)
            print(f"Error generating image for '{file_path_str}': {e}")
            error_count += 1
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred for '{file_path_str}': {e}")
            error_count += 1

    print("\nDummy file creation process finished.")
    print("-" * 30)
    # Note: created_count reflects the number of successful save operations,
    # which will match len(file_list) if no errors occur, even with duplicates.
    print(f"Total operations attempted: {len(file_list)}")
    print(f"Files created/overwritten successfully: {created_count}")
    print(f"Errors encountered: {error_count}")
    if duplicate_paths_found:
         print("Reminder: Some file paths were duplicated and overwritten.")

# This block ensures the code runs only when the script is executed directly
if __name__ == "__main__":
    create_dummy_files(files_to_create)
