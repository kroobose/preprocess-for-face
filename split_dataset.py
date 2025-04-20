import os
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional # For type hinting

def gather_data_by_class(root_dir: str) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    """
    Scans the root directory and groups relative image file paths by their class label.

    Args:
        root_dir (str): Path to the root directory of the dataset.
                        Assumes class subdirectories are directly under this root.

    Returns:
        Tuple[Dict[int, List[str]], Dict[int, str]]:
            - data_by_class: A dictionary mapping class labels (int) to a list of
                             relative image paths (str, e.g., "classname/image.jpg").
            - label_to_classname: A dictionary mapping class labels (int) to
                                  class names (str, i.e., subdirectory names).

    Raises:
        FileNotFoundError: If the root directory does not exist.
        ValueError: If no subdirectories (classes) are found in the root directory,
                    or if no image files are found in any class subdirectory.
    """
    print(f"Scanning dataset root directory: {root_dir}")
    # Check if root directory exists
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Error: Root directory not found: {root_dir}")

    # Use defaultdict for convenient list appending for each label
    data_by_class: Dict[int, List[str]] = defaultdict(list)
    label_to_classname: Dict[int, str] = {}
    classname_to_label: Dict[str, int] = {}
    current_label: int = 0
    total_images_found: int = 0

    # --- 1. Discover classes and assign labels ---
    try:
        # List only directories directly under root_dir
        class_names: List[str] = sorted([
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ])
        if not class_names:
            raise ValueError(f"Error: No subdirectories found in root directory '{root_dir}'. Expected class folders.")
    except Exception as e:
        print(f"Error reading subdirectories from '{root_dir}': {e}")
        raise # Re-raise the exception to stop execution

    print(f"Found {len(class_names)} potential classes (subdirectories):")
    # Create mapping from class name to label and vice-versa
    for class_name in class_names:
        if class_name not in classname_to_label:
            print(f"- '{class_name}' -> Label {current_label}")
            classname_to_label[class_name] = current_label
            label_to_classname[current_label] = class_name
            current_label += 1
    print("-" * 30)

    # --- 2. Gather image file paths for each class ---
    print("Gathering image paths per class...")
    for class_name, label in classname_to_label.items():
        subdir_path: str = os.path.join(root_dir, class_name)
        images_in_class: int = 0
        try:
            # Iterate through items in the class subdirectory
            filename: str
            for filename in os.listdir(subdir_path):
                file_path: str = os.path.join(subdir_path, filename)
                # Check if it's a file and has a common image extension
                if os.path.isfile(file_path) and \
                   filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                    # Store the relative path format: "classname/filename.ext"
                    # Use forward slash for consistency, even on Windows
                    relative_path: str = f"{class_name}/{filename}"
                    data_by_class[label].append(relative_path)
                    images_in_class += 1
        except Exception as e:
            print(f"Warning: Could not read files from subdirectory '{subdir_path}': {e}")
            # Continue to the next class even if one fails

        if images_in_class > 0:
            # Only count classes where images were actually found
            total_images_found += images_in_class
        else:
            # If a directory initially found has no images, remove it from mappings
            print(f"Warning: No image files found in initially detected class '{class_name}'. Excluding this class.")
            # Remove the label if no images were added for it
            if label in data_by_class and not data_by_class[label]:
                 del data_by_class[label]
                 del label_to_classname[label]
                 # Keep classname_to_label as is, though this label won't appear in output


    # Final check if any images were collected at all
    if total_images_found == 0:
        raise ValueError("Error: No valid image files found in any class subdirectories.")

    print(f"\nFinished gathering data. Found {total_images_found} images across {len(data_by_class)} classes with images.")
    print("-" * 30)
    return data_by_class, label_to_classname


def split_data_stratified(
    data_by_class: Dict[int, List[str]],
    train_ratio: float,
    seed: Optional[int] = None
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Splits the data stratified by class into training and validation sets.

    Args:
        data_by_class (Dict[int, List[str]]): Data grouped by class label, where
                                                keys are labels (int) and values are
                                                lists of relative paths (str).
        train_ratio (float): The proportion of data to allocate to the training set
                             (e.g., 0.8 for 80%). Must be between 0 and 1.
        seed (Optional[int]): Random seed for shuffling to ensure reproducibility.
                              If None, shuffle will be different each time.

    Returns:
        Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
            - train_set: A list of (relative_path, label) tuples for training.
            - val_set: A list of (relative_path, label) tuples for validation.

    Raises:
        ValueError: If train_ratio is not between 0 and 1.
    """
    if not (0 < train_ratio < 1):
        raise ValueError("Error: Train ratio must be between 0 and 1 (exclusive).")

    # Set the random seed if provided for reproducible results
    if seed is not None:
        print(f"Using random seed: {seed} for shuffling.")
        random.seed(seed)
    else:
        print("No random seed provided. Split will vary on each run.")

    all_train_data: List[Tuple[str, int]] = []
    all_val_data: List[Tuple[str, int]] = []

    print(f"Splitting data per class (Train Ratio: {train_ratio:.2f})...")

    # Sort class labels for consistent processing order (good practice)
    sorted_labels: List[int] = sorted(data_by_class.keys())

    # Iterate through each class label
    label: int
    for label in sorted_labels:
        class_paths: List[str] = data_by_class[label]
        num_images_in_class: int = len(class_paths)

        # --- Handle edge cases for small classes ---
        if num_images_in_class == 0:
            # Should not happen if gather_data_by_class cleaned up, but check anyway
            print(f"  Skipping Class {label}: No images.")
            continue
        elif num_images_in_class == 1:
            # If only one image, assign it to the training set
            print(f"  Warning: Class {label} has only 1 image. Assigning to training set.")
            all_train_data.append((class_paths[0], label))
            continue # Move to the next class

        # --- Perform splitting for classes with >= 2 images ---
        # Shuffle the paths within the current class randomly
        random.shuffle(class_paths)

        # Calculate the number of training samples for this class
        n_train: int = int(num_images_in_class * train_ratio)

        # Ensure at least one sample in validation set if train_ratio is not 1.0
        # This prevents val set being empty if rounding causes n_train = num_images
        if n_train == num_images_in_class and train_ratio < 1.0:
            n_train = num_images_in_class - 1 # Force at least one for validation

        # Ensure at least one sample in training set if train_ratio is not 0.0
        # This prevents train set being empty if rounding causes n_train = 0
        if n_train == 0 and train_ratio > 0.0:
            n_train = 1 # Force at least one for training

        # Split the shuffled list into training and validation paths
        train_paths: List[str] = class_paths[:n_train]
        val_paths: List[str] = class_paths[n_train:]

        # Add the (path, label) tuples to the corresponding global lists
        all_train_data.extend([(path, label) for path in train_paths])
        all_val_data.extend([(path, label) for path in val_paths])

        # Log the split for this class
        # print(f"  Class {label}: Total={num_images_in_class}, Train={len(train_paths)}, Val={len(val_paths)}")

    print("Finished splitting per class.")
    print("-" * 30)

    # Optional: Shuffle the combined train and validation sets again
    # This mixes samples from different classes within each set
    print("Shuffling the final combined train and validation sets...")
    if seed is not None: # Use the same seed again for consistency if provided
        random.seed(seed)
    random.shuffle(all_train_data)
    if seed is not None:
        random.seed(seed) # Re-seed before shuffling val_data if seed is used
    random.shuffle(all_val_data)

    print(f"Final Training Set Size: {len(all_train_data)} samples")
    print(f"Final Validation Set Size: {len(all_val_data)} samples")
    print("-" * 30)

    return all_train_data, all_val_data


def write_output_file(data: List[Tuple[str, int]], output_file: str) -> None:
    """
    Writes the list of (path, label) tuples to a text file, one per line.

    Args:
        data (List[Tuple[str, int]]): List containing (relative_path, label) tuples.
        output_file (str): The path to the output text file.
    """
    print(f"Writing {len(data)} lines to: {output_file}...")
    try:
        # Use pathlib to handle paths and ensure parent directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data to the file with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            line_content: str
            for path, label in data:
                # Format line as: relative/path/image.jpg label
                line_content = f"{path} {label}\n"
                f.write(line_content)
        print(f"Successfully wrote to {output_file}")

    except IOError as e:
        print(f"Error: Could not write to output file '{output_file}': {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred while writing to '{output_file}': {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Scans a dataset directory with class subfolders, performs stratified splitting "
                    "into training and validation sets, and creates corresponding label files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    # Define command-line arguments
    parser.add_argument(
        '--root_dir', type=str, required=True,
        help="Path to the root directory of the dataset. This directory should contain subdirectories, "
             "each representing a class and containing image files for that class."
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help="Proportion of the data to use for the training set (e.g., 0.8 means 80%% train, 20%% validation)."
    )
    parser.add_argument(
        '--train_output', type=str, default='train.txt',
        help="Output file path for the training set labels (format: path/image.jpg label)."
    )
    parser.add_argument(
        '--val_output', type=str, default='val.txt',
        help="Output file path for the validation set labels (format: path/image.jpg label)."
    )
    parser.add_argument(
        '--seed', type=int, default=None, # Default is None, meaning non-reproducible shuffle
        help="Optional random seed (integer) for shuffling to ensure reproducible splits. "
             "If not provided, the split will be random each time."
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # --- Execute the workflow ---
    try:
        # Step 1: Gather data grouped by class label
        data_by_class, _ = gather_data_by_class(args.root_dir)
        # We get label_to_classname back, but don't strictly need it here

        # Step 2: Perform stratified split
        train_data, val_data = split_data_stratified(
            data_by_class,
            args.train_ratio,
            args.seed
        )

        # Step 3: Write the results to output files
        write_output_file(train_data, args.train_output)
        write_output_file(val_data, args.val_output)

        print("\nDataset splitting process completed successfully.")

    except (FileNotFoundError, ValueError, IOError, Exception) as e:
        # Catch expected errors and general exceptions
        print(f"\nError during processing: {e}")
        print("Process failed.")
