import os
import argparse
from typing import List, Dict, Optional # Import necessary types

def create_label_file(root_dir: str, output_file: str) -> None:
    """
    Scans the specified root directory and outputs image file paths and labels
    to a text file.

    Args:
        root_dir (str): Path to the root directory of the dataset.
                        It assumes that subdirectories for each class (e.g., '10184')
                        exist directly under this directory.
        output_file (str): Path to the output text file.
    """
    # Dictionary to hold the mapping between subdirectory names (str) and labels (int)
    label_map: Dict[str, int] = {}
    current_label: int = 0
    # List to store the lines to be written to the output file
    output_lines: List[str] = []

    print(f"Dataset root directory: {root_dir}")
    print(f"Output file: {output_file}")

    # Get subdirectories directly under the root directory and sort them
    # This ensures consistent label assignment across runs
    try:
        # List of subdirectory names (strings)
        subdirs: List[str] = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not subdirs:
            print(f"Error: No subdirectories found in root directory '{root_dir}'.")
            print("Please check the dataset structure. Subdirectories for each class are expected directly under the root.")
            return # Exit the function as no subdirectories were found
    except FileNotFoundError:
        print(f"Error: Specified root directory '{root_dir}' not found.")
        return # Exit the function as the root directory doesn't exist
    except Exception as e:
        print(f"Error: An issue occurred while reading the root directory: {e}")
        return # Exit on other directory reading errors

    print("Detected subdirectories (classes):")
    subdir_name: str
    for subdir_name in subdirs:
        print(f"- {subdir_name}")
        # Assign a new label if this subdirectory hasn't been seen yet
        if subdir_name not in label_map:
            label_map[subdir_name] = current_label
            current_label += 1

    print("\nLabel mapping:")
    name: str
    label_val: int # Use a different variable name to avoid conflict with the label below
    for name, label_val in label_map.items():
        print(f"  '{name}' -> {label_val}")

    print("\nStarting file scan...")
    # Process each subdirectory
    for subdir_name in subdirs:
        label: int = label_map[subdir_name]
        subdir_path: str = os.path.join(root_dir, subdir_name)

        try:
            # List files within the subdirectory
            filename: str # Declare type for loop variable
            for filename in os.listdir(subdir_path):
                file_path: str = os.path.join(subdir_path, filename)
                # Check if it's a file (ignore subdirectories within class directories)
                if os.path.isfile(file_path):
                    # Simple check for image files (add/modify extensions as needed)
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                        # Create the output format "subdirectory_name/filename label"
                        # Manually use '/' for cross-platform compatibility instead of os.path.join
                        relative_path: str = f"{subdir_name}/{filename}"
                        output_lines.append(f"{relative_path} {label}")
                    # else:
                        # print(f"  Skipping: {filename} (not an image file)")
        except Exception as e:
            print(f"Warning: An error occurred while processing subdirectory '{subdir_name}': {e}")
            continue # Continue processing other directories even if one fails

    print(f"\nFile scan completed. Found {len(output_lines)} images.")

    # Write the results to the output file
    try:
        # The 'with open...' statement handles the file object type
        with open(output_file, 'w', encoding='utf-8') as f:
            line: str # Declare type for loop variable
            for line in output_lines:
                f.write(line + '\n')
        print(f"Label file '{output_file}' created successfully.")
    except IOError as e:
        print(f"Error: An error occurred while writing to output file '{output_file}': {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred during file writing: {e}")

# The code below runs only when the script is executed directly
if __name__ == "__main__":
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(
        description="Scans the root directory of a face recognition dataset and creates a text file containing image paths and labels.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve newlines in help messages
    )

    # Required argument: root directory
    parser.add_argument(
        '--root_dir',
        type=str,
        required=True,
        help="Path to the root directory of the dataset.\n"
        "Example: /path/to/your/dataset or C:\\Users\\YourUser\\Dataset"
    )

    # Optional argument: output file name
    parser.add_argument(
        '--output_file',
        type=str,
        default='labels.txt',
        help="Path to the output label file (default: labels.txt)"
    )

    # Parse the arguments. The return type is argparse.Namespace
    args: argparse.Namespace = parser.parse_args()

    # Execute the main function using the parsed arguments
    # args.root_dir and args.output_file are guaranteed to be strings
    # due to the 'type=str' setting in add_argument.
    create_label_file(args.root_dir, args.output_file)
