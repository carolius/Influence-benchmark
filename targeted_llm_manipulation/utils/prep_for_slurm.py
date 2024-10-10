import os
import re
import sys


def modify_data_root_file(file_path, package_name):
    """
    Modifying the root file to import PROJECT_ROOT and PROJECT_DATA from the package itself,
    so that experiment data is saved in the package directory itself, rather than a temp location.

    This is tricky. We want all the data loading/saving to point to the original repo, but everything
    else to point to the local copy. This modification assumes that all data reading/writing happens
    with paths built from the PROJECT_DATA path present in data_root.py
    """
    content = f"""from pathlib import Path
from {package_name}.data_root import PROJECT_DATA as _PROJECT_DATA
PROJECT_DATA = _PROJECT_DATA
"""
    with open(file_path, "w") as f:
        f.write(content)


def modify_imports_in_file(file_path, package_name, add_sys_path=False):
    """
    Modifies the file at `file_path` to make it ready for being run on SLURM, without
    depending on the installed module code. This is done by adding sys.path modification
    at the beginning of the file and removing the package name from imports (having them be relative).
    """
    if os.path.basename(file_path) == "data_root.py":
        modify_data_root_file(file_path, package_name)
        print(f"data_root.py file modified: {main_file}")
        return

    with open(file_path, "r") as f:
        content = f.read()

    # Handle imports with 'as' keyword
    content = re.sub(
        f"import {package_name}([.\\w]+) as (\\w+)",
        lambda m: f"import {m.group(1).lstrip('.')} as {m.group(2)}",
        content,
    )

    # Handle imports of specific variables
    content = re.sub(
        f"from {package_name}([.\\w]+) import ([\\w, ]+)",
        lambda m: f"from {m.group(1).lstrip('.')} import {m.group(2)}",
        content,
    )

    # Remove package name from remaining imports
    content = re.sub(f"from {package_name}([.\\w]+) import", lambda m: f"from {m.group(1).lstrip('.')} import", content)
    content = re.sub(f"import {package_name}([.\\w]+)", lambda m: f"import {m.group(1).lstrip('.')}", content)

    # Remove any remaining leading dots from imports
    content = re.sub(r"from \.+([\w.]+) import", r"from \1 import", content)
    content = re.sub(r"import \.+([\w.]+)", r"import \1", content)

    if add_sys_path:
        # Add sys.path modification at the beginning of the file
        sys_path_addition = (
            "import sys\n"
            "import os\n"
            "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n"
        )
        content = sys_path_addition + content
        print(f"sys.path modification added to {main_file}")

    with open(file_path, "w") as f:
        f.write(content)


def prepare_dir_for_slurm(directory, main_file=None):
    """Iterates through all the files in the directory and modifies the imports in each file."""
    package_name = "targeted_llm_manipulation"
    main_filename = os.path.basename(main_file) if main_file else None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                add_sys_path = main_filename and file == main_filename
                modify_imports_in_file(file_path, package_name, add_sys_path)  # type: ignore


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python modify_imports.py <directory> [main_file]")
        sys.exit(1)

    directory = sys.argv[1]
    main_file = sys.argv[2] if len(sys.argv) == 3 else None

    prepare_dir_for_slurm(directory, main_file)
    print(f"Imports modified in {directory}")
