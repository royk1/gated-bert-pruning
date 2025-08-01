#!/usr/bin/env python3
"""
Backup script that creates a backup of Python files, shell scripts, results directories, and readme.md
Usage: python backup.py -m "Your message text here"
"""

import os
import zipfile
import argparse
import glob
from datetime import datetime

def get_script_directory():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def create_backup_directory():
    """Create backups directory if it doesn't exist in the script's directory."""
    script_dir = get_script_directory()
    backup_dir = os.path.join(script_dir, "backups")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"Created directory: {backup_dir}")
    return backup_dir

def generate_file_structure():
    """Generate a tree-like file structure of the current directory and subdirectories."""
    script_dir = get_script_directory()
    
    def get_tree_structure(path, prefix="", max_depth=3, current_depth=0):
        """Recursively build tree structure."""
        if current_depth >= max_depth:
            return ""
        
        items = []
        try:
            entries = sorted(os.listdir(path))
            # Filter out hidden files, __pycache__, and backups
            entries = [e for e in entries if not e.startswith('.') and e != '__pycache__' and e != 'backups']
            
            for i, entry in enumerate(entries):
                entry_path = os.path.join(path, entry)
                is_last = i == len(entries) - 1
                
                if os.path.isdir(entry_path):
                    # Directory
                    current_prefix = "└── " if is_last else "├── "
                    items.append(f"{prefix}{current_prefix}{entry}/")
                    
                    # Recurse into subdirectory
                    extension_prefix = "    " if is_last else "│   "
                    subtree = get_tree_structure(
                        entry_path, 
                        prefix + extension_prefix, 
                        max_depth, 
                        current_depth + 1
                    )
                    if subtree:
                        items.append(subtree)
                else:
                    # File
                    current_prefix = "└── " if is_last else "├── "
                    items.append(f"{prefix}{current_prefix}{entry}")
        except PermissionError:
            pass
        
        return "\n".join(items)
    
    # Get directory name
    dir_name = os.path.basename(script_dir)
    tree = f"{dir_name}/\n{get_tree_structure(script_dir)}"
    return tree

def create_readme(message):
    """Create readme.md file with the provided message and file structure."""
    script_dir = get_script_directory()
    readme_path = os.path.join(script_dir, "readme.md")
    
    # Generate file structure
    file_structure = generate_file_structure()
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create readme content
    content = f"""# Project Backup

## Description
{message}

## Backup Information
- **Created:** {timestamp}
- **Directory:** {script_dir}

## File Structure
```
{file_structure}
```

## Files Included in Backup
This backup contains all Python files (*.py) and this readme.md from the main directory and all subdirectories.
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created {readme_path} with message and file structure")
    return readme_path

def get_backup_filename():
    """Generate backup filename with current date and time."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"backup_{timestamp}.zip"

def find_all_python_files(directory):
    """Find all Python files in directory and subdirectories."""
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories, __pycache__, and backups
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'backups']
        
        for file in files:
            if file.endswith('.py'):
                # Get relative path from the starting directory
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, directory)
                python_files.append(rel_path)
    
    return python_files

def find_all_files_by_extension(directory, extensions):
    """Find all files in directory and subdirectories matching given extensions (list of .ext)."""
    matched_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories, __pycache__, and backups
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'backups']
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    matched_files.append(rel_path)
                    break
    return matched_files

def find_all_results_directories(directory):
    """Find all results_* directories in the given directory."""
    results_dirs = []
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and item.startswith('results_'):
                results_dirs.append(item)
    except PermissionError:
        pass
    return results_dirs

def create_backup_zip(backup_dir, filename):
    """Create zip file with all .py, .sh, .log, .png, .json, .bak (from bak/), results_* directories, and readme.md in script's directory and subdirectories."""
    script_dir = get_script_directory()
    original_cwd = os.getcwd()
    os.chdir(script_dir)
    try:
        # Find all relevant files
        py_files = find_all_python_files('.')
        log_files = find_all_files_by_extension('.', ['.log'])
        png_files = find_all_files_by_extension('.', ['.png'])
        json_files = find_all_files_by_extension('.', ['.json'])
        sh_files = find_all_files_by_extension('.', ['.sh'])
        # .bak files only from bak/ if exists
        bak_files = []
        if os.path.exists('bak'):
            bak_files = find_all_files_by_extension('bak', ['.bak'])
            bak_files = [os.path.join('bak', f) if not f.startswith('bak'+os.sep) else f for f in bak_files]
        # results_* directories
        results_dirs = find_all_results_directories('.')
        results_files = []
        for results_dir in results_dirs:
            if os.path.exists(results_dir):
                # Add all files from results directories
                for root, dirs, files in os.walk(results_dir):
                    # Skip backup files within results directories
                    dirs[:] = [d for d in dirs if not d.startswith('backup_')]
                    for file in files:
                        if not file.startswith('backup_'):
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, '.')
                            results_files.append(rel_path)
        # readme.md
        readme_file = 'readme.md'
        files_to_backup = py_files + log_files + png_files + json_files + sh_files + bak_files + results_files
        if os.path.exists(readme_file):
            files_to_backup.append(readme_file)
        # Remove duplicates
        files_to_backup = list(sorted(set(files_to_backup)))
        if not files_to_backup:
            print("No files found to backup!")
            return None
        backup_path = os.path.join(backup_dir, filename)
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files_to_backup:
                if os.path.exists(file):
                    zipf.write(file)
                    print(f"Added {file} to backup")
        print(f"Backup created: {backup_path}")
        print(f"Total files backed up: {len(files_to_backup)}")
        if results_dirs:
            print(f"Results directories included: {', '.join(results_dirs)}")
        return backup_path
    finally:
        os.chdir(original_cwd)

def main():
    """Main function to orchestrate the backup process."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a backup of Python files, shell scripts, results directories, and readme.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backup.py -m "Initial backup of the project"
  python backup.py -m "Added new features and bug fixes"
        """
    )
    
    parser.add_argument(
        '-m', '--message',
        type=str,
        required=True,
        help='Message to write in readme.md file'
    )
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        return
    
    # Execute backup process
    try:
        script_dir = get_script_directory()
        print(f"Starting backup process in: {script_dir}")
        
        # Step 1: Create backups directory if needed
        backup_dir = create_backup_directory()
        
        # Step 2: Create readme.md with message and file structure
        create_readme(args.message)
        
        # Step 3: Create backup zip file
        backup_filename = get_backup_filename()
        backup_path = create_backup_zip(backup_dir, backup_filename)
        
        if backup_path:
            print(f"\n✅ Backup completed successfully!")
            print(f"📁 Backup location: {backup_path}")
            print(f"📂 Working directory: {script_dir}")
        else:
            print("\n❌ Backup failed - no files to backup")
            
    except Exception as e:
        print(f"\n❌ Error during backup: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())