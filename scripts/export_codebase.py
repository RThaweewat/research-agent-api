"""
Script for making file path structure for my README
"""

import os
from pathlib import Path
import json

def get_file_content(file_path: str) -> str:
    """Read and return file content with proper error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def should_ignore(path: str) -> bool:
    """Check if path should be ignored"""
    ignore_patterns = [
        '__pycache__',
        '.git',
        '.env',
        '.venv',
        'node_modules',
        '.pytest_cache',
        '.coverage',
        '.idea',
        '.vscode',
        'dist',
        'build',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.DS_Store',
    ]
    
    return any(pattern in path for pattern in ignore_patterns)

def export_codebase(root_dir: str, output_file: str):
    """Export codebase structure and content to a text file"""
    
    # Get project name from root directory
    project_name = os.path.basename(os.path.abspath(root_dir))
    
    # Initialize output content
    output = [
        f"# {project_name} Codebase Export",
        "\n## Project Structure\n",
    ]
    
    # Track all files for content export
    files_to_export = []
    
    # First, build directory structure
    for root, dirs, files in os.walk(root_dir):
        if should_ignore(root):
            continue
            
        # Calculate relative path and indent level
        rel_path = os.path.relpath(root, root_dir)
        indent = '  ' * (len(Path(rel_path).parts) - 1)
        
        # Add directory to structure
        if rel_path != '.':
            output.append(f"{indent}📁 {os.path.basename(root)}/")
        
        # Add files to structure and track for content export
        for file in sorted(files):
            if file.endswith(('.py', '.txt', '.json', '.yml', '.yaml', '.md', '.env.example')):
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(file_path, root_dir)
                output.append(f"{indent}  📄 {file}")
                files_to_export.append(rel_file_path)
    
    # Add file contents
    output.append("\n## File Contents\n")
    
    for file_path in files_to_export:
        abs_path = os.path.join(root_dir, file_path)
        output.extend([
            f"\n### 📄 {file_path}",
            "```" + (file_path.split('.')[-1] if '.' in file_path else ''),
            get_file_content(abs_path),
            "```\n"
        ])
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

if __name__ == "__main__":
    # Get the project root directory (assuming this script is in a scripts folder)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "codebase_export.txt")
    
    export_codebase(project_root, output_path)
    print(f"Codebase exported to: {output_path}") 