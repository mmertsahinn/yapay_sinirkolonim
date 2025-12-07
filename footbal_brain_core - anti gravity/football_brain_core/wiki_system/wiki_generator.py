import os
import ast
import inspect
from pathlib import Path
from typing import List, Dict, Optional

class WikiGenerator:
    """
    Auto-generates documentation for the codebase, similar to a simplified Code Wiki.
    
    Features:
    - Scans Python files for classes and functions.
    - Extracts docstrings.
    - Generates Markdown files.
    """
    
    def __init__(self, root_dir: str, output_dir: str):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.structure = {}

    def scan_codebase(self, max_files=None):
        """Scans the codebase and builds a structure of modules, classes, and functions."""
        count = 0
        print(f"Scanning directory: {self.root_dir.absolute()}")
        for root, dirs, files in os.walk(self.root_dir):
            # Skip hidden directories, venv, and cache
            dirs[:] = [d for d in dirs if not d.startswith('.') and 'venv' not in d and '__' not in d and 'wiki_docs' not in d]
            
            for file in files:
                if file.endswith(".py"):
                    full_path = Path(root) / file
                    # print(f"Parsing: {full_path}") # Reduce noise for full scan
                    self._parse_file(full_path)
                    count += 1
                    if max_files and count >= max_files:
                        print(f"Reached limit of {max_files} files.")
                        return
        print(f"Scanned {count} files.")

    def _parse_file(self, file_path: Path):
        """Parses a single Python file to extract docstrings and signatures."""
        try:
            relative_path = file_path.relative_to(self.root_dir)
            module_name = str(relative_path).replace(os.sep, ".")
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            try:
                tree = ast.parse(content)
            except SyntaxError:
                print(f"SyntaxError in {file_path}, skipping.")
                return

            module_doc = ast.get_docstring(tree)
            self.structure[str(relative_path)] = {
                "doc": module_doc,
                "classes": [],
                "functions": []
            }
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    self._parse_class(node, str(relative_path))
                elif isinstance(node, ast.FunctionDef):
                    self._parse_function(node, str(relative_path))
                    
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    def _parse_class(self, node: ast.ClassDef, module_path: str):
        class_info = {
            "name": node.name,
            "doc": ast.get_docstring(node),
            "methods": []
        }
        
        for item in node.body:
             if isinstance(item, ast.FunctionDef):
                 class_info["methods"].append({
                     "name": item.name,
                     "doc": ast.get_docstring(item),
                     "args": [a.arg for a in item.args.args]
                 })
        
        self.structure[module_path]["classes"].append(class_info)

    def _parse_function(self, node: ast.FunctionDef, module_path: str):
        func_info = {
            "name": node.name,
            "doc": ast.get_docstring(node),
            "args": [a.arg for a in node.args.args]
        }
        self.structure[module_path]["functions"].append(func_info)

    def clean_output(self):
        """Cleans the output directory."""
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
            print(f"Cleaned output directory: {self.output_dir}")

    def generate_markdown(self):
        """Generates Markdown files from the scanned structure."""
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        index_content = "# Code Wiki\n\n## Modules\n\n"

        for file_path, info in self.structure.items():
            safe_filename = str(file_path).replace(os.sep, "_").replace(".py", ".md")
            md_path = self.output_dir / safe_filename
            
            # Link formatting
            link_path = safe_filename
            index_content += f"- [{file_path}]({link_path})\n"
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Module: {file_path}\n\n")
                if info["doc"]:
                    f.write(f"{info['doc']}\n\n")
                
                if info["classes"]:
                    f.write("## Classes\n\n")
                    for cls in info["classes"]:
                        f.write(f"### {cls['name']}\n")
                        if cls["doc"]:
                            f.write(f"{cls['doc']}\n\n")
                        
                        if cls["methods"]:
                            f.write("#### Methods\n")
                            for method in cls["methods"]:
                                f.write(f"- **{method['name']}**({', '.join(method['args'])})\n")
                                if method["doc"]:
                                    f.write(f"  - {method['doc']}\n")
                                f.write("\n")
                
                if info["functions"]:
                    f.write("## Functions\n\n")
                    for func in info["functions"]:
                        f.write(f"### {func['name']}({', '.join(func['args'])})\n")
                        if func["doc"]:
                            f.write(f"{func['doc']}\n\n")

        with open(self.output_dir / "Home.md", "w", encoding="utf-8") as f:
            f.write(index_content)
        
        print(f"Generated {len(self.structure)} markdown files in {self.output_dir}")

if __name__ == "__main__":
    # Example usage
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")
    generator = WikiGenerator(root_dir=".", output_dir="wiki_docs")
    generator.clean_output()
    generator.scan_codebase() # Full scan
    generator.generate_markdown()
    print("Wiki generation complete.")
