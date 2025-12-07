# Module: wiki_system\wiki_generator.py

## Classes

### WikiGenerator
Auto-generates documentation for the codebase, similar to a simplified Code Wiki.

Features:
- Scans Python files for classes and functions.
- Extracts docstrings.
- Generates Markdown files.

#### Methods
- **__init__**(self, root_dir, output_dir)

- **scan_codebase**(self, max_files)
  - Scans the codebase and builds a structure of modules, classes, and functions.

- **_parse_file**(self, file_path)
  - Parses a single Python file to extract docstrings and signatures.

- **_parse_class**(self, node, module_path)

- **_parse_function**(self, node, module_path)

- **clean_output**(self)
  - Cleans the output directory.

- **generate_markdown**(self)
  - Generates Markdown files from the scanned structure.

