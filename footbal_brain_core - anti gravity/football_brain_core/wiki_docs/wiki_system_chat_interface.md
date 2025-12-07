# Module: wiki_system\chat_interface.py

## Classes

### ChatInterface
A simple interface to query the generated Code Wiki.

#### Methods
- **__init__**(self, wiki_dir)

- **load_index**(self)
  - Loads the markdown files into a simple memory index.

- **search**(self, query)
  - Searches the index for the query string.

- **start_loop**(self)
  - Starts the interactive loop.

