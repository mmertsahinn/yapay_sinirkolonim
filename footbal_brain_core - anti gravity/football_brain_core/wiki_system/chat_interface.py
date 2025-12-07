import os
from pathlib import Path

class ChatInterface:
    """
    A simple interface to query the generated Code Wiki.
    """
    def __init__(self, wiki_dir: str):
        self.wiki_dir = Path(wiki_dir)
        self.index = {}
        self.load_index()

    def load_index(self):
        """Loads the markdown files into a simple memory index."""
        if not self.wiki_dir.exists():
            print("Wiki directory not found. Please run wiki_generator.py first.")
            return

        print(f"Loading Wiki from {self.wiki_dir}...")
        for file in os.listdir(self.wiki_dir):
            if file.endswith(".md") and file != "Home.md":
                with open(self.wiki_dir / file, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.index[file] = content.lower()
        print(f"Loaded {len(self.index)} documents.")

    def search(self, query: str):
        """Searches the index for the query string."""
        query = query.lower()
        results = []
        
        for filename, content in self.index.items():
            if query in content:
                # Simple score: count occurrences
                score = content.count(query)
                results.append((filename, score))
        
        # Sort by score desc
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def start_loop(self):
        """Starts the interactive loop."""
        print("\nWelcome to Code Wiki Chat! (Type 'exit' to quit)")
        while True:
            query = input("\nAsk about the codebase: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            results = self.search(query)
            if results:
                print(f"\nFound {len(results)} relevant documents:")
                for filename, score in results[:5]: # Top 5
                    print(f"- {filename} (Matches: {score})")
            else:
                print("No results found.")

if __name__ == "__main__":
    wiki_path = Path("wiki_docs") # Default to relative wiki_docs
    if not wiki_path.exists():
         # Fallback for running from wiki_system dir
         wiki_path = Path("../wiki_docs")
         
    chat = ChatInterface(wiki_dir=str(wiki_path))
    chat.start_loop()
