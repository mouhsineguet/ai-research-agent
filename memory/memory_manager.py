import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class MemoryManager:
    """
    Manages research memory: stores, retrieves, and manages user interactions and research context.
    """
    def __init__(self, persist_path: Optional[str] = None):
        self.interactions: List[Dict[str, Any]] = []
        self.persist_path = persist_path
        if self.persist_path:
            self._load()

    def add_interaction(self, query: str, result: str, feedback: Optional[str] = None):
        """Add a new interaction to memory."""
        entry = {
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback or ""
        }
        self.interactions.append(entry)
        if self.persist_path:
            self._save()

    def get_context(self, n: int = 5) -> str:
        """Retrieve the last n interactions as context (summarized)."""
        context_entries = self.interactions[-n:]
        context = "\n\n".join([
            f"Q: {entry['query']}\nA: {entry['result']}" for entry in context_entries
        ])
        return context

    def clear(self):
        """Clear all memory."""
        self.interactions.clear()
        if self.persist_path:
            self._save()

    def _save(self):
        """Persist memory to disk (if enabled)."""
        try:
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(self.interactions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MemoryManager] Failed to save memory: {e}")

    def _load(self):
        """Load memory from disk (if enabled)."""
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                self.interactions = json.load(f)
        except FileNotFoundError:
            self.interactions = []
        except Exception as e:
            print(f"[MemoryManager] Failed to load memory: {e}") 