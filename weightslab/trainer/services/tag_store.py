"""
HDF5-backed tag storage for data annotations, keyed by UID.

Tags are stored at {root_log_dir}/data/tags.h5 and support:
- Multiple tags per UID (stored as comma-separated string)
- Auto-load on restart
- Stream-friendly: loads only requested UIDs
"""
import logging
import threading
from pathlib import Path
from typing import Dict, List, Iterable

import pandas as pd


logger = logging.getLogger(__name__)


class TagsStore:
    """HDF5 tag store with support for multiple tags per UID."""

    def __init__(self, root_log_dir: Path, filename: str = "tags.h5"):
        self.root_log_dir = Path(root_log_dir) / "data"
        self.root_log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_log_dir / filename
        self._lock = threading.RLock()
        logger.info(f"TagsStore initialized at {self.path}")

    def load_tags(self, uids: Iterable[int]) -> Dict[int, List[str]]:
        """Load tags for the specified UIDs from HDF5.
        
        Returns dict mapping uid -> list of tags (empty list if no tags).
        """
        ids = sorted({int(uid) for uid in (uids or [])})
        if not ids or not self.path.exists():
            return {}

        with self._lock:
            try:
                with pd.HDFStore(self.path, mode="r") as store:
                    if "tags" not in store:
                        return {}

                    # Stream-friendly: query only the requested UIDs
                    where_clause = f"uid in {ids}"
                    try:
                        df = store.select("tags", where=where_clause)
                    except Exception:
                        # Fallback if query fails
                        df = store.select("tags")
                        df = df[df["uid"].isin(ids)]
            except Exception as e:
                logger.warning(f"Failed to load tags: {e}")
                return {}

        result = {}
        for row in df.itertuples():
            uid = int(row.uid)
            tags_str = str(row.tags) if hasattr(row, 'tags') else ""
            # Parse comma-separated tags
            tag_list = [t.strip() for t in tags_str.split(",") if t.strip()]
            result[uid] = tag_list

        return result

    def save_tags(self, tag_map: Dict[int, List[str]]) -> None:
        """Persist tags to HDF5 (upsert: replaces existing tags for these UIDs).
        
        Args:
            tag_map: Dict mapping uid -> list of tags
        """
        if not tag_map:
            return

        # Convert tag lists to comma-separated strings
        df = pd.DataFrame({
            "uid": [int(k) for k in tag_map.keys()],
            "tags": [",".join(v) if v else "" for v in tag_map.values()],
        })

        with self._lock:
            with pd.HDFStore(self.path, mode="a", complevel=1, complib="blosc") as store:
                # Remove old entries for these UIDs
                if "tags" in store:
                    for uid in df["uid"]:
                        try:
                            store.remove("tags", where=f"uid=={int(uid)}")
                        except Exception:
                            pass
                # Append new/updated entries
                store.append("tags", df, format="table", data_columns=["uid"])

        logger.info(f"Saved tags for {len(tag_map)} UID(s) to {self.path}")

    def add_tags(self, uid: int, tags_to_add: List[str]) -> None:
        """Add tags to a UID without removing existing ones."""
        existing = self.load_tags([uid])
        current_tags = existing.get(uid, [])
        # Add new tags (avoid duplicates)
        for tag in tags_to_add:
            if tag not in current_tags:
                current_tags.append(tag)
        self.save_tags({uid: current_tags})

    def remove_tags(self, uid: int, tags_to_remove: List[str]) -> None:
        """Remove specific tags from a UID."""
        existing = self.load_tags([uid])
        current_tags = existing.get(uid, [])
        updated_tags = [t for t in current_tags if t not in tags_to_remove]
        self.save_tags({uid: updated_tags})

    def clear_tags(self, uid: int) -> None:
        """Remove all tags from a UID."""
        self.save_tags({uid: []})
