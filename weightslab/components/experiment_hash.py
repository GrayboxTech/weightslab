"""
Experiment Hash Generation Module

This module generates stable, deterministic hashes for experiment tracking.
The hash is based on three key components:
    1. Model architecture (structure and layer configuration) - 8 bytes
    2. Hyperparameters (config values) - 8 bytes
    3. Data state (UIDs, discard status, tags) - 8 bytes

Combined into a 24-byte hash that allows tracking what changed between versions.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Set

import torch as th


logger = logging.getLogger(__name__)


class ExperimentHashGenerator:
    """Generates deterministic hashes for experiment tracking.

    Computes three separate 8-byte hashes:
    - Hyperparameters hash (learning rate, batch size, etc.)
    - Model architecture hash (layers, parameters, structure)
    - Data hash (UIDs, discard status, tags)

    These are combined into a final 24-byte hash in order: HP_MODEL_DATA
    This allows tracking what changed between experiment versions.

    Attributes:
        _last_hash (str): The most recently generated combined hash (24 chars)
        _last_hp_hash (str): Hash of the hyperparameters (8 chars)
        _last_model_hash (str): Hash of the model architecture (8 chars)
        _last_data_hash (str): Hash of the data state (8 chars)
    """

    def __init__(self):
        self._last_hash: Optional[str] = None
        self._last_hp_hash: Optional[str] = None
        self._last_model_hash: Optional[str] = None
        self._last_data_hash: Optional[str] = None

    def generate_hash(
        self,
        model: Optional[th.nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        data_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a unique hash for the current experiment configuration.

        Computes three separate 8-byte hashes and combines them into a 24-byte hash.
        This allows tracking which component changed (model, config, or data).

        Args:
            model: PyTorch model to hash (architecture only, not weights)
            config: Dictionary of hyperparameters
            data_state: Dictionary with 'uids', 'discarded', 'tags' for data samples

        Returns:
            str: A 24-character hexadecimal hash string (8 + 8 + 8)
        """
        # Generate individual 8-byte hashes
        hp_hash = self._hash_config(config) if config is not None else "00000000"
        model_hash = self._hash_model(model) if model is not None else "00000000"
        data_hash = self._hash_data_state(data_state) if data_state is not None else "00000000"

        # Combine into 24-byte hash: HP (8) + MODEL (8) + DATA (8)
        final_hash = f"{hp_hash}{model_hash}{data_hash}"

        # Store for comparison
        self._last_hash = final_hash
        self._last_hp_hash = hp_hash
        self._last_model_hash = model_hash
        self._last_data_hash = data_hash

        logger.info(f"Generated experiment hash: {final_hash}")
        logger.debug(f"  HP hash: {hp_hash}")
        logger.debug(f"  Model hash: {model_hash}")
        logger.debug(f"  Data hash: {data_hash}")

        return final_hash

    def has_changed(
        self,
        model: Optional[th.nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        data_state: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> tuple[bool, Set[str]]:
        """Check if the experiment configuration has changed.

        Checks all three components: model architecture, hyperparameters, and data.

        Args:
            model: PyTorch model to check
            config: Dictionary of hyperparameters to check
            data_state: Dictionary with data state (uids, discarded, tags)

        Returns:
            tuple: (has_changed: bool, changed_components: Set[str])
                where changed_components can contain 'model', 'config', 'data'
        """
        changed_components = set()

        # Check HP
        if config is not None:
            hp_hash = self._hash_config(config)
            if hp_hash != self._last_hp_hash or force:
                changed_components.add('hp')

        # Check model
        if model is not None:
            model_hash = self._hash_model(model)
            if model_hash != self._last_model_hash or force:
                changed_components.add('model')

        # Check data
        if data_state is not None:
            data_hash = self._hash_data_state(data_state)
            if data_hash != self._last_data_hash or force:
                changed_components.add('data')

        has_changed = len(changed_components) > 0

        if has_changed:
            logger.info(f"Experiment configuration changed: {changed_components}")

        return has_changed, changed_components

    def _hash_model(self, model: th.nn.Module) -> str:
        """Generate a hash from model architecture.

        This captures the model structure (layer types, parameters, connections)
        but not the actual weights, so the same architecture always produces
        the same hash.

        Args:
            model: PyTorch model

        Returns:
            str: Hash of model architecture (8 bytes)

        TODO (GP): Hash should be generated directly from the model class and computed on demand. Same for data and HP.
        Maybe later neurons tracking and values in the hash.
        """
        try:
            # Get model architecture info
            arch_info = []

            # Model class name
            arch_info.append(f"class:{model.__class__.__name__}")

            # Layer structure
            for name, module in model.named_modules():
                # Remove these trackers from hash
                if 'train_dataset_tracker' in name or 'eval_dataset_tracker' in name:
                    continue
                if name:  # Skip root module
                    module_info = f"{name}:{module.__class__.__name__}"

                    # Add key parameters for common layer types
                    if isinstance(module, th.nn.Module) and hasattr(module, 'in_neurons') and hasattr(module, 'out_neurons'):
                        module_info += f"_in{module.in_neurons}_out{module.out_neurons}"
                    if hasattr(module, 'operation_age'):
                        for op_type, age in module.operation_age.items():
                            module_info += f"_{op_type}->{age}"

                    arch_info.append(module_info)

            # Create hash from architecture description (8 bytes = 8 chars hex)
            arch_str = "|".join(sorted(arch_info))
            return hashlib.sha256(arch_str.encode()).hexdigest()[:8]

        except Exception as e:
            logger.warning(f"Failed to hash model architecture: {e}")
            # Fallback: use model repr (8 bytes)
            try:
                return hashlib.sha256(str(model).encode()).hexdigest()[:8]
            except Exception:
                return "00000000"

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate a hash from hyperparameters configuration.

        Args:
            config: Dictionary of hyperparameters

        Returns:
            str: Hash of configuration (8 bytes)
        """
        # Remove random state from config, i.e., root log dir as can be generated randomly
        config_cp = config.copy()
        config_cp.pop('root_log_dir', None)
        config_cp.pop('is_training', None)

        try:
            # Sort keys for deterministic hashing
            # Convert to JSON string for stable representation
            config_str = json.dumps(config_cp, sort_keys=True, default=str)
            return hashlib.sha256(config_str.encode()).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"Failed to hash config: {e}")
            return "00000000"

    def _hash_data_state(self, data_state: Dict[str, Any]) -> str:
        """Generate a hash from data state (UIDs, discard status, tags).

        Args:
            data_state: Dictionary with 'uids', 'discarded', 'tags'
                - uids: List of sample UIDs
                - discarded: Set of discarded UIDs
                - tags: Dict mapping UID to list of tags

        Returns:
            str: Hash of data state (8 bytes)
        """
        try:
            # Extract components
            uids = list(data_state.get('discarded', dict()).keys())
            discarded = data_state.get('discarded', dict())
            tags = data_state.get('tags', {})

            # Create deterministic representation
            # Sort UIDs and include discard status and tags
            data_info = []
            for uid in sorted(uids):
                is_discarded = discarded[uid]
                uid_tags = sorted(tags.get(uid, []))
                data_info.append(f"{uid}:d{int(is_discarded)}:t{','.join(uid_tags)}")

            data_str = "|".join(data_info)
            return hashlib.sha256(data_str.encode()).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"Failed to hash data state: {e}")
            return "00000000"

    def get_last_hash(self) -> Optional[str]:
        """Get the most recently generated hash.

        Returns:
            str or None: Last generated hash (24 bytes), or None if no hash generated yet
        """
        return self._last_hash

    def get_component_hashes(self) -> Dict[str, Optional[str]]:
        """Get individual component hashes.

        Returns:
            dict: Dictionary with 'hp', 'model', 'data' hash values (8 bytes each)
                 and 'combined' (24 bytes total)
        """
        return {
            'hp': self._last_hp_hash,
            'model': self._last_model_hash,
            'data': self._last_data_hash,
            'combined': self._last_hash
        }

    def restore_hashes(
        self,
        component_hashes: Optional[Dict[str, Optional[str]]] = None,
        combined_hash: Optional[str] = None
    ) -> None:
        """Restore last known hashes so change detection stays consistent.

        This is used when reloading manager state to keep the hash generator
        in sync with the previously computed hashes. If a combined hash is
        provided, it is split into component hashes when they are missing.
        """
        hashes = component_hashes or {}

        hp_hash = hashes.get('hp') if isinstance(hashes, dict) else None
        model_hash = hashes.get('model') if isinstance(hashes, dict) else None
        data_hash = hashes.get('data') if isinstance(hashes, dict) else None

        combined = (hashes.get('combined') if isinstance(hashes, dict) else None) or combined_hash

        # If we have a combined hash, use it to fill missing components
        if combined and len(str(combined)) >= 24:
            combined_str = str(combined)[:24]
            hp_hash = hp_hash or combined_str[0:8]
            model_hash = model_hash or combined_str[8:16]
            data_hash = data_hash or combined_str[16:24]
            combined = combined_str

        # If components are present but combined is missing, rebuild combined
        if not combined and hp_hash and model_hash and data_hash:
            combined = f"{hp_hash}{model_hash}{data_hash}"

        self._last_hp_hash = hp_hash
        self._last_model_hash = model_hash
        self._last_data_hash = data_hash
        self._last_hash = combined

        logger.debug(
            f"Restored hashes hp={hp_hash}, model={model_hash}, data={data_hash}, combined={combined}"
        )

    def compare_hashes(self, hash1: str, hash2: str) -> Set[str]:
        """Compare two 24-byte hashes and identify what changed.

        Args:
            hash1: First hash (24 chars)
            hash2: Second hash (24 chars)

        Returns:
            Set of changed components: 'model', 'config', 'data'
        """
        if len(hash1) != 24 or len(hash2) != 24:
            logger.warning(f"Invalid hash lengths: {len(hash1)}, {len(hash2)}")
            return set()

        changed = set()

        # Compare each 8-byte segment (HP_MODEL_DATA)
        if hash1[0:8] != hash2[0:8]:
            changed.add('hp')
        if hash1[8:16] != hash2[8:16]:
            changed.add('model')
        if hash1[16:24] != hash2[16:24]:
            changed.add('data')

        return changed
