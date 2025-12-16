"""
Comprehensive experiment dump/restore functionality.
Saves and loads complete experiment state including:
- Model architecture and weights
- Optimizer state
- Training step counter
- Hyperparameters
- Metrics/signals history
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch as th

from weightslab.backend.ledgers import (
    get_model,
    get_optimizer,
    list_hyperparams,
    get_hyperparams,
    set_hyperparam,
)

_logger = logging.getLogger("experiment_dump")


class ExperimentDumper:
    """
    Manages complete experiment state persistence.
    """
    
    def __init__(self, root_log_dir: str):
        self.root_log_dir = Path(root_log_dir)
        self.root_log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.root_log_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Tracking for current experiment state
        self.current_step = 0
        self.metrics_history = {}  # {metric_name: [(step, value), ...]}
        
        # Experiment lineage tracking (git-like branches)
        self.current_experiment_id = self._generate_experiment_id()
        self.parent_experiment_id = None
        self.experiment_tree = {}  # {experiment_id: {parent_id, metadata, checkpoints}}
        self._load_experiment_tree()
        
        # Ramification tracking
        self.last_signature = self._compute_experiment_signature()
        self.auto_dump_on_change = True  # Enable automatic dumps before modifications
        self.pending_ramification = False
        
        # Resume-check gating: only branch/dump on resume after a change
        self.resume_check_pending = False
        self.resume_check_signature = None
        self.resume_check_reason = ""
        
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _load_experiment_tree(self):
        """Load the experiment tree from disk."""
        tree_file = self.root_log_dir / 'experiment_tree.json'
        if tree_file.exists():
            try:
                with open(tree_file, 'r') as f:
                    self.experiment_tree = json.load(f)
                _logger.info(f"Loaded experiment tree with {len(self.experiment_tree)} experiments")
            except Exception as e:
                _logger.warning(f"Could not load experiment tree: {e}")
                self.experiment_tree = {}
    
    def _save_experiment_tree(self):
        """Save the experiment tree to disk."""
        tree_file = self.root_log_dir / 'experiment_tree.json'
        try:
            with open(tree_file, 'w') as f:
                json.dump(self.experiment_tree, f, indent=2)
        except Exception as e:
            _logger.warning(f"Could not save experiment tree: {e}")
    
    def _compute_experiment_signature(self) -> Dict[str, Any]:
        """
        Compute a signature of the current experiment configuration.
        Used to detect ramifications (architecture or hyperparameter changes).
        """
        signature = {}
        
        # Get model architecture signature
        try:
            model = get_model(None)
            if model is not None:
                if hasattr(model, 'get') and callable(model.get):
                    model = model.get()
                
                signature['model_class'] = type(model).__name__
                signature['num_parameters'] = sum(p.numel() for p in model.parameters())
                
                # Create architecture fingerprint from layer structure
                layer_structure = []
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Leaf modules only
                        layer_structure.append(f"{name}:{type(module).__name__}")
                signature['architecture_fingerprint'] = hash(tuple(layer_structure))
        except Exception:
            pass
        
        # Get hyperparameters signature
        try:
            hp_names = list_hyperparams()
            hp_signature = {}
            for name in hp_names:
                hp = get_hyperparams(name)
                if hp:
                    # Only include training-critical hyperparameters
                    for key in ['learning_rate', 'batch_size', 'optimizer', 'loss_function']:
                        if key in hp:
                            hp_signature[key] = hp[key]
            signature['hyperparameters'] = hp_signature
        except Exception:
            pass
        
        return signature
    
    def _detect_ramification(self, previous_signature: Dict[str, Any]) -> bool:
        """
        Detect if current configuration differs from previous (indicates ramification).
        """
        current_signature = self._compute_experiment_signature()
        
        # Compare architecture
        if previous_signature.get('architecture_fingerprint') != current_signature.get('architecture_fingerprint'):
            _logger.info("🔀 Architecture change detected - creating new experiment branch")
            return True
        
        # Compare critical hyperparameters
        prev_hp = previous_signature.get('hyperparameters', {})
        curr_hp = current_signature.get('hyperparameters', {})
        
        if prev_hp != curr_hp:
            _logger.info("🔀 Hyperparameter change detected - creating new experiment branch")
            return True
        
        return False
        
    def check_for_changes(self) -> bool:
        """
        Check if architecture or hyperparameters have changed.
        If changes detected and auto_dump enabled, trigger automatic dump.
        If in reload mode (after reload_and_branch), creates new experiment branch on first modification.
        Returns True if ramification was detected.
        """
        current_signature = self._compute_experiment_signature()
        
        if self._detect_ramification(self.last_signature):
            # Check if we need to create a new branch due to reload + modification
            # This happens when reload_and_branch was called and user modified the model
            if not self.pending_ramification and self.parent_experiment_id:
                # We're in reload mode - modifications detected, create new branch NOW
                _logger.info(f"🌳 Modifications detected after reload - creating new branch...")
                old_exp_id = self.current_experiment_id
                self.current_experiment_id = self._generate_experiment_id()
                _logger.info(f"✨ New branch created: {self.current_experiment_id} (parent: {old_exp_id})")
            
            if self.auto_dump_on_change:
                _logger.info("🔄 Auto-dumping before ramification...")
                # Dump current state before the change
                self.dump_experiment(
                    step=self.current_step,
                    is_ramification=False,  # This is the "before" dump
                    additional_metadata={'auto_dump_reason': 'pre_ramification'}
                )
            
            # Mark that next dump should be a ramification
            self.pending_ramification = True
            self.last_signature = current_signature
            return True
        
        return False

    # ------------------------------------------------------------------
    # Resume-gated change detection
    # ------------------------------------------------------------------
    def request_resume_check(self, reason: str = ""):
        """Mark that a resume-time change check is required (captures pre-change signature)."""
        try:
            self.resume_check_signature = self._compute_experiment_signature()
        except Exception:
            self.resume_check_signature = self.last_signature
        self.resume_check_reason = reason
        self.resume_check_pending = True
        _logger.info(f"⏸️ Change requested ({reason or 'unspecified'}); will verify on resume")

    def perform_resume_check(self) -> Dict[str, Any]:
        """
        If a resume check is pending, compare signatures and branch/dump if changed.
        Returns a dict describing what happened.
        """
        if not self.resume_check_pending:
            return {'checked': False, 'changed': False, 'message': 'No pending resume check'}
        
        previous_signature = self.resume_check_signature or self.last_signature
        changed = self._detect_ramification(previous_signature)
        self.resume_check_pending = False
        self.resume_check_signature = None
        
        if not changed:
            self.last_signature = self._compute_experiment_signature()
            return {
                'checked': True,
                'changed': False,
                'message': 'No changes detected after resume'
            }
        
        # Prepare branch and dump as ramification
        self.pending_ramification = True
        checkpoint_path = self.dump_experiment(
            step=self.current_step,
            is_ramification=True,
            additional_metadata={
                'auto_dump_reason': 'resume_change',
                'resume_reason': self.resume_check_reason,
            },
        )
        self.last_signature = self._compute_experiment_signature()
        return {
            'checked': True,
            'changed': True,
            'message': 'Changes detected after resume; new branch created',
            'checkpoint_path': str(checkpoint_path),
            'experiment_id': self.current_experiment_id,
            'parent_experiment_id': self.parent_experiment_id,
        }
    
    def update_step(self, step: int):
        """Update the current training step counter."""
        self.current_step = step
        # Change detection is now gated to resume events; no periodic checks here
        
    def record_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Record a metric value at a given step with experiment ID for branch tracking."""
        if step is None:
            step = self.current_step
            
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append({
            'step': step,
            'value': float(value),
            'experiment_id': self.current_experiment_id  # Track which branch this metric belongs to
        })
    
    def dump_experiment(
        self,
        step: Optional[int] = None,
        model_name: Optional[str] = None,
        optimizer_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save complete experiment state to disk.
        Detects ramifications (architecture/hyperparameter changes) and creates new experiment branches.
        
        Args:
            step: Current training step (uses self.current_step if None)
            model_name: Name of model in ledger (auto-detect if None)
            optimizer_name: Name of optimizer in ledger (auto-detect if None)
            experiment_name: Name for this experiment
            additional_metadata: Extra metadata to save
            
        Returns:
            Path to the checkpoint directory
        """
        if step is None:
            step = self.current_step
        else:
            self.current_step = step
        
        # Compute current experiment signature
        current_signature = self._compute_experiment_signature()
        
        # Check if this is a ramification (new branch)
        is_ramification = self.pending_ramification  # Use pending flag from auto-detection
        
        if is_ramification or (self.current_experiment_id in self.experiment_tree):
            if not is_ramification:
                previous_signature = self.experiment_tree[self.current_experiment_id].get('signature', {})
                is_ramification = self._detect_ramification(previous_signature)
            
            if is_ramification:
                # Create new experiment branch
                self.parent_experiment_id = self.current_experiment_id
                self.current_experiment_id = self._generate_experiment_id()
                self.pending_ramification = False  # Clear flag
                _logger.info(f"🌿 New experiment branch: {self.current_experiment_id} (parent: {self.parent_experiment_id})")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint_step{step}_{timestamp}"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        _logger.info(f"Dumping experiment to: {checkpoint_path}")
        
        # 1. Save metadata with experiment lineage
        metadata = {
            'checkpoint_name': checkpoint_name,
            'timestamp': timestamp,
            'step': step,
            'created_at': datetime.now().isoformat(),
            'experiment_name': experiment_name or 'default_experiment',
            'experiment_id': self.current_experiment_id,
            'parent_experiment_id': self.parent_experiment_id,
            'is_ramification': is_ramification,
            'signature': current_signature
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        
        with open(checkpoint_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 2. Save hyperparameters from ledger
        hyperparams = {}
        try:
            hp_names = list_hyperparams()
            _logger.info(f"Found {len(hp_names)} hyperparameter groups: {hp_names}")
            for name in hp_names:
                hp = get_hyperparams(name)
                if hp:
                    hyperparams[name] = dict(hp)
                    _logger.info(f"  - {name}: {len(hp)} parameters")
                else:
                    _logger.warning(f"  - {name}: returned None/empty")
        except Exception as e:
            _logger.warning(f"Could not retrieve hyperparameters: {e}")
        
        # Add current step to hyperparameters
        hyperparams['current_step'] = step
        
        # Log what we're saving
        _logger.info(f"Saving {len(hyperparams)} hyperparameter groups to checkpoint")
        
        with open(checkpoint_path / 'hyperparameters.json', 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        # 3. Save metrics history
        with open(checkpoint_path / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # 4. Save model state
        try:
            model = get_model(model_name)
            if model is not None:
                # Handle proxy wrappers
                try:
                    if hasattr(model, 'get') and callable(model.get):
                        model = model.get()
                except Exception:
                    pass
                
                if hasattr(model, 'state_dict'):
                    model_state = model.state_dict()
                    th.save(model_state, checkpoint_path / 'model_state.pt')
                    _logger.info("[OK] Model state saved")
                    
                    # Also save model architecture info
                    model_info = {
                        'class': type(model).__name__,
                        'num_parameters': sum(p.numel() for p in model.parameters()),
                    }
                    with open(checkpoint_path / 'model_info.json', 'w') as f:
                        json.dump(model_info, f, indent=2)
        except Exception as e:
            _logger.warning(f"Could not save model: {e}")
        
        # 5. Save optimizer state
        try:
            optimizer = get_optimizer(optimizer_name)
            if optimizer is not None:
                # Handle proxy wrappers
                try:
                    if hasattr(optimizer, 'get') and callable(optimizer.get):
                        optimizer = optimizer.get()
                except Exception:
                    pass
                
                if hasattr(optimizer, 'state_dict'):
                    optimizer_state = optimizer.state_dict()
                    th.save(optimizer_state, checkpoint_path / 'optimizer_state.pt')
                    _logger.info("[OK] Optimizer state saved")
        except Exception as e:
            _logger.warning(f"Could not save optimizer: {e}")
        
        # 6. Update experiment tree
        if self.current_experiment_id not in self.experiment_tree:
            self.experiment_tree[self.current_experiment_id] = {
                'parent_id': self.parent_experiment_id,
                'created_at': metadata['created_at'],
                'signature': current_signature,
                'checkpoints': []
            }
        
        self.experiment_tree[self.current_experiment_id]['checkpoints'].append({
            'step': step,
            'checkpoint_name': checkpoint_name,
            'timestamp': timestamp,
            'is_ramification': is_ramification
        })
        
        self._save_experiment_tree()
        
        _logger.info(f"[OK] Experiment dump complete at step {step} (experiment: {self.current_experiment_id})")
        return checkpoint_path
    
    def load_experiment(
        self,
        checkpoint_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        optimizer_name: Optional[str] = None,
        restore_hyperparams: bool = True,
        merge_all_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Load complete experiment state from disk.
        
        Args:
            checkpoint_path: Path to checkpoint directory (loads latest if None)
            model_name: Name of model in ledger
            optimizer_name: Name of optimizer in ledger
            restore_hyperparams: Whether to restore hyperparameters to ledger
            merge_all_metrics: Whether to merge metrics from all checkpoints (True) or just the target checkpoint (False)
            
        Returns:
            Dictionary with metadata and loaded state info
        """
        # Auto-detect latest checkpoint if not specified
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                _logger.warning("No checkpoints found")
                return {'success': False, 'error': 'No checkpoints found'}
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            _logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return {'success': False, 'error': 'Checkpoint not found'}
        
        _logger.info(f"Loading experiment from: {checkpoint_path}")
        
        result = {'success': True, 'checkpoint_path': str(checkpoint_path)}
        
        # 1. Load metadata
        try:
            with open(checkpoint_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            result['metadata'] = metadata
            self.current_step = metadata.get('step', 0)
            
            # Restore experiment lineage
            self.current_experiment_id = metadata.get('experiment_id', self._generate_experiment_id())
            self.parent_experiment_id = metadata.get('parent_experiment_id')
            
            _logger.info(f"[OK] Metadata loaded - resuming from step {self.current_step} (experiment: {self.current_experiment_id})")
        except Exception as e:
            _logger.warning(f"Could not load metadata: {e}")
        
        # 2. Load hyperparameters
        if restore_hyperparams:
            try:
                with open(checkpoint_path / 'hyperparameters.json', 'r') as f:
                    hyperparams = json.load(f)
                
                # Restore to ledger
                for name, hp_dict in hyperparams.items():
                    if name == 'current_step':
                        continue
                    if isinstance(hp_dict, dict):
                        for key, value in hp_dict.items():
                            try:
                                set_hyperparam(name, key, value)
                            except Exception:
                                pass
                
                _logger.info("[OK] Hyperparameters restored")
                result['hyperparameters'] = hyperparams
            except Exception as e:
                _logger.warning(f"Could not load hyperparameters: {e}")
        
        # 3. Load metrics history - MERGE FROM ALL CHECKPOINTS
        if merge_all_metrics:
            merged_metrics = self._load_and_merge_all_metrics()
            self.metrics_history = merged_metrics
            result['metrics_loaded'] = len(self.metrics_history)
            result['metrics_sources'] = len(self._get_all_checkpoint_paths())
            _logger.info(f"[OK] Merged {len(self.metrics_history)} metric histories from {result['metrics_sources']} checkpoints")
        else:
            # Load only from target checkpoint
            try:
                with open(checkpoint_path / 'metrics_history.json', 'r') as f:
                    self.metrics_history = json.load(f)
                result['metrics_loaded'] = len(self.metrics_history)
                _logger.info(f"[OK] Loaded {len(self.metrics_history)} metric histories from target checkpoint")
            except Exception as e:
                _logger.warning(f"Could not load metrics: {e}")
        
        # 4. Load model state
        try:
            model = get_model(model_name)
            if model is not None:
                # Handle proxy wrappers
                try:
                    if hasattr(model, 'get') and callable(model.get):
                        model = model.get()
                except Exception:
                    pass
                
                model_state_path = checkpoint_path / 'model_state.pt'
                if model_state_path.exists():
                    model_state = th.load(model_state_path)
                    model.load_state_dict(model_state, strict=False)
                    _logger.info("[OK] Model state restored")
                    result['model_loaded'] = True
        except Exception as e:
            _logger.warning(f"Could not load model: {e}")
            result['model_loaded'] = False
        
        # 5. Load optimizer state
        try:
            optimizer = get_optimizer(optimizer_name)
            if optimizer is not None:
                # Handle proxy wrappers
                try:
                    if hasattr(optimizer, 'get') and callable(optimizer.get):
                        optimizer = optimizer.get()
                except Exception:
                    pass
                
                optimizer_state_path = checkpoint_path / 'optimizer_state.pt'
                if optimizer_state_path.exists():
                    optimizer_state = th.load(optimizer_state_path)
                    optimizer.load_state_dict(optimizer_state)
                    _logger.info("[OK] Optimizer state restored")
                    result['optimizer_loaded'] = True
        except Exception as e:
            _logger.warning(f"Could not load optimizer: {e}")
            result['optimizer_loaded'] = False
        
        _logger.info(f"[OK] Experiment load complete - ready to resume from step {self.current_step}")
        return result
    
    def reload_and_branch(
        self,
        checkpoint_path: Path,
        new_branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reload a checkpoint for \"time travel\".
        A new experiment branch will be created ONLY when modifications are detected.
        Until then, continues on the same experiment branch.
        
        This allows trying different modifications from a previous state.
        If you modify the model and continue training, a new branch is automatically created.
        
        Args:
            checkpoint_path: Path to checkpoint to reload
            new_branch_name: Optional name for the new branch (when created)
            
        Returns:
            Result dict with success status and reload info
        """
        _logger.info(f"⏮️ Reloading checkpoint from: {checkpoint_path}")
        
        # Load the checkpoint normally
        result = self.load_experiment(
            checkpoint_path=checkpoint_path,
            restore_model=True,
            restore_optimizer=True,
            restore_hyperparams=True,
            merge_all_metrics=False
        )
        
        if not result.get('success'):
            return result
        
        # DO NOT create a new branch immediately
        # Instead, mark that we're in "reload mode" - branch will be created if modifications detected
        metadata = result.get('metadata', {})
        reloaded_from_step = metadata.get('step', 0)
        reloaded_from_exp_id = metadata.get('experiment_id')
        
        # Update last signature to the reloaded state
        # This is critical: next check_for_changes() will compare against this baseline
        self.last_signature = self._compute_experiment_signature()
        
        # Stay on the current experiment branch until modifications are detected
        # check_for_changes() will handle branch creation if changes are made
        self.pending_ramification = False
        
        _logger.info(f"⏮️ Reloaded from step {reloaded_from_step} (exp: {reloaded_from_exp_id})")
        _logger.info(f"🔀 New branch will be created automatically if modifications are detected during next training")
        
        # Update result with reload info
        result['reloaded_from_step'] = reloaded_from_step
        result['reloaded_from_experiment_id'] = reloaded_from_exp_id
        result['branch_created'] = False
        result['message'] = f"Reloaded from step {reloaded_from_step}. New branch will be created on next modification."
        
        return result
    
    def _get_all_checkpoint_paths(self) -> list[Path]:
        """Get all checkpoint directories sorted by creation time."""
        if not self.checkpoints_dir.exists():
            return []
        
        checkpoints = []
        for item in self.checkpoints_dir.iterdir():
            if item.is_dir() and (item / 'metadata.json').exists():
                try:
                    with open(item / 'metadata.json', 'r') as f:
                        metadata = json.load(f)
                    created_at = metadata.get('created_at', '')
                    step = metadata.get('step', 0)
                    checkpoints.append((created_at, step, item))
                except Exception:
                    continue
        
        # Sort by creation time (oldest first)
        checkpoints.sort(key=lambda x: x[0])
        return [item[2] for item in checkpoints]
    
    def _load_and_merge_all_metrics(self) -> Dict[str, list]:
        """
        Load metrics from ALL checkpoints and merge them intelligently.
        
        Handles overlapping step ranges by keeping all unique (step, value) pairs
        and sorting by step. This allows visualization of complete training history
        even if checkpoints were saved with overlapping ranges.
        
        Example:
            Checkpoint 1: loss at steps [0, 10, 20, 30, 40]
            Checkpoint 2: loss at steps [25, 35, 45, 55]
            Merged: loss at steps [0, 10, 20, 25, 30, 35, 40, 45, 55]
        """
        all_checkpoints = self._get_all_checkpoint_paths()
        merged_metrics = {}
        
        _logger.info(f"Merging metrics from {len(all_checkpoints)} checkpoints...")
        
        for checkpoint_path in all_checkpoints:
            metrics_file = checkpoint_path / 'metrics_history.json'
            if not metrics_file.exists():
                continue
            
            try:
                with open(metrics_file, 'r') as f:
                    checkpoint_metrics = json.load(f)
                
                # Merge each metric
                for metric_name, data_points in checkpoint_metrics.items():
                    if metric_name not in merged_metrics:
                        merged_metrics[metric_name] = []
                    
                    # Add all points from this checkpoint
                    if isinstance(data_points, list):
                        for point in data_points:
                            if isinstance(point, dict) and 'step' in point and 'value' in point:
                                merged_metrics[metric_name].append({
                                    'step': point['step'],
                                    'value': point['value'],
                                    'source': checkpoint_path.name
                                })
                
            except Exception as e:
                _logger.warning(f"Could not load metrics from {checkpoint_path.name}: {e}")
        
        # Sort and deduplicate each metric by step
        for metric_name in merged_metrics:
            # Sort by step
            merged_metrics[metric_name].sort(key=lambda x: x['step'])
            
            # Remove duplicate steps (keep first occurrence)
            seen_steps = set()
            deduplicated = []
            for point in merged_metrics[metric_name]:
                step = point['step']
                if step not in seen_steps:
                    seen_steps.add(step)
                    deduplicated.append(point)
            
            merged_metrics[metric_name] = deduplicated
            
            _logger.debug(f"  {metric_name}: {len(deduplicated)} unique data points")
        
        return merged_metrics
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint directory."""
        if not self.checkpoints_dir.exists():
            return None
        
        checkpoints = []
        for item in self.checkpoints_dir.iterdir():
            if item.is_dir() and (item / 'metadata.json').exists():
                try:
                    with open(item / 'metadata.json', 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append((metadata.get('created_at', ''), item))
                except Exception:
                    continue
        
        if not checkpoints:
            return None
        
        # Sort by timestamp and return latest
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]
    
    def get_current_step(self) -> int:
        """Get the current training step counter."""
        return self.current_step
    
    def get_metrics_history(self) -> Dict[str, list]:
        """Get the full metrics history."""
        return self.metrics_history
    
    def get_dump_points(self) -> list[Dict[str, Any]]:
        """
        Get all dump/checkpoint points for marking in plots.
        Returns list of {step, is_ramification, experiment_id, timestamp}.
        """
        dump_points = []
        for exp_id, exp_data in self.experiment_tree.items():
            for checkpoint in exp_data.get('checkpoints', []):
                dump_points.append({
                    'step': checkpoint['step'],
                    'is_ramification': checkpoint.get('is_ramification', False),
                    'experiment_id': exp_id,
                    'timestamp': checkpoint['timestamp'],
                    'checkpoint_name': checkpoint['checkpoint_name']
                })
        
        # Sort by step
        dump_points.sort(key=lambda x: x['step'])
        return dump_points
    
    def get_experiment_tree_visualization(self) -> Dict[str, Any]:
        """
        Get experiment tree formatted for git-like visualization.
        Returns structure suitable for d3.js or similar tree rendering.
        """
        nodes = []
        links = []
        
        for exp_id, exp_data in self.experiment_tree.items():
            parent_id = exp_data.get('parent_id')
            checkpoints = exp_data.get('checkpoints', [])
            
            if not checkpoints:
                continue
            
            # Get step range for this experiment
            steps = [cp['step'] for cp in checkpoints]
            min_step = min(steps)
            max_step = max(steps)
            
            node = {
                'id': exp_id,
                'parent_id': parent_id,
                'step_range': [min_step, max_step],
                'checkpoint_count': len(checkpoints),
                'created_at': exp_data.get('created_at'),
                'checkpoints': checkpoints
            }
            nodes.append(node)
            
            # Create link to parent if exists
            if parent_id and parent_id in self.experiment_tree:
                links.append({
                    'source': parent_id,
                    'target': exp_id,
                    'branch_point': min_step
                })
        
        return {
            'nodes': nodes,
            'links': links,
            'current_experiment': self.current_experiment_id
        }
    
    def get_merged_metrics_for_visualization(self) -> Dict[str, Any]:
        """
        Get metrics formatted for UI visualization with checkpoint source information.
        
        Returns:
            Dictionary with metric traces grouped by metric name, including source checkpoint info.
        """
        visualization_data = {}
        
        for metric_name, data_points in self.metrics_history.items():
            if not data_points:
                continue
            
            # Group by source checkpoint for color coding
            by_source = {}
            for point in data_points:
                source = point.get('source', 'unknown')
                if source not in by_source:
                    by_source[source] = {'steps': [], 'values': []}
                by_source[source]['steps'].append(point['step'])
                by_source[source]['values'].append(point['value'])
            
            visualization_data[metric_name] = {
                'total_points': len(data_points),
                'step_range': [data_points[0]['step'], data_points[-1]['step']],
                'by_source': by_source,
                'all_steps': [p['step'] for p in data_points],
                'all_values': [p['value'] for p in data_points]
            }
        
        return visualization_data


# Global instance
_global_dumper: Optional[ExperimentDumper] = None


def initialize_experiment_dumper(root_log_dir: str) -> ExperimentDumper:
    """Initialize the global experiment dumper."""
    global _global_dumper
    _global_dumper = ExperimentDumper(root_log_dir)
    return _global_dumper


def get_experiment_dumper() -> Optional[ExperimentDumper]:
    """Get the global experiment dumper instance."""
    return _global_dumper
