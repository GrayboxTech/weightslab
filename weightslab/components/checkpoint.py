import enum
import os
import json
import uuid
import logging
import torch as th
import dill
import pickle
from pathlib import Path
from typing import Set, Optional

from weightslab.backend.ledgers import (
    get_model,
    get_optimizer,
    get_dataloader,
    get_dataloaders,
    register_model,
    list_models,
    resolve_hp_name,
    set_hyperparam,
)

CHECKPOPINTS_METADATA_FILE_NAME = 'checkpoints.metadata'

_logger = logging.getLogger("checkpoint_manager")


class _CheckpointMetadataDictKeys(str, enum.Enum):
    NEXT_ID = 'next_id'
    PRNT_ID = 'parent_id'
    ID_2_PRNT = 'id_2_prnt'
    ID_2_PATH = 'id_2_path'
    ID_2_META = 'id_2_meta'


class _CheckpointDictKeys(str, enum.Enum):
    MODEL = 'model'
    MODEL_FULL = 'model_full'
    OPTIM = 'optimizer'
    LRATE = 'learning_rate'
    BSIZE = 'batch_size'
    DATASETS = 'datasets'
    ENAME = 'experiment_name'


class CheckpointManager(object):
    def __init__(self, root_directory: str = 'root_experiment') -> None:
        # Init paths
        self.root_directory = Path(root_directory)
        self.root_directory.mkdir(parents=True, exist_ok=True)
        self.root_directory = self.root_directory.absolute()  # Get abs path

        self.next_id = -1
        self.prnt_id = -1
        self.id_to_path = {}
        self.id_to_prnt = {}
        self.id_to_meta = {}

        # Load metadata from chkpt if exists
        self._load_metadata()

    def __repr__(self) -> str:
        return f'CheckpointManager(root_directory={self.root_directory})\n' + \
            f'next_id={self.next_id}\n' + \
            f'prnt_id={self.prnt_id}\n' + \
            f'id_to_prnt={self.id_to_prnt}\n' + \
            f'id_to_path={self.id_to_path}\n' + \
            f'id_to_meta={self.id_to_meta}\n'

    def get_ids(self) -> Set[int]:
        return set(self.id_to_path.keys())

    def get_path_for_id(self, id: int) -> Path:
        return self.id_to_path[id]

    def _generate_checkpoint_id(self):
        self.next_id += 1
        return self.next_id

    def attach_metadata(self, checkpoint_id: int, metadata: dict):
        if checkpoint_id in self.id_to_path:
            raise ValueError(f"Checkpoint {checkpoint_id} does not exist.")

        self.id_to_meta[checkpoint_id] = dict(metadata)

    def get_metadata(self, checkpoint_id: int) -> dict:
        if checkpoint_id in self.id_to_path:
            raise ValueError(f"Checkpoint {checkpoint_id} does not exist.")

        return self.id_to_meta[checkpoint_id]

    def get_latest_experiment(self):
        return self.next_id

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Return the most-recent checkpoint path saved in the manager, or None."""
        try:
            if not self.id_to_path:
                return None
            # id_to_path values may be strings
            paths = [Path(p) for p in self.id_to_path.values()]
            existing = [p for p in paths if p.exists()]
            if not existing:
                return None
            latest = max(existing, key=lambda p: p.stat().st_mtime)
            return latest
        except Exception:
            return None

    def _dump_metadata(self):
        state_dict = {
            _CheckpointMetadataDictKeys.NEXT_ID: self.next_id,
            _CheckpointMetadataDictKeys.PRNT_ID: self.prnt_id,
            _CheckpointMetadataDictKeys.ID_2_PATH: self.id_to_path,
            _CheckpointMetadataDictKeys.ID_2_PRNT: self.id_to_prnt,
            _CheckpointMetadataDictKeys.ID_2_META: self.id_to_meta
        }

        file_path = self.root_directory.joinpath(
            CHECKPOPINTS_METADATA_FILE_NAME)
        with open(file_path, 'w') as ckpt_metadata_file:
            ckpt_metadata_file.write(json.dumps(state_dict))

    def _load_metadata(self):
        state_dict = None
        file_path = self.root_directory.joinpath(
            CHECKPOPINTS_METADATA_FILE_NAME)

        if not file_path.exists():
            return

        with open(file_path, 'r') as ckpt_metadata_file:
            state_dict = json.loads(ckpt_metadata_file.read())

        if state_dict is None:
            return

        self.next_id = state_dict[_CheckpointMetadataDictKeys.NEXT_ID]
        self.prnt_id = state_dict[_CheckpointMetadataDictKeys.PRNT_ID]
        self.id_to_path = state_dict[_CheckpointMetadataDictKeys.ID_2_PATH]
        self.id_to_prnt = state_dict[_CheckpointMetadataDictKeys.ID_2_PRNT]

    def dump(self,
             model_name: Optional[str] = None,
             optimizer_name: Optional[str] = None,
             experiment_name: Optional[str] = None,
             override_filepath: Optional[str] = None,
             save_full_model: bool = False,
             ) -> int | str | None:
        """
        Dump a checkpoint using ledger-registered objects.

        If a specific name is provided, the corresponding object is used;
        otherwise the ledger's single registered object is used (if any).
        Returns the checkpoint id (int) for managed dumps, or -1 for an
        override filepath write.
        """
        # If override filepath requested, write a small metadata JSON and return
        if override_filepath:
            ckpt_save_path = self.root_directory.joinpath(Path(override_filepath))
            info = {
                "experiment_name": experiment_name,
            }
            # try to enrich info from ledger objects
            try:
                model = get_model(model_name)
                try:
                    info["model"] = type(model).__name__
                except Exception:
                    pass
            except Exception:
                pass

            try:
                with open(ckpt_save_path, 'w') as f:
                    json.dump(info, f, indent=2)
            except Exception:
                pass
            return -1

        # Normal managed dump: create id and path
        current_ckpt_id = self._generate_checkpoint_id()
        _logger.info("Dumping checkpoint: %d", current_ckpt_id)
        self.id_to_prnt[current_ckpt_id] = self.prnt_id
        self.prnt_id = current_ckpt_id
        file_name = "ckpt_" + str(current_ckpt_id) + "_" + str(uuid.uuid4())
        ckpt_save_path = self.root_directory.joinpath(Path(file_name))
        self.id_to_path[current_ckpt_id] = str(ckpt_save_path)
        self._dump_metadata()

        # Gather states defensively from ledger
        model_state = None
        optimizer_state = None
        datasets_state = {}
        batch_size = None
        learning_rate = None
        full_model_bytes = None

        # Model
        try:
            model = get_model(model_name)
            # If ledger returned a Proxy-like wrapper, try to get underlying object
            try:
                underlying = getattr(model, 'get', None)
                if callable(underlying):
                    model_obj = model.get()
                else:
                    model_obj = model
            except Exception:
                model_obj = model

            try:
                if model_obj is not None and hasattr(model_obj, 'state_dict'):
                    model_state = model_obj.state_dict()
                else:
                    model_state = None

                if save_full_model:
                    try:
                        # try cloudpickle first (can handle local classes/functions)
                        if dill is not None:
                            full_model_bytes = dill.dumps(model_obj)
                        else:
                            # fallback to standard pickle
                            full_model_bytes = pickle.dumps(model_obj)
                    except Exception:
                        try:
                            full_model_bytes = pickle.dumps(model_obj)
                        except Exception:
                            full_model_bytes = None
            except Exception:
                # if model is a Proxy or missing, skip
                model_state = None
        except Exception:
            model_state = None

        # Optimizer
        try:
            optim = get_optimizer(optimizer_name)
            try:
                # unwrap Proxy if present
                try:
                    opt_underlying = getattr(optim, 'get', None)
                    if callable(opt_underlying):
                        optim_obj = optim.get()
                    else:
                        optim_obj = optim
                except Exception:
                    optim_obj = optim

                optimizer_state = optim_obj.state_dict()
                # attempt to extract lr from param groups
                try:
                    lrs = []
                    for g in optim_obj.param_groups:
                        lrs.append(g.get('lr'))
                    if lrs:
                        learning_rate = lrs[0]
                except Exception:
                    learning_rate = None
            except Exception:
                optimizer_state = None
        except Exception:
            optimizer_state = None

        # All dataloaders dynamically from ledger
        try:
            loader_names = get_dataloaders()
            for loader_name in loader_names:
                try:
                    loader = get_dataloader(loader_name)
                    try:
                        # unwrap proxy if present
                        try:
                            loader_under = getattr(loader, 'get', None)
                            if callable(loader_under):
                                loader_obj = loader.get()
                            else:
                                loader_obj = loader
                        except Exception:
                            loader_obj = loader

                        # Get dataset state
                        if hasattr(loader_obj, 'dataset') and hasattr(loader_obj.dataset, 'state_dict'):
                            datasets_state[loader_name] = loader_obj.dataset.state_dict()

                        # Get batch size from first loader if not set
                        if batch_size is None:
                            try:
                                batch_size = getattr(loader_obj, 'batch_size', None)
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        payload = {
            _CheckpointDictKeys.ENAME: experiment_name,
            _CheckpointDictKeys.BSIZE: batch_size,
            _CheckpointDictKeys.LRATE: learning_rate,
            _CheckpointDictKeys.MODEL: model_state,
            _CheckpointDictKeys.MODEL_FULL: full_model_bytes,
            _CheckpointDictKeys.OPTIM: optimizer_state,
        }

        try:
            th.save(payload, ckpt_save_path)
        except Exception:
            _logger.exception("Failed to save checkpoint to %s", ckpt_save_path)

        return current_ckpt_id

    def load(self,
             checkpoint_id_or_path: int | str,
             model_name: Optional[str] = None,
             optimizer_name: Optional[str] = None,
             hyperparams_name: Optional[str] = None,
             ) -> None:
        """
        Load a checkpoint and apply contained states to ledger-registered objects.

        `checkpoint_id_or_path` may be an integer id managed by this manager,
        or a direct filesystem path.
        """
        _logger.info(f"Loading checkpoint: {checkpoint_id_or_path}")

        path = None
        try:
            if isinstance(checkpoint_id_or_path, (int,)):
                if checkpoint_id_or_path not in self.id_to_path:
                    _logger.warning(f"Checkpoint {checkpoint_id_or_path} not found")
                    return
                path = self.id_to_path[checkpoint_id_or_path]
            else:
                # try as path or string id
                if str(checkpoint_id_or_path) in self.id_to_path:
                    path = self.id_to_path[str(checkpoint_id_or_path)]
                elif os.path.exists(str(checkpoint_id_or_path)):
                    path = str(checkpoint_id_or_path)
                else:
                    _logger.warning(f"Checkpoint {checkpoint_id_or_path} not found")
                    return

            if not path or not os.path.exists(path):
                _logger.warning(f"Checkpoint file not found: {path}")
                return

            self.prnt_id = checkpoint_id_or_path
            ckpt_dict = th.load(path, weights_only=False)
        except Exception as e:
            _logger.error(f"Could not load checkpoint due to {str(e)}")
            return

        # Apply to model if present in ledger
        try:
            model = None
            try:
                model = get_model(model_name)
            except Exception:
                model = None

            if model is not None and ckpt_dict.get(_CheckpointDictKeys.MODEL) is not None:
                try:
                    model.load_state_dict(ckpt_dict[_CheckpointDictKeys.MODEL], strict=False)
                except Exception:
                    try:
                        model.load_state_dict(ckpt_dict[_CheckpointDictKeys.MODEL], strict=True)
                    except Exception:
                        _logger.exception("Failed to load model state from checkpoint; attempting full-model replacement if available")
                        # If a full-model object was saved, try to replace the registered model
                        try:
                            full_obj = ckpt_dict.get(_CheckpointDictKeys.MODEL_FULL, None)
                            # If bytes were stored, attempt to deserialize (cloudpickle preferred)
                            if isinstance(full_obj, (bytes, bytearray)):
                                try:
                                    if dill is not None:
                                        full_obj = dill.loads(full_obj)
                                    else:
                                        full_obj = pickle.loads(full_obj)
                                except Exception:
                                    try:
                                        full_obj = pickle.loads(full_obj)
                                    except Exception:
                                        full_obj = None

                            if full_obj is not None:
                                # Determine target ledger name: prefer provided model_name, else use sole registered model
                                target_name = model_name
                                if target_name is None:
                                    try:
                                        models = list_models()
                                        if len(models) == 1:
                                            target_name = models[0]
                                    except Exception:
                                        target_name = None

                                if target_name is not None:
                                    try:
                                        register_model(target_name, full_obj)
                                        _logger.info("Replaced ledger model '%s' with full-model from checkpoint", target_name)
                                    except Exception:
                                        _logger.exception("Failed to register full-model into ledger")
                                else:
                                    _logger.warning("No target model name available to register full-model from checkpoint")
                        except Exception:
                            _logger.exception("Unexpected error while attempting full-model replacement")
        except Exception:
            pass

        # Apply optimizer state if possible
        try:
            optim = None
            try:
                optim = get_optimizer(optimizer_name)
            except Exception:
                optim = None

            if optim is not None and ckpt_dict.get(_CheckpointDictKeys.OPTIM) is not None:
                try:
                    optim.load_state_dict(ckpt_dict[_CheckpointDictKeys.OPTIM])
                except Exception:
                    _logger.exception("Failed to load optimizer state from checkpoint")
        except Exception:
            pass

        # Update hyperparameters in ledger if requested
        try:
            hp_name = hyperparams_name
            if hp_name is None:
                hp_name = resolve_hp_name()

            if hp_name is not None:
                try:
                    # set batch size and learning rate if present
                    b = ckpt_dict.get(_CheckpointDictKeys.BSIZE, None)
                    lr = ckpt_dict.get(_CheckpointDictKeys.LRATE, None)
                    if b is not None:
                        set_hyperparam(hp_name, 'batch_size', b)
                    if lr is not None:
                        set_hyperparam(hp_name, 'learning_rate', lr)
                except Exception:
                    pass
        except Exception:
            pass
