import enum
import os
import json
import uuid
import logging
import torch as th

from pathlib import Path
from typing import Set, Optional

from weightslab.backend.ledgers import (
    get_model,
    get_optimizer,
    get_dataloader,
    list_hyperparams,
    get_hyperparams,
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
    OPTIM = 'optimizer'
    LRATE = 'learning_rate'
    BSIZE = 'batch_size'
    TDATA = 'train_dataset'
    EDATA = 'eval_dataset'
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
             train_loader_name: Optional[str] = None,
             eval_loader_name: Optional[str] = None,
             experiment_name: Optional[str] = None,
             override_filepath: Optional[str] = None,
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
        train_data_state = None
        eval_data_state = None
        batch_size = None
        learning_rate = None

        # Model
        try:
            model = get_model(model_name)
            try:
                model_state = model.state_dict()
            except Exception:
                # if model is a Proxy or missing, skip
                model_state = None
        except Exception:
            model_state = None

        # Optimizer
        try:
            optim = get_optimizer(optimizer_name)
            try:
                optimizer_state = optim.state_dict()
                # attempt to extract lr from param groups
                try:
                    lrs = []
                    for g in optim.param_groups:
                        lrs.append(g.get('lr'))
                    if lrs:
                        learning_rate = lrs[0]
                except Exception:
                    learning_rate = None
            except Exception:
                optimizer_state = None
        except Exception:
            optimizer_state = None

        # Train / Eval dataloaders
        try:
            tr = get_dataloader(train_loader_name)
            try:
                train_data_state = tr.dataset.state_dict()
                try:
                    batch_size = getattr(tr, 'batch_size', None)
                except Exception:
                    batch_size = None
            except Exception:
                train_data_state = None
        except Exception:
            train_data_state = None

        try:
            ev = get_dataloader(eval_loader_name)
            try:
                eval_data_state = ev.dataset.state_dict()
            except Exception:
                eval_data_state = None
        except Exception:
            eval_data_state = None

        payload = {
            _CheckpointDictKeys.ENAME: experiment_name,
            _CheckpointDictKeys.BSIZE: batch_size,
            _CheckpointDictKeys.LRATE: learning_rate,
            _CheckpointDictKeys.MODEL: model_state,
            _CheckpointDictKeys.OPTIM: optimizer_state,
            _CheckpointDictKeys.EDATA: eval_data_state,
            _CheckpointDictKeys.TDATA: train_data_state,
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
             train_loader_name: Optional[str] = None,
             eval_loader_name: Optional[str] = None,
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
            ckpt_dict = th.load(path)
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
                        model.load_state_dict(ckpt_dict[_CheckpointDictKeys.MODEL])
                    except Exception:
                        _logger.exception("Failed to load model state from checkpoint")
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

        # Restore dataset states
        try:
            tr = None
            try:
                tr = get_dataloader(train_loader_name)
            except Exception:
                tr = None
            if tr is not None and ckpt_dict.get(_CheckpointDictKeys.TDATA) is not None:
                try:
                    tr.dataset.load_state_dict(ckpt_dict[_CheckpointDictKeys.TDATA])
                except Exception:
                    _logger.exception("Failed to load train dataset state from checkpoint")
        except Exception:
            pass

        try:
            ev = None
            try:
                ev = get_dataloader(eval_loader_name)
            except Exception:
                ev = None
            if ev is not None and ckpt_dict.get(_CheckpointDictKeys.EDATA) is not None:
                try:
                    ev.dataset.load_state_dict(ckpt_dict[_CheckpointDictKeys.EDATA])
                except Exception:
                    _logger.exception("Failed to load eval dataset state from checkpoint")
        except Exception:
            pass

        # Update hyperparameters in ledger if requested
        try:
            hp_name = hyperparams_name
            if hp_name is None:
                names = list_hyperparams()
                if 'main' in names:
                    hp_name = 'main'
                elif 'experiment' in names:
                    hp_name = 'experiment'
                elif len(names) == 1:
                    hp_name = names[0]

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
