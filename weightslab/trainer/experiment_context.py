import logging

logger = logging.getLogger(__name__)


class ExperimentContext:
    """
    Ledger-backed experiment context.

    Resolves and caches components from the GLOBAL_LEDGER:
    model, train/test dataloaders, optimizer, hyperparams, logger,
    and builds hyper_parameter descriptors used by the protocol.
    """

    def __init__(self, exp_name: str | None = None):
        # Accept an explicit experiment name or resolve from the ledger.
        self._exp_name = exp_name
        # Components resolved from GLOBAL_LEDGER (model, dataloaders, optimizer,
        # hyperparams, logger). Populated lazily by ensure_components().
        self._components = {}
        # Hyper-parameter descriptors
        self.hyper_parameters = None

    @property
    def components(self):
        self.ensure_components()
        return self._components

    @property
    def exp_name(self):
        return self._exp_name

    def ensure_components(self):
        """Ensure ledger-backed components are resolved and available on
        `self` (model, train/test dataloaders, optimizer, hyperparams,
        logger). Raises RuntimeError when mandatory components are missing.
        """
        from weightslab.backend.ledgers import (
            get_hyperparams,
            list_hyperparams,
            get_model,
            list_models,
            get_dataloader,
            list_dataloaders,
            get_optimizer,
            list_optimizers,
            get_logger,
            list_loggers,
            resolve_hp_name,
        )

        # resolve model
        model = None
        try:
            names = list_models()
            if self._exp_name and self._exp_name in names:
                model = get_model(self._exp_name)
            elif "experiment" in names:
                model = get_model("experiment")
            elif len(names) == 1:
                model = get_model()
        except Exception:
            model = None

        # resolve dataloaders (prefer explicit names 'train' / 'eval' / 'test' / 'train_loader' / 'test_loader')
        data_loaders = {}
        try:
            dnames = list_dataloaders()
            for dname in dnames:
                data_loaders[dname] = get_dataloader(dname)  # pre-load to catch errors early
        except Exception:
            logger.error("Error while listing/resolving dataloaders", exc_info=True)
            pass

        # resolve optimizer
        optimizer = None
        try:
            onames = list_optimizers()
            if len(onames) == 1:
                optimizer = get_optimizer()
            elif "_optimizer" in onames:
                optimizer = get_optimizer("_optimizer")
        except Exception:
            optimizer = None

        # resolve hyperparams (by exp_name or single set)
        hyperparams = None
        try:
            hp_names = list_hyperparams()
            if self._exp_name and self._exp_name in hp_names:
                hyperparams = get_hyperparams(self._exp_name)
            else:
                hp_name = resolve_hp_name()
                if hp_name:
                    hyperparams = get_hyperparams(hp_name)
        except Exception:
            hyperparams = None

        # resolve logger
        signal_logger = None
        try:
            lnames = list_loggers()
            if len(lnames) == 1:
                signal_logger = get_logger()
            elif "main" in lnames:
                signal_logger = get_logger("main")
        except Exception:
            signal_logger = None

        self._components = {
            "model": model,
            "optimizer": optimizer,
            "hyperparams": hyperparams,
            "signal_logger": signal_logger,
        }
        self._components.update(data_loaders)  # add all dataloaders found

        # Build hyper-parameter descriptors used by the protocol. Use
        # ledger-backed hyperparams when available, with safe fallbacks.
        def _hp_getter(key, default=None):
            def _g():
                try:
                    hp = self._components.get("hyperparams")
                    if "." in key:
                        parts = key.split(".") if key else []
                        cur = hp
                        for p in parts:
                            cur = cur[p]
                        return cur
                    if isinstance(hp, dict):
                        return hp.get(key, default)
                    elif hasattr(hp, "get"):
                        return hp.get(key, default)
                except Exception:
                    pass
                return default

            return _g

        # TODO (GP): expand hyper-parameters exposed here
        self.hyper_parameters = {
            ("Experiment Name", "experiment_name", "text", lambda: _hp_getter("experiment_name", "Anonymous")()),
            ("Left Training Steps", "training_left", "number", _hp_getter("training_steps_to_do", 999)),
            ("Eval Frequency", "eval_frequency", "number", _hp_getter("eval_full_to_train_steps_ratio", 100)),
            ("Checkpoint Frequency", "checkpooint_frequency", "number", _hp_getter("experiment_dump_to_train_steps_ratio", 100)),
            ("Learning Rate", "learning_rate", "number", _hp_getter("optimizer.lr", 1e-4)),
            ("Batch Size", "batch_size", "number", _hp_getter("data.train_loader.batch_size", 8)),
            ("Is Training", "is_training", "number", _hp_getter("is_training", 0)),
        }
