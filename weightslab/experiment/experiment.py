""" The Experiment class is the main class of the graybox package.
It is used to train and evaluate models. """

import torch as th

from tqdm import tqdm, trange
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from threading import Lock, RLock
from enum import Enum, auto

from weightslab.components.checkpoint import CheckpointManager
from weightslab.data.data_samples_with_ops import \
    DataSampleTrackingWrapper
from weightslab.components.tracking import TrackingMode
from weightslab.components.tracking import add_tracked_attrs_to_input_tensor
from weightslab.components.monitoring import \
    NeuronStatsWithDifferencesMonitor
from weightslab.backend.watcher_editor import WatcherEditor


class ArchitectureOpType(Enum):
    ADD_NEURONS = auto()
    PRUNE = auto()
    REINITIALIZE = auto()
    FREEZE = auto()


class Experiment:
    """
        Experiment class is the main class of the graybox package.
        It is used to train and evaluate models. Every change to the models, or
        the experiment parameters are made through this class
    """

    def __init__(
            self,
            model,
            input_shape,
            optimizer_class,
            train_dataset,
            eval_dataset,
            device,
            learning_rate: float,
            batch_size: int,
            criterion=None,
            metrics=None,
            task_type="classification",
            training_steps_to_do: int = 256,
            name: str = "baseline",
            root_log_dir: str = "root_experiment",
            logger=None,
            train_shuffle: bool = True,
            tqdm_display: bool = True,
            get_train_data_loader: None = None,
            get_eval_data_loader: None = None,
            skip_loading: bool = False,
            tasks: list | None = None):

        self.name = name
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.batch_size = batch_size
        self.criterion = criterion or th.nn.CrossEntropyLoss(reduction='none')
        self.metrics = metrics or {}
        self.task_type = task_type
        self.eval_dataset = eval_dataset
        self.tqdm_display = tqdm_display
        self.learning_rate = learning_rate
        self.train_dataset = train_dataset
        self.optimizer_class = optimizer_class
        self.train_shuffle = train_shuffle
        self.root_log_dir = Path(root_log_dir)
        self.get_train_data_loader = get_train_data_loader
        self.get_eval_data_loader = get_eval_data_loader
        self.last_input = None
        self.is_training = False
        self.training_steps_to_do = training_steps_to_do
        self.tasks = tasks or []
        self.lock = Lock()
        self.architecture_guard = RLock()

        self.eval_full_to_train_steps_ratio = 256
        self.experiment_dump_to_train_steps_ratio = 1024
        self.occured_train_steps = 0
        self.occured_eval__steps = 0
        self.train_loop_callbacks = []
        self.train_loop_clbk_freq = 50
        self.train_loop_clbk_call = True

        if not self.root_log_dir.exists():
            self.root_log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or SummaryWriter(root_log_dir)
        self.optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate)

        if self.train_dataset is not None:
            self.train_tracked_dataset = DataSampleTrackingWrapper(
                self.train_dataset,
                task_type=self.task_type)
            self.train_tracked_dataset._map_updates_hook_fns.append(
                self.reset_data_iterators)
            self.train_loader = th.utils.data.DataLoader(
                self.train_tracked_dataset, batch_size=self.batch_size,
                shuffle=train_shuffle)
        if self.get_train_data_loader is not None:
            self.train_loader, self.train_tracked_dataset = (
                self.get_train_data_loader()
            )
        self.train_iterator = iter(self.train_loader)
        if self.eval_dataset is not None:
            self.eval_tracked_dataset = DataSampleTrackingWrapper(
                self.eval_dataset,
                task_type=self.task_type)
            self.eval_loader = th.utils.data.DataLoader(
                self.eval_tracked_dataset, batch_size=self.batch_size)
        if self.get_eval_data_loader is not None:
            self.eval_loader, self.eval_tracked_dataset = (
                self.get_eval_data_loader())
        self.eval_iterator = iter(self.eval_loader)

        # Model WatcherEditor
        self.model = WatcherEditor(
            self.model,
            dummy_input=th.randn(self.input_shape),
            device=device
        )
        self.model.to(self.device)

        self.chkpt_manager = CheckpointManager(root_log_dir)
        self.stats_monitor = NeuronStatsWithDifferencesMonitor()

        if not skip_loading:
            self.chkpt_manager.load(
                self.chkpt_manager.get_latest_experiment(), self)

        self.model.register_hook_fn_for_architecture_change(
            lambda model: self._update_optimizer(model))

        if self.criterion.reduction != 'none':
            raise ValueError(
                "Criterion reduction must be 'none' in order to access "
                "per-sample stats")

    def __repr__(self):
        with self.lock:
            return f"Experiment[{id(self)}, {self.name}] is_train: {self.is_training} " + \
                f"steps: {self.training_steps_to_do}"

    def _update_optimizer(self, model):
        self.optimizer = self.optimizer_class(
            model.parameters(), lr=self.learning_rate)

    def register_train_loop_callback(self, callback):
        """Add callback that will be called every train_loop_clbk_freq steps
        during the training loop

        Args:
            callback (function): a function that will be called in training
        """
        self.train_loop_callbacks.append(callback)

    def unregister_train_loop_callback(self, callback):
        """Remove callback from the list of callbacks that are called during
        training.

        Args:
            callback (function): the function handle to be removed
        """
        self.train_loop_callbacks.remove(callback)

    def toggle_train_loop_callback_calls(self):
        """Toggle the calling of the callbacks during training loop
            This either enables or disables the callbacks.
        """
        self.train_loop_clbk_call = not self.train_loop_clbk_call

    def set_train_loop_clbk_freq(self, freq: int):
        """Set the frequency of the callback calls during training loop.

        Args:
            freq (int): the frequency of the callback calls
        """
        self.train_loop_clbk_freq = freq

    def performed_train_steps(self):
        """Return the number of training steps that have been performed.

        Returns:
            int: the number of training steps that have been performed
        """
        return self.occured_train_steps

    def performed_eval_steps(self):
        """Return the number of evaluation steps that have been performed.

        Returns:
            int: the number of evaluation steps that have been performed
        """
        return self.occured_eval__steps

    def display_stats(self):
        """Display the statistics of the model. This is done in the command
        line prompt. """
        self.stats_monitor.display_stats(self)

    def _pick_legacy_dense_pred(self, preds, x):
        HxW = None
        if isinstance(x, th.Tensor) and x.ndim >= 4:
            HxW = (int(x.shape[-2]), int(x.shape[-1]))

        best, best_score = None, -1.0
        for p in preds.values():
            if not isinstance(p, th.Tensor):
                continue
            if not (p.ndim >= 3 or (p.ndim == 2 and p.shape[1] >= 64)):
                continue
            if p.ndim == 3:  # [N, H, W]
                H, W = int(p.shape[-2]), int(p.shape[-1])
            elif p.ndim >= 4:
                H, W = int(p.shape[-2]), int(p.shape[-1])
            else:
                continue
            score = float(H * W)
            if HxW and (H, W) == HxW:
                score += 1e9  
            if score > best_score:
                best, best_score = p, score

        # fallback: first pred if nothing dense
        return best if best is not None else next(iter(preds.values()))

    def dump(self):
        """Dump the experiment into a checkpoint. Marks the checkpoint on the
        plots."""
        self.chkpt_manager.dump(self)
        graph_names = self.logger.get_graph_names()
        self.logger.add_annotations(
            graph_names, self.name, "checkpoint", self.model.get_age(),
            {
                "checkpoint_id": self.chkpt_manager.get_latest_experiment()
            }
        )

    def load(self, checkpoint_id: int):
        """Loads the given checkpoint with a given id.

        Args:
            checkpoint_id (int): the checkpoint id to be loaded
        """
        self.optimizer.zero_grad()
        self.chkpt_manager.load(checkpoint_id, self)

    def print_checkpoints_tree(self):
        """Display the checkpoints tree."""
        print(self.chkpt_manager.id_to_path)

    def reset_data_iterators(self):
        """Reset the data iterators. This is necessary when anything related to
        datasets or dataloaders changes."""
        if self.get_train_data_loader is None:
            self.train_loader = th.utils.data.DataLoader(
                self.train_tracked_dataset,
                batch_size=self.batch_size, shuffle=self.train_shuffle)
            self.train_iterator = iter(self.train_loader)
            self.eval_loader = th.utils.data.DataLoader(
                self.eval_tracked_dataset, batch_size=self.batch_size)
            self.eval_iterator = iter(self.eval_loader)
        else:
            self.train_loader, self.train_tracked_dataset = (
                self.get_train_data_loader(self.batch_size)
            )
            self.train_iterator = iter(self.train_loader)
            self.eval_loader, self.eval_tracked_dataset = (
                self.get_eval_data_loader(self.batch_size)
            )
            self.eval_iterator = iter(self.eval_loader)

    def set_learning_rate(self, learning_rate: float):
        """Set the learning rate of the optimizer.
        Args:
            learning_rate (float): the new learning rate
        """
        with self.lock:
            self.learning_rate = learning_rate
            self.optimizer = self.optimizer_class(
                self.model.parameters(), lr=self.learning_rate)

    def set_batch_size(self, batch_size: int):
        """Set the batch size of the optimizer.
        Args:
            batch_size (int): the new batch size
        """
        with self.lock:
            self.batch_size = batch_size
            self.reset_data_iterators()

    def _pass_one_batch(self, loader_iterator):
        # From the dataset we get: item, index, target
        try:
            input_in_id_label = next(loader_iterator)
        except Exception as e:
            # print("Exception in _pass_one_batch: ", e, self.occured_train_steps)
            # import pdb; pdb.set_trace()

            raise StopIteration

        input_in_id_label = [
            tensor.to(self.device) for tensor in input_in_id_label]
        data, in_id, label = input_in_id_label
        add_tracked_attrs_to_input_tensor(
            data, in_id_batch=in_id, label_batch=label)
        self.last_input = data
        return data, self.model(data)

    def train_one_step(self):
        """Train the model for one step."""
        with self.lock:
            if self.is_training is False:
                return
            self.occured_train_steps += 1

        with self.architecture_guard:
            self.model.train()
            self.model.set_tracking_mode(TrackingMode.TRAIN)
            self.optimizer.zero_grad()
            model_age = self.model.get_age()
            try:
                input, output = self._pass_one_batch(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                input, output = self._pass_one_batch(self.train_iterator)

            # if multitask
            if getattr(self, "tasks", None):
                head_outputs = {t.name: t.forward(input) for t in self.tasks}
                per_task_losses = {}
                per_task_preds = {}
                combined_losses_batch = None
                loss = 0.0

                for t in self.tasks:
                    # each task decides where its targets come from
                    targets = t.get_targets(input) if hasattr(t, "get_targets") else input.label_batch
                    losses_batch = t.compute_loss(head_outputs[t.name], targets)
                    if losses_batch.ndim == 0:  # keep old safety
                        losses_batch = losses_batch.unsqueeze(0)
                    per_task_losses[t.name] = losses_batch.detach()
                    per_task_preds[t.name] = t.infer_pred(head_outputs[t.name]).detach()
                    weighted = losses_batch * float(getattr(t, "loss_weight", 1.0))
                    combined_losses_batch = weighted if combined_losses_batch is None else (combined_losses_batch + weighted)
                    loss = loss + weighted.mean()

                try:
                    loss.backward()
                except Exception as e:
                    self.chkpt_manager.dump(self, f"crash_{self.name}_bs{self.batch_size}_step{self.performed_train_steps()}")
                    print(
                        'Loss backward error, losses_batch shape:',
                        list(combined_losses_batch.shape) if combined_losses_batch is not None else None,
                        'current batch_size:', self.batch_size,
                        'error:', str(e)
                    )
                self.optimizer.step()

                with th.no_grad():
                    for t in self.tasks:
                        if not getattr(t, "metrics", None):
                            continue
                        targets = t.get_targets(input)
                        outs = head_outputs[t.name]
                        for mname, metric in t.metrics.items():
                            m = metric.to(self.device) if hasattr(metric, "to") else metric
                            val = m(outs, targets)
                            if hasattr(m, "compute"):
                                val = m.compute().item()
                                if hasattr(m, "reset"):
                                    m.reset()
                            elif hasattr(val, "item"):
                                val = val.item()
                            self.logger.add_scalars(f"train-{t.name}-{mname}", {self.name: val}, global_step=model_age)

                    for t in self.tasks:
                        lb = per_task_losses[t.name]
                        self.logger.add_scalars(f"train-loss/{t.name}", {self.name: lb.mean().item()}, global_step=model_age)

            else:
                if self.task_type == "segmentation":
                    losses_batch = self.criterion(
                        output,
                        input.label_batch.long()
                    )
                    # Output: (N, C, H, W), argmax over channel dim
                    pred = output.argmax(dim=1)
                else:
                    losses_batch = self.criterion(
                        output.flatten(),
                        input.label_batch.flatten().float()
                    )
                    if output.ndim == 1:
                        pred = (output > 0.0).long()
                    else:
                        pred = output.argmax(dim=1, keepdim=True)

                if losses_batch.ndim == 0:
                    losses_batch = losses_batch.unsqueeze(0)
                loss = th.mean(losses_batch)
                try:
                    loss.backward()
                except Exception as e:
                    self.chkpt_manager.dump(self, f"crash_{self.name}_bs{self.batch_size}_step{self.performed_train_steps()}")
                    print(
                        'Loss backward error, losses_batch shape:',
                        losses_batch.shape,
                        'current batch_size:', self.batch_size,
                        'error:', str(e)
                    )
                self.optimizer.step()

        with self.lock:
            if getattr(self, "tasks", None):
                # combined into the primary per-sample channel (back-compat)
                ids_np = input.in_id_batch.detach().cpu().numpy()
                comb_np = combined_losses_batch.detach().cpu().numpy()

                # first_pred = next(iter(per_task_preds.values()))
                first_pred = self._pick_legacy_dense_pred(per_task_preds, input)

                self.train_loader.dataset.update_batch_sample_stats(
                    model_age,
                    ids_np,
                    comb_np,
                    first_pred.detach().cpu().numpy()
                )

                # extended per-task stats (non-breaking)
                stats_map = {}
                for name, lb in per_task_losses.items():
                    stats_map[f"loss/{name}"] = lb.detach().cpu().numpy()
                for name, pr in per_task_preds.items():
                    stats_map[f"pred/{name}"] = pr.detach().cpu().numpy()
                try:
                    self.train_loader.dataset.update_sample_stats_ex_batch(ids_np, stats_map)
                except Exception:
                    pass
            else:
                self.train_loader.dataset.update_batch_sample_stats(
                    model_age,
                    input.in_id_batch.detach().cpu().numpy(),
                    losses_batch.detach().cpu().numpy(),
                    pred.detach().cpu().numpy())

                try:
                    ids_np = input.in_id_batch.detach().cpu().numpy()
                    per_sample_loss_np = losses_batch.detach().cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    self.train_loader.dataset.update_sample_stats_ex_batch(
                        ids_np,
                        {
                            "loss/combined": per_sample_loss_np,
                            "pred": pred_np  # dense (e.g., seg masks) will be handled/downsampled in the dataset
                        }
                    )
                except Exception:
                    pass

        self.logger.add_scalars(
            'train-loss', {self.name: loss.detach().cpu().numpy()},
            global_step=model_age)

        # original per-epoch metrics remain for single-task (no change)
        if not getattr(self, "tasks", None):
            for name, metric in self.metrics.items():
                # torchmetrics.Metric
                if hasattr(metric, 'to'):
                    metric = metric.to(self.device)
                    metric_value = metric(
                        output.flatten(),
                        input.label_batch.flatten()
                    )
                    if hasattr(metric, 'compute'):
                        metric_value = metric.compute().item()
                        metric.reset()
                else:
                    # custom metric
                    metric_value = metric(output, input.label_batch)
                    if hasattr(metric_value, 'item'):
                        metric_value = metric_value.item()

                self.logger.add_scalars(
                    f'train-{name}',
                    {self.name: metric_value},
                    global_step=model_age
                )

        with self.lock:
            self.training_steps_to_do -= 1
            self.is_training = self.training_steps_to_do > 0

    def train_n_steps(self, n: int):
        """Train the model for n steps.

        Args:
            n (int): The number of steps to be performed.
        """
        train_range = trange(n, desc='Training..', total=len(n))
        try:
            for _ in train_range:
                self.train_one_step()
        except KeyboardInterrupt:
            pass

    def report_parameters_count(self):
        """Report the number of parameters of the model to the tensorboard."""
        self.logger.add_scalars(
            'model-params',
            {
                self.name: self.model.get_parameter_count()
            },
            global_step=self.model.get_age())

    @th.no_grad()
    def eval_one_step(self):
        """Evaluate the model for one step."""
        with self.architecture_guard:
            self.occured_eval__steps += 1
            self.model.eval()
            self.model.set_tracking_mode(TrackingMode.EVAL)
            try:
                input, output = self._pass_one_batch(self.eval_iterator)
            except StopIteration:
                self.eval_iterator = iter(self.eval_loader)
                input, output = self._pass_one_batch(self.eval_iterator)

        if getattr(self, "tasks", None):
            head_outputs = {t.name: t.forward(input) for t in self.tasks}
            per_task_losses = {}
            per_task_preds = {}
            combined_losses_batch = None

            for t in self.tasks:
                targets = t.get_targets(input) if hasattr(t, "get_targets") else input.label_batch
                losses_batch = t.compute_loss(head_outputs[t.name], targets)
                if losses_batch.ndim == 0:
                    losses_batch = losses_batch.unsqueeze(0)
                per_task_losses[t.name] = losses_batch.detach()
                per_task_preds[t.name] = t.infer_pred(head_outputs[t.name]).detach()
                weighted = losses_batch * float(getattr(t, "loss_weight", 1.0))
                combined_losses_batch = weighted if combined_losses_batch is None else (combined_losses_batch + weighted)

            test_loss = combined_losses_batch.mean() if combined_losses_batch.ndim > 0 else combined_losses_batch

            model_age = self.model.get_age()
            first_pred = self._pick_legacy_dense_pred(per_task_preds, input)
            self.eval_loader.dataset.update_batch_sample_stats(
                model_age,
                input.in_id_batch.detach().cpu().numpy(),
                combined_losses_batch.detach().cpu().numpy(),
                first_pred.detach().cpu().numpy()
            )


            try:
                ids_np = input.in_id_batch.detach().cpu().numpy()
                stats_map = {}
                for name, lb in per_task_losses.items():
                    stats_map[f"loss/{name}_eval"] = lb.detach().cpu().numpy()
                for name, pr in per_task_preds.items():
                    stats_map[f"pred/{name}_eval"] = pr.detach().cpu().numpy()
                self.eval_loader.dataset.update_sample_stats_ex_batch(ids_np, stats_map)
            except Exception:
                pass

            # preserve original return contract (loss tensor, metrics dict)
            metric_results = {}
            with th.no_grad():
                for t in self.tasks:
                    if not getattr(t, "metrics", None):
                        continue
                    targets = t.get_targets(input)
                    outs = head_outputs[t.name]
                    for mname, metric in t.metrics.items():
                        m = metric.to(self.device) if hasattr(metric, "to") else metric
                        val = m(outs, targets)
                        if hasattr(m, "compute"):
                            val = m.compute().item()
                            if hasattr(m, "reset"):
                                m.reset()
                        elif hasattr(val, "item"):
                            val = val.item()
                        metric_results[f"{t.name}/{mname}"] = val

                # also expose per-task mean loss (unweighted) for eval
                for t in self.tasks:
                    lb = per_task_losses[t.name]
                    self.logger.add_scalars(f"eval-loss/{t.name}", {self.name: lb.mean().item()}, global_step=model_age)
            return test_loss, metric_results

        if self.task_type == "segmentation":
            # For segmentation: output (N, C, H, W), label (N, H, W)
            losses_batch = self.criterion(output, input.label_batch.long())
            pred = output.argmax(dim=1)
        else:
            losses_batch = self.criterion(output, input.label_batch)
            if output.ndim == 1:
                pred = (output > 0.0).long()
            else:
                pred = output.argmax(dim=1, keepdim=True)

        if losses_batch.ndim == 0:
            losses_batch = losses_batch.unsqueeze(0)

        test_loss = th.sum(losses_batch) if losses_batch.ndim > 0 else losses_batch

        model_age = self.model.get_age()
        self.eval_loader.dataset.update_batch_sample_stats(
            model_age,
            input.in_id_batch.detach().cpu().numpy(),
            losses_batch.detach().cpu().numpy(),
            pred.detach().cpu().numpy()
        )

        try:
            ids_np = input.in_id_batch.detach().cpu().numpy()
            per_sample_loss_np = losses_batch.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            self.eval_loader.dataset.update_sample_stats_ex_batch(
                ids_np,
                {
                    "loss/combined_eval": per_sample_loss_np,
                    "pred_eval": pred_np
                }
            )
        except Exception:
            pass

        metric_results = {}
        for name, metric in self.metrics.items():
            if hasattr(metric, 'to'):
                metric = metric.to(self.device)
                metric_value = metric(output, input.label_batch)
                if hasattr(metric, 'compute'):
                    metric_value = metric.compute().item()
                    metric.reset()
            else:
                metric_value = metric(output, input.label_batch)
                if hasattr(metric_value, 'item'):
                    metric_value = metric_value.item()
            metric_results[name] = metric_value

        return test_loss, metric_results

    @th.no_grad()
    def eval_n_steps(self, n: int):
        losses = 0.0
        metric_totals = {name: 0.0 for name in self.metrics}
        count = 0
        eval_range = range(n)
        try:
            for _ in eval_range:
                step_loss, metric_results = self.eval_one_step()
                losses += step_loss
                for name, value in metric_results.items():
                    metric_totals[name] += value
                count += 1
        except KeyboardInterrupt:
            pass

        if count == 0:
            mean_loss = 0.0
            mean_metrics = {name: 0.0 for name in self.metrics}
        else:
            mean_loss = losses / count
            mean_metrics = {name: metric_totals[name] / count for name in self.metrics}
        return mean_loss.cpu(), mean_metrics

    @th.no_grad()
    def eval_full(self, skip_tensorboard: bool = False):
        """Evaluate the model on the full dataset."""

        mean_loss, mean_metrics = self.eval_n_steps(len(self.eval_loader))

        print("eval full: ", mean_loss, mean_metrics)

        if not skip_tensorboard:
            self.logger.add_scalars(
                'eval-loss', {self.name: mean_loss},
                global_step=self.model.get_age())
            for name, value in mean_metrics.items():
                self.logger.add_scalars(
                    f'eval-{name}', {self.name: value},
                    global_step=self.model.get_age())
            self.report_parameters_count()
        return mean_loss, mean_metrics

    def train_step_or_eval_full(self):
        """Train the model for one step or evaluate the model on the full."""
        step = self.performed_train_steps()

        # Skip eval/dump on first step
        if step > 0 and step % self.eval_full_to_train_steps_ratio == 0:
            self.eval_full()

        if step > 0 and step % self.experiment_dump_to_train_steps_ratio == 0:
            self.dump()

        if self.train_loop_clbk_call and step % self.train_loop_clbk_freq == 0:
            for callback_fn in self.train_loop_callbacks:
                callback_fn()
        self.train_one_step()

    def train_n_steps_with_eval_full(self, n: int):
        """Train the model for n steps and evaluate the model on the full
        dataset.

        Args:
            n (int): the number of training steps to be performed
        """
        train_range = tqdm(range(n)) if self.tqdm_display else range(n)
        try:
            for _ in train_range:
                self.train_step_or_eval_full()
        except KeyboardInterrupt:
            pass

    def toggle_training_status(self):
        """Toggle the training status. If the model is training, it will stop.
        """
        with self.lock:
            self.is_training = not self.is_training

    def set_training_steps_to_do(self, steps: int):
        """Set the number of training steps to be performed.
        Args:
            steps (int): the number of training steps to be performed
        """
        with self.lock:
            self.training_steps_to_do = steps

    def get_is_training(self) -> bool:
        """Returns whether the model is training."""
        with self.lock:
            return self.is_training

    def set_is_training(self, is_training: bool):
        """Set whether the model is training."""
        with self.lock:
            self.is_training = is_training
        print("[exp].set_is_training ", is_training)

    def get_training_steps_to_do(self) -> int:
        """"Get the number of training steps to be performed."""
        with self.lock:
            return self.training_steps_to_do

    def get_train_records(self):
        """"Get all the train samples are records."""
        with self.lock:
            return self.train_loader.dataset.as_records()

    def get_eval_records(self):
        """"Get all the train samples are records."""
        with self.lock:
            return self.eval_loader.dataset.as_records()

    def set_name(self, name: str):
        with self.lock:
            self.name = name

    def apply_architecture_op(self, op_type, **kwargs):
        if op_type == ArchitectureOpType.ADD_NEURONS:
            with self.architecture_guard, self.model as model:
                model.add_neurons(
                    layer_id=kwargs['layer_id'],
                    neuron_count=kwargs['neuron_count'],
                    skip_initialization=kwargs.get(
                        'skip_initialization',
                        False
                    )
                )
        elif op_type == ArchitectureOpType.PRUNE:
            with self.architecture_guard:
                self.model.prune(
                    layer_id=kwargs['layer_id'],
                    neuron_indices=kwargs['neuron_indices']
                )
        elif op_type == ArchitectureOpType.REINITIALIZE:
            with self.architecture_guard:
                self.model.reinit_neurons(
                    layer_id=kwargs['layer_id'],
                    neuron_indices=kwargs['neuron_indices']
                )
        elif op_type == ArchitectureOpType.FREEZE:
            with self.architecture_guard:
                self.model.freeze(
                    layer_id=kwargs['layer_id'],
                    neuron_ids=kwargs['neuron_ids']
                )
        else:
            raise ValueError(f"Unsupported op_type: {op_type}")
