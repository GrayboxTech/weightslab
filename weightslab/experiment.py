""" The Experiment class is the main class of the graybox package.
It is used to train and evaluate models. """

import torch as th
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from weightslab.checkpoint import CheckpointManager
from weightslab.data_samples_with_ops import DataSampleTrackingWrapper
from weightslab.tracking import TrackingMode
from weightslab.tracking import add_tracked_attrs_to_input_tensor
from weightslab.monitoring import NeuronStatsWithDifferencesMonitor


from threading import Lock


class Experiment:
    """
        Experiment class is the main class of the graybox package.
        It is used to train and evaluate models. Every change to the models, or
        the experiment parameters are made through this class
    """

    def __init__(
            self,
            model,
            optimizer_class,
            train_dataset,
            eval_dataset,
            device,
            learning_rate: float,
            batch_size: int,
            training_steps_to_do: int = 256,
            name: str = "baseline",
            root_log_dir: str = "root_experiment",
            logger=None,
            train_shuffle: bool = True,
            tqdm_display: bool = True,
            get_train_data_loader: None = None,
            get_eval_data_loader: None = None,
            skip_loading: bool = False):

        self.name = name
        self.model = model
        self.device = device
        self.batch_size = batch_size
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

        if not self.root_log_dir.exists():
            self.root_log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or SummaryWriter(root_log_dir)
        self.optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate)

        if self.train_dataset is not None:
            self.train_tracked_dataset = DataSampleTrackingWrapper(
                self.train_dataset)
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
                self.eval_dataset)
            self.eval_loader = th.utils.data.DataLoader(
                self.eval_tracked_dataset, batch_size=self.batch_size)
        if self.get_eval_data_loader is not None:
            self.eval_loader, self.eval_tracked_dataset = (
                self.get_eval_data_loader())
        self.eval_iterator = iter(self.eval_loader)

        self.eval_full_to_train_steps_ratio = 256
        self.experiment_dump_to_train_steps_ratio = 1024
        self.occured_train_steps = 0
        self.occured_eval__steps = 0

        self.model.to(self.device)
        self.chkpt_manager = CheckpointManager(root_log_dir)
        self.stats_monitor = NeuronStatsWithDifferencesMonitor()

        if not skip_loading:
            self.chkpt_manager.load(
                self.chkpt_manager.get_latest_experiment(), self)

        self.train_loop_callbacks = []
        self.train_loop_clbk_freq = 50
        self.train_loop_clbk_call = True

        self.model.register_hook_fn_for_architecture_change(
            lambda model: self._update_optimizer(model))

        self.lock = Lock()
        self.model.to(self.device)

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

        self.model.train()
        self.model.set_tracking_mode(TrackingMode.TRAIN)
        self.optimizer.zero_grad()
        model_age = self.model.get_age()
        try:
            input, output = self._pass_one_batch(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            input, output = self._pass_one_batch(self.train_iterator)

        losses_batch = F.cross_entropy(
            output, input.label_batch, reduction='none')

        loss = th.mean(losses_batch)
        loss.backward()
        pred = output.argmax(dim=1, keepdim=True)

        with self.lock:
            self.train_loader.dataset.update_batch_sample_stats(
                model_age,
                input.in_id_batch.detach().cpu().numpy(),
                losses_batch.detach().cpu().numpy(),
                pred.detach().cpu().numpy())

        correct = pred.eq(input.label_batch.view_as(pred)).sum().item()
        accuracy = 100. * correct / pred.shape[0]  # batch_size

        self.logger.add_scalars(
            'train-loss', {self.name: loss.detach().cpu().numpy()},
            global_step=model_age)
        self.logger.add_scalars(
            'train-acc', {self.name: accuracy}, global_step=model_age)
        self.optimizer.step()

        with self.lock:
            self.training_steps_to_do -= 1
            self.is_training = self.training_steps_to_do > 0

    def train_n_steps(self, n: int):
        """Train the model for n steps.

        Args:
            n (int): The number of steps to be performed.
        """
        train_range = range(n)
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
        self.occured_eval__steps += 1
        self.model.eval()
        self.model.set_tracking_mode(TrackingMode.EVAL)
        try:
            input, output = self._pass_one_batch(self.eval_iterator)
        except StopIteration:
            self.eval_iterator = iter(self.eval_loader)
            input, output = self._pass_one_batch(self.eval_iterator)

        losses_batch = F.cross_entropy(
            output, input.label_batch, reduction='none')
        test_loss = th.sum(losses_batch)
        pred = output.argmax(dim=1, keepdim=True)

        model_age = self.model.get_age()
        self.eval_loader.dataset.update_batch_sample_stats(
                model_age,
                input.in_id_batch.detach().cpu().numpy(),
                losses_batch.detach().cpu().numpy(),
                pred.detach().cpu().numpy())

        correct = pred.eq(input.label_batch.view_as(pred)).cpu().sum().item()
        return test_loss, correct

    @th.no_grad()
    def eval_n_steps(self, n: int):
        losses, correct = 0, 0
        eval_range = range(n)
        try:
            for _ in eval_range:
                step_loss, step_corrects = self.eval_one_step()
                losses += step_loss
                correct += step_corrects
        except KeyboardInterrupt:
            pass
        return losses.cpu(), correct

    @th.no_grad()
    def eval_full(self, skip_tensorboard: bool = False):
        """Evaluate the model on the full dataset."""

        losses, correct = self.eval_n_steps(len(self.eval_loader))

        losses /= len(self.eval_loader.dataset)
        accuracy = 100. * correct / len(self.eval_loader.dataset)

        print("eval full: ", losses, accuracy)

        if not skip_tensorboard:
            self.logger.add_scalars(
                'eval-loss', {self.name: losses},
                global_step=self.model.get_age())
            self.logger.add_scalars(
                'eval-acc', {self.name: accuracy},
                global_step=self.model.get_age())
            self.report_parameters_count()
        return losses, accuracy

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
