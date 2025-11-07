""" Module containing tracking related interfaces and classes. """

import enum
from collections import defaultdict
from typing import Set, List
import torch as th

from weightslab.modules.neuron_ops import NeuronWiseOperations


_TRACKING_TENSOR_DISPLAY_LIMIT = 8


class TrackingMode(str, enum.Enum):
    """ Tracking mode w.r.t the dataset. """

    EVAL = 'eval'
    TRAIN = 'train'
    DISABLED = 'disabled'


def add_tracked_attrs_to_input_tensor(
        indata: th.Tensor,
        in_id_batch: th.Tensor | None,
        label_batch: th.Tensor | None):

    """
    Helper function that attaches to the input tensors useful infos. This
    is a hacky way to enable statistics computation during training.

    Args:
    -----------
    indata: th.Tensor
        The input tensor that will be augmented.
    in_id_batch: th.Tensor
        A tensor containing a batch of sample ids that will be attached to
        indata.
    label_batch: th.Tensor
        A tensor containing the labels ids of the samples in the batch.
    """
    if type(indata) is th.Tensor:
        if indata is not None:
            setattr(indata, 'batch_size', indata.shape[0])

        if indata.batch_size != label_batch.shape[0]:
            raise ValueError(
                f"Augmented input ensor: batch size {indata.batch_size} "
                f"differs from label batch size {label_batch.shape[0]}.")
        setattr(indata, "label_batch", label_batch)

        if indata.batch_size != in_id_batch.shape[0]:
            raise ValueError(
                f"Augmented input ensor: batch size {indata.batch_size} "
                f"differs from input id size {in_id_batch.shape[0]}.")
        setattr(indata, "in_id_batch", in_id_batch)
    return indata

def copy_forward_tracked_attrs(
        indata: th.Tensor, indata_w_attrs: th.Tensor):
    """
    Helper function that attaches to the indata tensors useful infos such as
    label batch and sample ids batch from the indata_w_attrs.

    Args:
    -----------
    indata: th.Tensor
        The input tensor that will be augmented.
    indata_w_attrs: th.Tensor
        A tensor containing the tracked attributes to be copied over.
    """
    if hasattr(indata, "batch_size"):
        setattr(indata, 'batch_size', indata_w_attrs.batch_size)

    if hasattr(indata_w_attrs, 'in_id_batch'):
        setattr(indata, 'in_id_batch', indata_w_attrs.in_id_batch)

    if hasattr(indata_w_attrs, 'label_batch'):
        setattr(indata, 'label_batch', indata_w_attrs.label_batch)


class Tracker(NeuronWiseOperations, th.nn.Module):
    """ Tracker interface for neuron level statistics. """
    def update(self, tensor: th.Tensor):
        """
        Update the tracked statistics after each batch.

        Args:
            tensor (th.Tensor):
                the input tensor that has been passed through the layer.
        """
        raise NotImplementedError

    def forward(self, tensor):
        """
        Forward pass of the tracker.

        Args:
            tensor (th.Tensor):
                the input tensor that has been passed through the layer.
        """
        return self.update(tensor)


class TriggersTracker(Tracker):
    """
    Computes how often does neurons trigger on average during a given
    time frame. Keeps track of how many times a neuron triggered (a.k.a 
    its value was over 0) and how many samples tha neuron has seen.

    Args:
        number_of_neurons (int): The number of neurons in the tracker.
        device (torch.device, optional): The device on which to perform
            computations.
            Defaults to None.
    """
    def __init__(self, number_of_neurons: int, device: th.device = None):
        super().__init__()
        self.device = device
        self.number_of_neurons = number_of_neurons
        self.triggrs_by_neuron = th.zeros(number_of_neurons).long().to(self.device)
        self.updates_by_neuron = th.zeros(number_of_neurons).long().to(self.device)

    def reset_stats(self):
        """
        Reset statistics for all neurons. Statistics are just an
        approximation, so they may shift in time, hence after a significant
        amount of time they should not be representative anymore.
        """
        self.triggrs_by_neuron = th.zeros(self.number_of_neurons).long().to(
            self.device)
        self.updates_by_neuron = th.zeros(self.number_of_neurons).long().to(
            self.device)

    def __hash__(self):
        triggers_tuple = tuple(self.triggrs_by_neuron.tolist())
        updates_tuple = tuple(self.updates_by_neuron.tolist())

        # Convert device to string or another immutable representation
        return hash(
            (
                str(self.device),
                self.number_of_neurons,
                triggers_tuple,
                updates_tuple
            )
        )

    def __repr__(self) -> str:
        return "TriggersTracker[#%d]: [%s] & [%s]" % (\
            self.number_of_neurons,
            str(self.triggrs_by_neuron[:_TRACKING_TENSOR_DISPLAY_LIMIT]),
            str(self.updates_by_neuron[:_TRACKING_TENSOR_DISPLAY_LIMIT])
        )

    def __eq__(self, other: "TriggersTracker") -> bool:
        are_equals = self.number_of_neurons == other.number_of_neurons and \
            th.equal(self.triggrs_by_neuron, other.triggrs_by_neuron) and \
            th.equal(self.updates_by_neuron, other.updates_by_neuron)
        return are_equals

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        state[prefix + 'number_of_neurons'] = self.number_of_neurons
        state[prefix + 'triggrs_by_neuron'] = self.triggrs_by_neuron
        state[prefix + 'updates_by_neuron'] = self.updates_by_neuron
        return state

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        if state_dict:
            self.number_of_neurons = \
                state_dict[prefix + 'number_of_neurons']
            self.triggrs_by_neuron = \
                state_dict[prefix + 'triggrs_by_neuron'].to(self.device)
            self.updates_by_neuron = \
                state_dict[prefix + 'updates_by_neuron'].to(self.device)
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs)

    def to(
            self,
            device: th.device = None,
            dtype: th.dtype = None,
            non_blocking: bool = False):
        """ Moves the component fields and states to the given device.

        Args:
            device (th.device, optional): Defaults to None.
            dtype (th.dtype, optional): Defaults to None.
            non_blocking (bool, optional): Defaults to False.
        """
        self.device = device
        if self.device is not None:
            self.triggrs_by_neuron = self.triggrs_by_neuron.to(
                device, dtype=dtype, non_blocking=non_blocking)
            self.updates_by_neuron = self.updates_by_neuron.to(
                device, dtype=dtype, non_blocking=non_blocking)

    def update(self, tensor: th.Tensor):
        """
        Each batch of data is passed statistics are computed.

        Args:
            tensor (th.Tensor): the tensor that has been passed through the
                layer.
        """

        # Assumes that triggers per neuron have been pre-processed already.
        # Shape is expected to be in the form [batch_size x neuron_count]
        if len(tensor.shape) > 2:
            # raise ValueError(
            #     f"Neuron stats are updated on a per neuron level, hence only "
            #     f"two dims are expected [batch_size x neuron_count] but "
            #     f"activation map has shape: {str(tensor.shape)}")
            tensor = tensor.view(-1, self.number_of_neurons)
        try:
            if tensor.shape == th.Size([]):
                tensor = tensor[None, None]  # Add one dim
            bs = tensor.shape[0]
            self.triggrs_by_neuron += th.sum(
                tensor, dim=(0, )).view(-1).long().to(self.device)
            self.updates_by_neuron += th.ones(
                self.number_of_neurons).long().to(self.device) * bs
        except RuntimeError as err:
            raise err

    def prune(self, indices: Set[int], update_neuron_count: bool = True):
        neurons = set(range(self.number_of_neurons))
        indices = set(indices) if not isinstance(indices, set) else indices
        if not indices & neurons:
            raise ValueError(
                f"TriggersTracker.prune indices and neurons set do not "
                f"overlapp: {indices} & {neurons} => {indices & neurons}")

        kept_neurons = neurons - indices

        triggrs_by_neuron = th.zeros(len(kept_neurons))
        updates_by_neuron = th.zeros(len(kept_neurons))
        for filter_dest, filter_from in enumerate(kept_neurons):
            triggrs_by_neuron[filter_dest] = self.triggrs_by_neuron[
                filter_from]
            updates_by_neuron[filter_dest] = self.updates_by_neuron[
                filter_from]
        self.triggrs_by_neuron = triggrs_by_neuron
        self.updates_by_neuron = updates_by_neuron

        if update_neuron_count:
            self.number_of_neurons = len(kept_neurons)

        self.to(self.device)

    def reset(self, indices: Set[int]):
        for neuron_id in indices:
            try:
                self.triggrs_by_neuron[neuron_id] = 0
                self.updates_by_neuron[neuron_id] = 0
            except IndexError as e:
                raise ValueError(
                    f"TriggersTracker.reset: Can not reset neuron {neuron_id}"
                    f", it does not exist.") from e

    def add_neurons(self, neuron_count: int):
        if neuron_count <= 0:
            raise ValueError(
                f"TriggersTracker.add_neurons: "
                f"cannot add {neuron_count} neurons.")

        zeros = th.zeros(neuron_count).to(self.device)
        self.number_of_neurons += neuron_count
        self.triggrs_by_neuron = th.cat((self.triggrs_by_neuron, zeros)).long()
        self.updates_by_neuron = th.cat((self.updates_by_neuron, zeros)).long()

        self.to(self.device)

    def get_neuron_triggers(self, neuron_id: int):
        """
            Get number of times the value of the neuron with neuron_id was
            above 0.
        """
        return self.triggrs_by_neuron[neuron_id].item()

    def get_neuron_age(self, neuron_id: int):
        """ Get number of updates of the neuron with neuron_id."""
        return self.updates_by_neuron[neuron_id].item()

    def get_neuron_stats(self, neuron_id: int):
        """ Get how often did this neuron trigger on average. """
        return self.get_neuron_triggers(neuron_id) / \
            max(self.get_neuron_age(neuron_id), 1)

    def get_neuron_number(self) -> int:
        """ Get the number of neurons in the layer. """
        return self.number_of_neurons

    def get_neuron_pretty_repr(self, neuron_id: int, prefix: str = '') -> str:
        """ Get a pretty representation of the neuron statistics. """
        prefix = prefix[:5]
        frq = self.get_neuron_stats(neuron_id)
        cnt = self.get_neuron_triggers(neuron_id)
        age = self.get_neuron_age(neuron_id)
        return "%06s: %06.4f (%10d / %10d)" % (prefix, frq, cnt, age)


class TriggersTrackerClazzAndSampleID(TriggersTracker):
    """
    Computes how often does neurons trigger on average during a given
    time frame. It also keeps track of the input id and the class id
    of the input that triggered the neuron.

    Args:
        neuron_count (int): The number of neurons in the tracker.
        device (torch.device, optional): The device on which the tracker is
            located. Defaults to None.

    Attributes:
        triggers_by_in_id (List[Set[int]]):
            A list of sets, where each set contains the input ids that
            triggered the corresponding neuron.
        triggers_by_class (List[defaultdict[int]]):
            A list of dictionaries, where each dictionary contains the class
            ids and their corresponding occurrence count for the corresponding
            neuron.
    """
    def __init__(self, neuron_count: int, device=None):
        super().__init__(neuron_count, device)
        self.triggers_by_in_id = [set() for _ in range(neuron_count)]
        self.triggers_by_class = [
            defaultdict(lambda: 0) for _ in range(neuron_count)]

    def __hash__(self):
        return hash(str(self))

    def __repr__(self) -> str:
        triggers_by_class = [
            dict(def_dict) for def_dict in self.triggers_by_class]
        return "TrackerClazzAndSampleID[#%d]: [%s] & [%s] & [%s] & [%s]" % (\
            self.number_of_neurons,
            str(self.triggrs_by_neuron[:_TRACKING_TENSOR_DISPLAY_LIMIT]),
            str(self.updates_by_neuron[:_TRACKING_TENSOR_DISPLAY_LIMIT]),
            str(self.triggers_by_in_id), str(triggers_by_class))

    def __eq__(self, other: "TriggersTrackerClazzAndSampleID") -> bool:
        are_equals = self.number_of_neurons == other.number_of_neurons and \
            th.equal(self.triggrs_by_neuron, other.triggrs_by_neuron) and \
            th.equal(self.updates_by_neuron, other.updates_by_neuron) and \
            self.triggers_by_in_id == other.triggers_by_in_id and \
            self.triggers_by_class == other.triggers_by_class
        return are_equals

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        # Since we have for each neuron a set and a dictionary, we convert
        # each set into a tensor of integers and the dictionary we convert into
        # two tensors one for the keys and one for the values.
        for neuron_id in range(self.number_of_neurons):
            t0 = th.tensor(list(self.triggers_by_in_id[neuron_id])).long()
            state[prefix + 'triggers_by_in_id_%d' % neuron_id] = t0
            t1 = th.tensor(
                list(self.triggers_by_class[neuron_id].keys())).long()
            state[prefix + 'triggers_by_class_%d_keys' % neuron_id] = t1
            t2 = th.tensor(
                list(self.triggers_by_class[neuron_id].values())).long()
            state[prefix + 'triggers_by_class_%d_vals' % neuron_id] = t2
        return state

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

        triggers_by_in_id = []
        triggers_by_class = []

        def trg_key(neuron_id):
            nonlocal prefix
            return prefix + 'triggers_by_in_id_%d' % neuron_id

        def trg_class_key(neuron_id):
            nonlocal prefix
            return prefix + 'triggers_by_class_%d_keys' % neuron_id

        def trg_vals_key(neuron_id):
            nonlocal prefix
            return prefix + 'triggers_by_class_%d_vals' % neuron_id

        for neuron_id in range(self.number_of_neurons):
            triggers_by_in_id.append(
                set(state_dict[trg_key(neuron_id)].tolist()))
            keys = state_dict[trg_class_key(neuron_id)].tolist()
            vals = state_dict[trg_vals_key(neuron_id)].tolist()
            ddct = defaultdict(lambda: 0)
            ddct.update(zip(keys, vals))
            triggers_by_class.append(ddct)
        self.triggers_by_in_id = triggers_by_in_id
        self.triggers_by_class = triggers_by_class

    def update(self, tensor: th.Tensor):
        super().update(tensor)
        # Update trackers with class and sample ids. The shapes are expected
        # to be in the following form:
        #  * tensor:             [batch_size x neuron_count]
        #  * tensor.in_id_batch: [batch_size]
        #  * tensor.label_batch: [batch_size]

        if not hasattr(tensor, 'in_id_batch') or \
                not hasattr(tensor, 'label_batch'):
            raise ValueError(
                'Expected that tensor tensor contains both:'
                'in_id_batch and label_batch tensors.')

        for sample_idx, triggers_per_layer in enumerate(tensor):
            for neuron_idx, triggers in enumerate(triggers_per_layer):
                if triggers > 0:
                    if tensor.in_id_batch is not None:
                        sample_id = tensor.in_id_batch.long()[
                            sample_idx].item()
                        self.triggers_by_in_id[neuron_idx].add(sample_id)
                    if tensor.label_batch is not None:
                        label = tensor.label_batch.long()[sample_idx].item()
                        self.triggers_by_class[neuron_idx][label] += 1

    def reorder(self, indices: List[int]):
        super().reorder(indices)

        triggers_by_in_id = []
        triggers_by_class = []
        for _, index_from in enumerate(indices):
            triggers_by_in_id.append(self.triggers_by_in_id[index_from])
            triggers_by_class.append(self.triggers_by_class[index_from])
        self.triggers_by_in_id = triggers_by_in_id
        self.triggers_by_class = triggers_by_class

    def prune(self, indices: Set[int], update_neuron_count: bool = True):
        super().prune(indices, update_neuron_count=False)
        curr_neurons = set(range(self.number_of_neurons))
        kept_neurons = curr_neurons - indices

        triggers_by_in_id = []
        triggers_by_class = []
        for _, filter_from in enumerate(kept_neurons):
            triggers_by_in_id.append(self.triggers_by_in_id[filter_from])
            triggers_by_class.append(self.triggers_by_class[filter_from])
        self.triggers_by_in_id = triggers_by_in_id
        self.triggers_by_class = triggers_by_class

        if update_neuron_count:
            self.number_of_neurons = len(kept_neurons)

    def reset(self, indices: Set[int]):
        super().reset(indices)
        for neuron_id in indices:
            self.triggers_by_in_id[neuron_id] = set()
            self.triggers_by_class[neuron_id] = defaultdict(lambda: 0)

    def add_neurons(self, neuron_count: int):
        super().add_neurons(neuron_count)
        self.triggers_by_in_id.extend(set() for _ in range(neuron_count))
        self.triggers_by_class.extend(
            [defaultdict(lambda: 0) for _ in range(neuron_count)]
        )

    def get_neuron_triggers_by_input_id(self, neuron_id: int):
        """
        Returns the list of samples id for which the neuron triggered.

        Args:
            neuron_id (int): The id of the neuron for which to get the
                triggers.
        """
        return list(self.triggers_by_in_id[neuron_id])

    def get_neuron_triggers_by_label(
            self, neuron_id: int, order_by_occurences: bool = False):
        """
        Return the histogram broken down by labels of neuron triggers.

        Args:
            neuron_id (int): The id of the neuron for which to get the
                triggers.
            order_by_occurences (bool, optional): Whether to order the
                histogram by occurences or not. Defaults to False.
        """
        if not order_by_occurences:
            return self.triggers_by_class[neuron_id]
        return sorted(
            self.triggers_by_class[neuron_id].items(),
            key=lambda class_id_and_occurences: class_id_and_occurences[1],
            reverse=True)

    def get_neuron_stats(self, neuron_id: int, cutoff: int = 10):
        """
        Return a structured version of the neuron stats.

        Args:
            neuron_id (int): The id of the neuron for which to get the
                stats.
            cutoff (int, optional): The number of items to return for the
                input ids and labels. Defaults to 10.
        """
        try:
            count = self.get_neuron_age(neuron_id)
            trggr = self.get_neuron_triggers(neuron_id)
            in_id = self.get_neuron_triggers_by_input_id(neuron_id)[:cutoff]
            label = self.get_neuron_triggers_by_label(
                neuron_id, order_by_occurences=True)
            return (trggr / max(count, 1), in_id, label)
        except IndexError as e:
            print("TrackerClazzAndSampleID.get_neuron_stats error ", e)

    def get_neuron_pretty_repr(self, neuron_id: int, prefix: str = '') -> str:
        """
        Get a pretty representation of the neuron statistics.

        Args:
            neuron_id (int): The id of the neuron for which to get the
                stats.
            prefix (str, optional): The prefix to use for the representation.
                Defaults to ''.
        """
        prefix = prefix[:5]
        frq, ids, lbl = self.get_neuron_stats(neuron_id)
        cnt = self.get_neuron_triggers(neuron_id)
        age = self.get_neuron_age(neuron_id)

        return "%6s: %6.4f (%6d / %6d) [ids:%20s | lbl:%20s]" % (
            prefix, frq, cnt, age, ids, lbl)
