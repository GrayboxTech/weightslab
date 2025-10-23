import torch as th
import torch.nn.functional as F

from torch import nn
from copy import deepcopy


class Monitor:
    def display_stats(self, experiment: "Experiment"):
        raise NotImplementedError()


class NeuronTriggeringStatsMonitor(Monitor):

    def print_counters(self, layer):
        legend = "trigger_rate (triggers_count / number_of_seen_samples)"
        if isinstance(layer, nn.Conv2d):
            print("Conv2dLayer stats:" + legend)
        elif isinstance(layer, nn.Linear):
            print("LinearLayer stats:" + legend)
        else:
            return

        neuron_count = layer.train_dataset_tracker.get_neuron_number()
        for neuron_id in range(neuron_count):
            train_stats = layer.train_dataset_tracker.get_neuron_pretty_repr(
                neuron_id, 'train')
            eval_stats = layer.eval_dataset_tracker.get_neuron_pretty_repr(
                neuron_id, 'eval ')
            print("Neuron %05d:" % neuron_id, train_stats, end="\n")
            print("            :", eval_stats, end="\n")

    def display_stats(self, experiment: "Experiment"):
        print("=" * 100)
        for layer_idx, layer in enumerate(experiment.model.layers()):
            print(f"Layer#{layer_idx} ", end='')
            self.print_counters(layer)
            print('-' * 100)
        print("=" * 100)


class NeuronStatsWithDifferencesMonitor(Monitor):
    def __init__(self):
        self.last_model = None

        self.diff_nocrement_color_code = "1;36;40"
        self.diff_decrement_color_code = "1;31;40"
        self.diff_increment_color_code = "1;32;40"

        self.relative_change_percentage_cap = 100

    def _capped_relative_diff(self, relative_change: float):
        if abs(relative_change) > self.relative_change_percentage_cap:
            return self.relative_change_percentage_cap
        return relative_change

    def _value_2_sign(self, value: float):
        if value < 0:
            return "-"

        if value > 0:
            return "+"

        return " "

    def _sign_2_color_code(self, sign: str = "+"):
        if sign == "+":
            return self.diff_increment_color_code
        elif sign == "-":
            return self.diff_decrement_color_code
        else:
            return self.diff_nocrement_color_code

    def _diff_to_color_coded_text(self, diff_value: float):
        sign = self._value_2_sign(diff_value)
        dlta = self._capped_relative_diff(abs(diff_value))
        clr_start = "\x1b[%sm" % self._sign_2_color_code(sign)
        clr_stopp = "\x1b[0m"

        text = f"{clr_start}"
        text += "%s%8.4f" % (sign, dlta)
        text += f"%{clr_stopp}"
        return text

    def _colored_value_relative_diff(self, value: float, base_value: float):
        dlta = (value - base_value) / (base_value + 0.001)
        sign = self._value_2_sign(dlta)
        dlta = self._capped_relative_diff(abs(dlta)) * 100
        clr_start = "\x1b[%sm" % self._sign_2_color_code(sign)
        clr_stopp = "\x1b[0m"

        text = "%6.2f" % value
        text += f"[{clr_start}"
        text += "%s%10.2f" % (sign, dlta)
        text += f"%{clr_stopp}]"

        return text

    def pretty_str_stats(
            self, neuron_id, tracker_current, tracker_diff_base, prefix):
        frq_curr, cnt_curr, age_curr = 0, 0, 0
        frq_curr = tracker_current.get_neuron_stats(neuron_id)
        cnt_curr = tracker_current.get_neuron_triggers(neuron_id)
        age_curr = tracker_current.get_neuron_age(neuron_id)

        if tracker_diff_base is None or \
                neuron_id >= tracker_diff_base.number_of_neurons:
            text = "%s: %8.4f (%10d / %10d)" % (
                prefix, frq_curr, cnt_curr, age_curr)
            return text

        frq_bfor = tracker_diff_base.get_neuron_stats(neuron_id)
        # Format frequency displaying message.
        frq_diff = self._colored_value_relative_diff(frq_curr, frq_bfor)
        text = "%s: %s (%10d / %10d)" % (
                prefix, frq_diff, cnt_curr, age_curr)

        return text

    def pretty_diff_ds_stats_trigers(
            self, neuron_id, train_tracker, eval_tracker, prefix_diff,
            prefix_ratio):
        if train_tracker is None or eval_tracker is None:
            return ""

        frq_curr_train = train_tracker.get_neuron_stats(neuron_id)
        frq_curr_eval = eval_tracker.get_neuron_stats(neuron_id)

        frq_diff = frq_curr_train - frq_curr_eval
        frq_rtio = frq_diff / (frq_curr_train + 0.001)

        # Format frequency displaying message.

        text = "%s: (%6.2f) %s: (%6.2f)" % (
            prefix_diff, frq_diff, prefix_ratio, frq_rtio)

        return text

    def _compute_bias_relative_diff(self, neuron_id, layer, prev_exp_layer):
        if prev_exp_layer is None or \
                prev_exp_layer.bias.shape != layer.bias.shape:
            return 0
        bias_diff = layer.bias.data[neuron_id] - \
            prev_exp_layer.bias.data[neuron_id]
        return bias_diff / (prev_exp_layer.bias.data[neuron_id] + 0.001)

    def _compute_wght_relative_diff(self, neuron_id, layer, prev_exp_layer):
        if prev_exp_layer is None or \
                prev_exp_layer.weight.shape != layer.weight.shape:
            return 0
        wght_diff = layer.weight.data[neuron_id] - \
            prev_exp_layer.weight.data[neuron_id]

        rltv_diff = th.abs(wght_diff) / (
            th.abs(prev_exp_layer.weight.data[neuron_id]) + 0.001)

        return th.mean(rltv_diff)

    def print_counters(self, layer, prev_exp_layer):
        legend = "trigger_rate (triggers_count / number_of_seen_samples)"
        if isinstance(layer, nn.Conv2d):
            print("Conv2dLayer stats:" + legend)
        elif isinstance(layer, nn.Linear):
            print("LinearLayer stats:" + legend)
        else:
            return

        prev_exp_layer_tracker_train = None
        prev_exp_layer_tracker_eval = None
        if prev_exp_layer is not None:
            prev_exp_layer_tracker_train = prev_exp_layer.train_dataset_tracker
            prev_exp_layer_tracker_eval = prev_exp_layer.eval_dataset_tracker

        for neuron_id in range(layer.train_dataset_tracker.number_of_neurons):
            print("Neuron %05d:" % neuron_id, end=' ')
            wght_change = self._compute_wght_relative_diff(
                neuron_id, layer, prev_exp_layer) * 100
            print('W:', self._diff_to_color_coded_text(wght_change), end=' ')
            bias_change = self._compute_bias_relative_diff(
                neuron_id, layer, prev_exp_layer) * 100
            print('b:', self._diff_to_color_coded_text(bias_change), end=' ')
            print(self.pretty_str_stats(
                neuron_id, layer.train_dataset_tracker,
                prev_exp_layer_tracker_train, "T"), end=' ')
            print(self.pretty_str_stats(
                neuron_id, layer.eval_dataset_tracker,
                prev_exp_layer_tracker_eval, "E"), end=' ')

            print(self.pretty_diff_ds_stats_trigers(
                neuron_id, layer.train_dataset_tracker,
                layer.eval_dataset_tracker, "T-E", "T/E"))

    def display_stats(self, experiment: "Experiment"):
        print("=" * 100)
        for layer_idx, layer in enumerate(experiment.model.layers):
            prev_exp_layer = None
            if self.last_model is not None:
                prev_exp_layer = self.last_model.layers[
                    layer_idx]
            print(f"Layer#{layer_idx} ", end='')
            self.print_counters(layer, prev_exp_layer)
            print('-' * 100)
        print("=" * 100)

        self.last_model = deepcopy(experiment.model)


class PairwiseNeuronSimilarity(Monitor):
    def __init__(self):
        self.style = "1"
        self.positive_bg = "41"
        self.negative_bg = "42"

    def _value_to_fg_color(self, value: float) -> str:
        if value >= 0.7:
            return "33"  # yellow
        return "30"  # black

    def _value_to_bg_color(self, value: float) -> str:
        if value >= 0.0:
            return self.positive_bg
        else:
            return self.negative_bg

    def _value_to_colored_text(self, value: float):
        fg = self._value_to_fg_color(value)
        bg = self._value_to_bg_color(value)
        clr_start = ";".join([self.style, fg, bg])
        clr_start = "\x1b[%sm" % clr_start
        clr_stopp = "\x1b[0m"
        sign = "-" if value < 0 else " "
        text = f"{clr_start}"
        text += "%s%2.2f" % (sign, abs(value))
        text += f"{clr_stopp}"
        return text

    def two_d_tensor_str(self, tensor: th.Tensor) -> str:
        repr = ""
        # Build the first row
        repr += " " * 6
        repr += " ".join(
            ["%5d" % col_idx for col_idx in range(tensor.shape[0])])
        repr += "\n"

        for row_idx in range(tensor.shape[0]):
            repr += "%5d" % row_idx
            for col_idx in range(tensor.shape[0]):
                repr += " "
                repr += self._value_to_colored_text(tensor[row_idx][col_idx])
            repr += "\n"
        return repr

    def display_stats(self, experiment: "Experiment"):
        print("=" * 100)
        for layer_idx, layer in enumerate(experiment.model.layers):
            number_of_neurons = layer.weight.data.shape[0]
            pairwise_similarities = th.zeros(
                (number_of_neurons, number_of_neurons))
            print(f"Layer#{layer_idx} ", end='\n')
            weight = layer.weight.data.view(number_of_neurons, -1)

            for ni in range(number_of_neurons):
                for nj in range(number_of_neurons):
                    pairwise_similarities[ni][nj] = F.cosine_similarity(
                        weight[ni], weight[nj], dim=0)
            th.set_printoptions(precision=2, linewidth=160, sci_mode=False)
            if number_of_neurons <= 24:
                print(self.two_d_tensor_str(pairwise_similarities))
            else:
                print(pairwise_similarities)
            th.set_printoptions(profile='default')
            print('-' * 100)
        print("=" * 100)
