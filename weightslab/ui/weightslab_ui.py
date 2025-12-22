import io
import re
import ast
import dash
import plotly.graph_objs as go

import dash_daq as daq
import dash_bootstrap_components as dbc

import argparse
import base64
import grpc
import os
import time
import threading
import sys
import dash_bootstrap_components as dbc
import weightslab.proto.experiment_service_pb2 as pb2
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc
import pandas as pd
import plotly.graph_objs as go
import logging
import collections
import numpy as np
import random
import hashlib

from dash import dcc
from dash import html
from dash import dash_table
from dash import dcc, html, MATCH, ALL, no_update, ctx
from dash.dependencies import Input, Output, State
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import State

from typing import Tuple, Dict, List, Any
from flask import Response, request, abort, jsonify
from enum import Enum
from io import BytesIO
from PIL import Image
from collections import defaultdict
from dash.dash_table.Format import Format, Scheme
from weightslab.ui.utils.scope_timer import ScopeTimer
from dataclasses import dataclass
from math import isqrt


# Set up logging
logger = logging.getLogger("ui")


_HYPERPARAM_COLUMNS = ["label", "type", "name", "value"]

_NEURONS_DF_COLUMNS = [
    "layer_id", "neuron_id", "Age", "RTrn", "REval", "ADiff",
    "RDiff", "Frozen", "Status", "layer_type", "highlighted"]

_LAYER_DF_COLUMNS = [
    "layer_id", "layer_type", "layer_name", "outgoing", "incoming",
    "kernel_size", "sorted_by"]

_METRICS_DF_COLUMNS = [
    "experiment_name", "model_age", "metric_name", "metric_value", "run_id"]

_ANNOTATIONS_DF_COLUMNS = [
    "experiment_name", "model_age", "annotation", "metadata"]

_SAMPLES_DF_COLUMNS = [
    "SampleId", "Target", "Prediction", "LastLoss", "Encounters", "Discarded"
]

_PLOTS_COLOR_WHEEL = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-Green
    "#17becf",  # Cyan
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
    "#ff9896",  # Light Red
    "#c5b0d5",  # Light Purple
    "#c49c94",  # Light Brown
    "#f7b6d2",  # Light Pink
    "#c7c7c7",  # Light Gray
    "#dbdb8d",  # Light Yellow-Green
    "#9edae5"   # Light Cyan
]

_DISPLAY_COLUMNS = [
    "SampleId", "Target", "Prediction", "LastLoss"
]

_BUTTON_STYLE = {
    'width': '6vw',
    'height': '8vh',
}

_DEFAULT_CHECKLIST_VALUES = [
    "neuron_id", "neuron_age", "trigger_rate_train", "Status"]


_LAYER_BUTTON_WIDTH = '3vw'
_LAYER_BUTTON_HEIGHT = '5vw'
_LAYER_BUTTON_FNT_SZ = '30px'
LYR_BASE_WIDTH = 140
WIDTH_PER_COLUMN = 80


# Custom CSS styles (keep the existing styles)
custom_styles = {
    'container': {
        'background': 'linear-gradient(135deg, #E0E5EC 0%, #B8C6DB 100%)',
        'min-height': '100vh',
        'padding': '20px',
        'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    },
    'main_card': {
        'background': 'rgba(255, 255, 255, 0.95)',
        'backdrop-filter': 'blur(20px)',
        'border-radius': '20px',
        'box-shadow': '0 10px 20px rgba(0, 0, 0, 0.1)',
        'border': '1px solid rgba(255, 255, 255, 0.2)',
        'width': '95vw',      # Changed to 95% of viewport width
        'margin': '0 auto',
        'padding': '20px'
    },
    'header': {
        'display': 'flex',
        'justify-content': 'space-between',
        'align-items': 'center',
        'margin-bottom': '20px',
        'padding': '0 10px'
    },
    'logo': {
        'font-size': '24px',
        'font-weight': '700',
        'color': '#3B5998',
        'letter-spacing': '-0.5px',
        'margin': '0'
    },
    'config_section': {
        'padding': '0px'
    },
    'config_card': {
        'background': '#F7FAFC',
        'border-radius': '8px',
        'padding': '6px',
        'margin-bottom': '6px',
        'border': '1px solid #E2E8F0',
        'transition': 'all 0.3s ease'
    },
    'config_label': {
        'font-weight': '600',
        'color': '#4A5568',
        'font-size': '10px',
        'text-transform': 'uppercase',
        'letter-spacing': '0.5px',
        'width': '40%',
        'display': 'inline-block',
        'vertical-align': 'middle'
    },
    'input_style': {
        'border-radius': '4px',
        'border': '1px solid #E2E8F0',
        'font-weight': '500',
        'font-size': '10px',
        'height': '24px',
        'width': '60%',
        'display': 'inline-block',
        'vertical-align': 'middle'
    },
    'play_button': {
        'width': '48px',
        'height': '48px',
        'border-radius': '50%',
        'background': '#4267B2',
        'border': 'none',
        'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.1)',
        'cursor': 'pointer',
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'color': 'white',
        'font-size': '24px',
        'transition': 'all 0.3s ease',
        'padding': '0'
    },
    'logo_w': {
        'color': '#5D7A5D',  # Gray-green color
    },
    'logo_l': {
        'color': '#A07777',  # Gray-red color
    },
}

class KeyValueFormatter(logging.Formatter):
    _std = {
        'name','msg','args','levelname','levelno','pathname','filename','module',
        'exc_info','exc_text','stack_info','lineno','funcName','created','msecs',
        'relativeCreated','thread','threadName','processName','process','asctime',
        'message'
    }
    def format(self, record):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(record.created))
        base = f"{ts} {record.getMessage()}"
        extras = {k:v for k,v in record.__dict__.items() if k not in self._std}
        if not extras:
            return base
        lines = [base]
        for k, v in sorted(extras.items()):
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

def exponential_smoothing(values, alpha=0.6):
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    values = np.asarray(values, dtype=float)
    smoothed_values = []
    for idx, val in enumerate(values):
        if idx == 0:
            smoothed_values.append(val)
        else:
            smoothed_val = val * alpha + (1 - alpha) * smoothed_values[idx - 1]
            smoothed_values.append(smoothed_val)
    return smoothed_values


@dataclass
class PlotPoint:
    x: float | None
    y: float | None


class UIState:
    """
        A class to represent the state of the UI and all the objects and
        their properties necessary to be maintained for the UI to function.
    """
    def __init__(self, root_directory: str):
        self.root_directory = root_directory
        os.makedirs(self.root_directory, exist_ok=True)
        self.dirty = False
        self.dirty_dfs = []

        # Details about the hyperparameters
        self.hyperparam = pd.DataFrame(columns=_HYPERPARAM_COLUMNS)
        # Details about the layers
        self.layers_df = pd.DataFrame(columns=_LAYER_DF_COLUMNS)
        # Details about neurons
        self.neurons_df = pd.DataFrame(columns=_NEURONS_DF_COLUMNS)
        # Details about the metrics
        self.metrics_df = pd.DataFrame(columns=_METRICS_DF_COLUMNS)
        # Details about the annotations
        self.annotation = pd.DataFrame(columns=_ANNOTATIONS_DF_COLUMNS)
        # Details about the data
        self.samples_df = pd.DataFrame(columns=_SAMPLES_DF_COLUMNS)
        # Details about the eval data
        self.eval_samples_df = pd.DataFrame(columns=_SAMPLES_DF_COLUMNS)
        self.task_type = 'classification'


        self.metrics_df_path = os.path.join(
            self.root_directory, "statuses_df.csv")
        self.annotation_path = os.path.join(
            self.root_directory, "annotations.csv")

        if os.path.exists(self.metrics_df_path):
            self.metrics_df = pd.read_csv(self.metrics_df_path)
        if os.path.exists(self.annotation_path):
            self.annotation = pd.read_csv(self.annotation_path)

        # These amount are in vw
        self.layer_div_width_per_col = 4.5
        self.layer_div_width_minimum = 20
        self.layer_id_to_df_row_idx = {}  # layer_id -> idx
        self.neuron_id_to_df_row_idx = {}  # (layer_id, neuron_id) -> idx

        self.selected_neurons = defaultdict(lambda: [])  # layer_id -> List[neuron_id]
        self.lock = threading.RLock()
        self.metrics_lock = threading.RLock()

        # Cache plots for faster rendering
        self.exp_name_2_color = defaultdict(lambda: "blue")
        self.exp_name_metric_name_2_plot = defaultdict(lambda: None)
        self.exp_name_metric_name_2_anot = defaultdict(lambda: [])
        self.exp_name_2_need_redraw = defaultdict(lambda: False)
        self.current_run_id = 0

        self.exp_names = set()
        self.met_names = set()
        self.ant_names = set()

        self.exp_name_metric_name_annot_name_2_values = defaultdict(list)
        if not self.metrics_df.empty:
            self.exp_names = set(self.metrics_df['experiment_name'].unique())
            self.met_names = set(self.metrics_df['metric_name'].unique())
            for idx, exp_name in enumerate(self.exp_names):
                self.exp_name_2_color[exp_name] = _PLOTS_COLOR_WHEEL[
                    idx % len(_PLOTS_COLOR_WHEEL)]
        if not self.annotation.empty:
            self.ant_names = set(self.annotation['annotation'].unique())

        self.plot_name_2_selection_point = defaultdict(lambda: None)
        self.plot_name_2_curr_head_point = defaultdict(lambda: None)
        if not self.metrics_df.empty:
            for met_name in self.met_names:
                relevant_df = self.metrics_df.query(
                    f"metric_name == '{met_name}'")
                if not relevant_df.empty:
                    self.plot_name_2_curr_head_point[met_name] = PlotPoint(
                        relevant_df.iloc[-1]["model_age"],
                        relevant_df.iloc[-1]["metric_value"])

    def get_neurons_df(self):
        with self.lock:
            return self.neurons_df.copy()

    def get_layers_df(self):
        with self.lock:
            return self.layers_df

    def get_plots_for_exp_name_metric_name(
            self, metric_name, exp_name):

        key = (metric_name, exp_name)
        if self.exp_name_metric_name_2_plot[key] and \
                not self.exp_name_2_need_redraw[exp_name]:
            return [self.exp_name_metric_name_2_plot[key]] + \
                self.exp_name_metric_name_2_anot[key]

        with self.metrics_lock:
            relevant_df = self.metrics_df.query(
                f"metric_name == '{metric_name}' and "
                f"experiment_name == '{exp_name}'")

        if relevant_df.empty:
            return []
        first = True
        if 'run_id' in relevant_df.columns:
            traces = []
            for run_id, run_df in relevant_df.groupby("run_id"):
                trace = go.Scatter(
                    x=run_df["model_age"],
                    y=run_df["metric_value"],
                    mode='lines',
                    name=exp_name,
                    line=dict(color=self.exp_name_2_color[exp_name]),
                    showlegend=first,
                )
                traces.append(trace)
                first = False
        else:
            # fallback: original behavior
            traces = [go.Scatter(
                x=relevant_df["model_age"],
                y=relevant_df["metric_value"],
                mode='lines',
                name=exp_name,
                line=dict(color=self.exp_name_2_color[exp_name]),
            )]

        annotation_plots = self._get_annot_plots(exp_name, metric_name)
        return traces + annotation_plots


    def _get_annot_plots(self, exp_name, met_name):
        plots = []
        for anot_name in self.ant_names:
            with self.metrics_lock:
                relevant_df = self.annotation.query(
                    f"experiment_name == '{exp_name}' and "
                    f"annotation == '{anot_name}'")
            if relevant_df.empty:
                continue

            metric_name_2_annot_values = defaultdict(list)
            for _, row in relevant_df.iterrows():
                metadadata = ast.literal_eval(row["metadata"])
                for k, v in metadadata.items():
                    metric_name_2_annot_values[k].append(v)

            # Translate annotation name to metadata key
            annotation_name_in_keys = anot_name
            for key in metric_name_2_annot_values:
                if anot_name in key:
                    annotation_name_in_keys = key
                    break

            if met_name not in metric_name_2_annot_values:
                continue
            if not metric_name_2_annot_values[met_name]:
                continue
            anot = go.Scatter(
                x=relevant_df["model_age"],
                y=metric_name_2_annot_values[met_name],
                mode='markers',
                marker_symbol="diamond",
                name=f"ckpt-{exp_name}",
                customdata=metric_name_2_annot_values[annotation_name_in_keys],
                marker={
                    "color": self.exp_name_2_color[exp_name],
                    "size": 10,
                }
            )
            plots.append(anot)
        return plots

    def __repr__(self) -> str:
        return f"UIState[{self.root_directory}][dirty={self.dirty}]" + \
            f"({len(self.hyperparam)} hyper-parameters, " + \
            f"{len(self.layers_df)} monitored layers, " + \
            f"{len(self.neurons_df)} monitored neurons, " + \
            f"{len(self.metrics_df)} metrics values, " + \
            f"{len(self.annotation)} annotations, " + \
            f"{len(self.samples_df)} monitored samples, " + \
            f"{self.exp_names} " + \
            f"{self.met_names} " + \
            f"{self.ant_names} " + \
            ")"

    def update_from_server_state(
        self, server_state: pb2.CommandResponse):
        """
            Update the UI state with the new state from the server.
        """
        # print("[UIState] Updating from server state.")
        self.update_hyperparams_from_server(
            server_state.hyper_parameters_descs)
        self.update_neurons_from_server(
            server_state.layer_representations)
        self.update_samples_from_server(
            server_state.sample_statistics)

    def update_hyperparams_from_server(
            self, hyper_parameters_descs: List[pb2.HyperParameterDesc]):
        hyper_parameters_descs.sort(key=lambda x: x.name)
        hidx = -1
        for hidx, hyper_parameter_desc in enumerate(hyper_parameters_descs):
            if hyper_parameter_desc.type == "number":
                self.hyperparam.loc[hidx] = [
                    hyper_parameter_desc.label,
                    hyper_parameter_desc.type,
                    hyper_parameter_desc.name,
                    hyper_parameter_desc.numerical_value
                ]
            else:
                self.hyperparam.loc[hidx] = [
                    hyper_parameter_desc.label,
                    hyper_parameter_desc.type,
                    hyper_parameter_desc.name,
                    hyper_parameter_desc.string_value
                ]
        self.hyperparam.loc[hidx + 1] = [
            "Play/Pause Train",
            "button",
            "play_pause",
            False, # is_training
        ]

    def layer_representation_to_df_row(self, layer_representation):
        layer_row = [
            layer_representation.layer_id,
            layer_representation.layer_type,
            layer_representation.layer_name,
            layer_representation.neurons_count,
            layer_representation.incoming_neurons_count,
            layer_representation.kernel_size,
            None,
        ]
        return layer_row

    def neuron_statistics_to_df_row_v2(self, neuron_stats):
        adiff = abs(
                neuron_stats.train_trigger_rate - \
                neuron_stats.eval_trigger_rate)
        rdiff = 0
        if neuron_stats.train_trigger_rate > 0:
            rdiff = adiff / neuron_stats.train_trigger_rate
        neuron_row = [
            -1,
            neuron_stats.neuron_id.neuron_id,
            neuron_stats.neuron_age,
            neuron_stats.train_trigger_rate,
            neuron_stats.eval_trigger_rate,
            adiff,
            rdiff,
            neuron_stats.learning_rate == 0,
            get_neuron_status(neuron_stats).value,
            '',
            False,  # highlighted
        ]
        return neuron_row

    def neuron_statistics_to_df_row(self, layer_representation, neuron_stats):
        adiff = abs(
                neuron_stats.train_trigger_rate - \
                neuron_stats.eval_trigger_rate)
        rdiff = 0
        if neuron_stats.train_trigger_rate > 0:
            rdiff = adiff / neuron_stats.train_trigger_rate
        neuron_row = [
            layer_representation.layer_id,
            neuron_stats.neuron_id.neuron_id,
            neuron_stats.neuron_age,
            neuron_stats.train_trigger_rate,
            neuron_stats.eval_trigger_rate,
            adiff,
            rdiff,
            neuron_stats.learning_rate == 0,
            get_neuron_status(neuron_stats).value,
            layer_representation.layer_type,
            False,  # highlighted
        ]
        return neuron_row

    def update_neurons_from_server(
            self, layer_representations: List[pb2.LayerRepresentation]):
        """
            Update the neurons dataframe with the new neurons details.
        """
        if not layer_representations:
            return
        neuron_row_idx = 0
        neurons_df = pd.DataFrame(columns=_NEURONS_DF_COLUMNS)
        layers_df = pd.DataFrame(columns=_LAYER_DF_COLUMNS)

        for lyr_idx, layer_representation in enumerate(layer_representations):
            layer_row = self.layer_representation_to_df_row(
                layer_representation)
            layers_df.loc[lyr_idx] = layer_row
            self.layer_id_to_df_row_idx[
                layer_representation.layer_id] = lyr_idx

            for neuron_stats in layer_representation.neurons_statistics:
                neuron_row = self.neuron_statistics_to_df_row(
                    layer_representation, neuron_stats)
                neurons_df.loc[neuron_row_idx] = neuron_row
                neuron_row_idx += 1

        neurons_df.set_index(["layer_id", "neuron_id"], inplace=True)

        with self.lock:
            self.neurons_df = neurons_df
            self.layers_df = layers_df

    def update_metrics_from_server(
            self, status: pb2.TrainingStatusEx):
        # print(f"UIState.update_metrics_from_server: {status}")
        self.exp_names.add(status.experiment_name)
        self.exp_name_2_need_redraw[status.experiment_name] = True

        if status.HasField("metrics_status"):
            metrics_row = [
                status.experiment_name,
                status.model_age,
                status.metrics_status.name,
                status.metrics_status.value,
                self.current_run_id
            ]
            with self.metrics_lock:
                self.metrics_df.loc[len(self.metrics_df)] = metrics_row

            self.plot_name_2_curr_head_point[status.metrics_status.name] = \
                PlotPoint(status.model_age, status.metrics_status.value)

            if len(self.metrics_df) % 1000 == 999:
                with self.metrics_lock:
                    self.metrics_df.to_csv(self.metrics_df_path, index=False)
                logger.info(
                    "metrics_checkpoint_saved",
                    extra={
                        "experiment": status.experiment_name,
                        "rows": int(len(self.metrics_df)),
                        "path": self.metrics_df_path,
                        "run_id": self.current_run_id
                    }
                )
            self.met_names.add(status.metrics_status.name)

            if status.experiment_name not in self.exp_name_2_color:
                self.exp_name_2_color[status.experiment_name] = \
                    _PLOTS_COLOR_WHEEL[ \
                        len(self.exp_names) % len(_PLOTS_COLOR_WHEEL)]
        elif status.HasField("annotat_status"):
            other_metrics = {}
            with self.metrics_lock:
                for _, row in self.metrics_df.iloc[::-1].iterrows():
                    if row["experiment_name"] == status.experiment_name and \
                            row['metric_name'] not in other_metrics:
                        other_metrics[row['metric_name']] = row["metric_value"]
                    if len(other_metrics) == len(self.met_names):
                        break

            metadata = status.annotat_status.metadata
            metadata.update(other_metrics)
            annotation_row = [
                status.experiment_name,
                status.model_age,
                status.annotat_status.name,
                str(metadata),  # TODO: revise this
            ]
            with self.metrics_lock:
                self.annotation.loc[len(self.annotation)] = annotation_row
                self.annotation.to_csv(self.annotation_path, index=False)
            self.ant_names.add(status.annotat_status.name)

            logger.info(
                "annotation_recorded",
                extra={
                    "experiment": status.experiment_name,
                    "model_age": status.model_age,
                    "annotation": status.annotat_status.name,
                    "annotation_path": self.annotation_path,
                    "run_id": self.current_run_id
                }
            )

    # updated function to take lists instead of maps

    def update_samples_from_server(self, sample_statistics: pb2.SampleStatistics):
        try:
            rows = []
            extra_keys = set()
            self.task_type = getattr(sample_statistics, 'task_type', self.task_type)
            for record in sample_statistics.records:
                row = {
                    "SampleId": int(record.sample_id),
                    "Target": list(record.sample_label),
                    "Prediction": list(record.sample_prediction),
                    "LastLoss": float(record.sample_last_loss),
                    "Encounters": int(record.sample_encounters),
                    "Discarded": bool(record.sample_discarded),
                }
                for field in getattr(record, "extra_fields", []):
                    if field.HasField("float_value"):
                        row[field.name] = field.float_value
                    elif field.HasField("int_value"):
                        row[field.name] = field.int_value
                    elif field.HasField("string_value"):
                        row[field.name] = field.string_value
                    elif field.HasField("bytes_value"):
                        row[field.name] = field.bytes_value
                    elif field.HasField("bool_value"):
                        row[field.name] = field.bool_value
                    else:
                        row[field.name] = None
                    extra_keys.add(field.name)
                rows.append(row)

            all_columns = list(_SAMPLES_DF_COLUMNS) + sorted(extra_keys)

            df = pd.DataFrame(rows)
            for col in all_columns:
                if col not in df.columns:
                    df[col] = None
            with self.lock:
                if sample_statistics.origin == "train":
                    self.samples_df = df[all_columns]
                elif sample_statistics.origin == "eval":
                    self.eval_samples_df = df[all_columns]

        except Exception:
            logger.exception(
                "sample_update_failed",
                extra={"origin": getattr(sample_statistics, "origin", None)}
            )

    def get_layer_df_row_by_id(self, layer_id: int):
        with self.lock:
            return self.layers_df.loc[self.layer_id_to_df_row_idx[layer_id]]


def get_pause_play_button():
    return dbc.Button(
        id='resume-pause-train-btn',
        children="▶",
        style=custom_styles['play_button'],
        n_clicks=0
    )


def get_label_and_input_row(
    dfrow_param_desc: Dict[str, Any] | None = None
):
    label = dfrow_param_desc["label"]
    ident = dfrow_param_desc["name"]
    type_ = dfrow_param_desc["type"]
    deflt = dfrow_param_desc["value"]
    return html.Div([
        html.Label(label, style=custom_styles['config_label']),
        dbc.Input(
            id={"type": "hyper-params-input", "idx": ident},
            type=type_,
            value=deflt,
            style=custom_styles['input_style']
        )
    ], style=custom_styles['config_card'])


def get_header_hyper_params_div(
        ui_state: UIState | None = None
) -> html.Div:
    children = []
    for idx, row in ui_state.hyperparam.iterrows():
        if row["name"] == "play_pause":
            continue
        children.append(
            dbc.Col([get_label_and_input_row(row)], md=3, lg=2))

    section = html.Div(
        id="hyper-parameters-panel",
        children=[
            dbc.Row(
                id="hyper-params-row",
                children=children,
                style={
                    "textWeight": "bold",
                    "width": "100%",  # Changed from 80vw to 100%
                    "align": "center",
                    'margin': '0 auto',
                    'padding': '5px',
                    'flexWrap': 'wrap'  # Ensure items wrap to next line
                }
            ),
        ],
        style={
            # Removed backgroundColor: "#DDD"
            "width": "100%"  # Ensure full width
        }
    )

    header_div = html.Div([
        html.H1("WeightsLab", style={'textAlign': 'center', 'display': 'inline-block', 'marginRight': '20px'}),
        get_pause_play_button()
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'})

    return html.Div(
        [header_div, section],
        style=custom_styles['main_card']
    )

def _get_next_layer_id(ui_state: UIState, curr_layer_id: int) -> int | None:
    layers_df = ui_state.get_layers_df().sort_values("layer_id")
    ids = list(layers_df["layer_id"].values)
    if curr_layer_id not in ids:
        return None
    idx = ids.index(curr_layer_id)
    return ids[idx + 1] if idx + 1 < len(ids) else None

def _get_incoming_count(ui_state: UIState, layer_id: int) -> int | None:
    row = ui_state.get_layer_df_row_by_id(layer_id)
    if isinstance(row, pd.Series) and 'incoming' in row.index:
        val = int(row['incoming'])
        return val
    return None

class NeuronStatus(str, Enum):
    """Enum for neuron status in order to properly color code them."""
    NEUTRAL = "N/A"  # "neutral"
    OVERFIT = "OVRFT"  # "overfit"
    SUSPECT = "BAD"  # "suspect"
    IMPACT0 = "DEAD"  # "impact0"
    FROZEND = "FROZEN"  # "frozend"
    HEALTHY = "GREAT"  # "healthy"


def get_minus_neurons_button(layer_id):
    button = dbc.Button(
        "-",
        id={"type": "layer-rem-btn", "layer_id": layer_id},
        color='transparent',
        n_clicks=0,
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "red",
            'width': _LAYER_BUTTON_WIDTH,
            'height': _LAYER_BUTTON_HEIGHT,
        }
    )
    return button


def get_plus_neurons_button(layer_id):
    button = dbc.Button(
        "+",
        id={"type": "layer-add-btn", "layer_id": layer_id},
        color='transparent',
        n_clicks=0,
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "green",
            'width': _LAYER_BUTTON_WIDTH,
            'height': _LAYER_BUTTON_HEIGHT,
        }
    )
    return button


def get_inspect_neurons_button(layer_id):
    button = dbc.Button(
        "🔍",
        id={"type": "layer-see-btn", "layer_id": layer_id},
        color='transparent',
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "blue",
            'width': _LAYER_BUTTON_WIDTH,
            'height':_LAYER_BUTTON_HEIGHT,
        }
    )
    return button


def get_freeze_neurons_button(layer_id):
    button = dbc.Button(
        "❄",
        id={"type": "layer-freeze-btn", "layer_id": layer_id},
        color='transparent',
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "blue",
            'width': _LAYER_BUTTON_WIDTH,
            'height': _LAYER_BUTTON_HEIGHT,
        }
    )
    return button

def get_reset_neurons_button(layer_id):
    button = dbc.Button(
        "R",
        id={"type": "layer-reset-btn", "layer_id": layer_id},
        color='transparent',
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "black",
            'width': _LAYER_BUTTON_WIDTH,
            'height': _LAYER_BUTTON_HEIGHT,
        }
    )
    return button


def get_layer_ops_buttons(layer_id):
    button_minus = get_minus_neurons_button(layer_id)
    button_plus = get_plus_neurons_button(layer_id)
    button_freeze = get_freeze_neurons_button(layer_id)
    button_reset = get_reset_neurons_button(layer_id)
    # button_inspect = get_inspect_neurons_button(layer_id)
    return html.Div(
        dbc.Row(
            dbc.Col(
                [button_minus, button_plus, button_freeze, button_reset]
            )
        ),
        style={
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
        }
    )


def get_neuron_status(neuron_stats):
    adiff = abs(
        neuron_stats.train_trigger_rate - \
        neuron_stats.eval_trigger_rate)
    rdiff = 0
    if neuron_stats.train_trigger_rate > 0:
        rdiff = adiff / neuron_stats.train_trigger_rate

    if neuron_stats.learning_rate == 0.0:
        return NeuronStatus.FROZEND

    if neuron_stats.neuron_age <= 10:
        return NeuronStatus.NEUTRAL

    if neuron_stats.train_trigger_rate <= 0.01 or \
            neuron_stats.eval_trigger_rate <= 0.01:
        return NeuronStatus.IMPACT0

    if rdiff >= .50:
        return NeuronStatus.OVERFIT
    elif rdiff >= .10:
        return NeuronStatus.SUSPECT

    return NeuronStatus.HEALTHY


def get_neuron_status_color(neuron_status: NeuronStatus) -> str:
    if neuron_status == NeuronStatus.NEUTRAL:
        return "gray"
    elif neuron_status == NeuronStatus.OVERFIT:
        return "red"
    elif neuron_status == NeuronStatus.SUSPECT:
        return "orange"
    elif neuron_status == NeuronStatus.IMPACT0:
        return "black"
    elif neuron_status == NeuronStatus.FROZEND:
        return "blue"
    elif neuron_status == NeuronStatus.HEALTHY:
        return "green"
    else:
        return "PURPLE"


def format_large_int(n, max_length=5):
    suffix = ''
    if abs(n) >= 10**12:
        n, suffix = n / 10**12, 'T'
    elif abs(n) >= 10**9:
        n, suffix = n / 10**9, 'B'
    elif abs(n) >= 10**6:
        n, suffix = n / 10**6, 'M'
    elif abs(n) >= 10**3:
        n, suffix = n / 10**3, 'K'
    else:
        return int(n)

    formatted_number = f"{n:.1f}{suffix}"
    if len(formatted_number) > max_length:
        formatted_number = f"{int(n)}{suffix}"

    if len(formatted_number) > max_length:
        formatted_number = f"{n:.0f}{suffix}"

    return formatted_number


def format_value(value):
    if isinstance(value, float):
        return round(value, 2)
    return value


def get_neuron_stats_div(neuron_row, checklist_values):
    COL_MAX_LEN = 5
    row = []
    for checklist_value in checklist_values:
        value_str = ""
        if checklist_value == 'neuron_age':
            value_str = format_large_int(neuron_row['Age'])
        elif checklist_value == 'trigger_rate_train':
            value_str = "%05.4f" % neuron_row['RTrn']
        elif checklist_value == 'trigger_rate_eval':
            value_str = "%05.4f" % neuron_row['REval']
        elif checklist_value == 'abs_diff':
            value_str = "%05.4f" % neuron_row['ADiff']
        elif checklist_value == 'rel_diff':
            value_str = "%05.4f" % neuron_row['RDiff']

        if value_str:
            col = dbc.Col(value_str[:COL_MAX_LEN])
            row.append(col)

    for checklist_value in checklist_values:
        if checklist_value == 'frozen':
            freeze_button = daq.BooleanSwitch(
                id={
                    'type': 'neuron-frozen-switch',
                    'layer': neuron_row['layer_id'],
                    'neuron': neuron_row['neuron_id'],
                },
                on=neuron_row['Frozen'],
            )
            col = dbc.Col(freeze_button)
            row.append(col)
        if checklist_value == 'status':
            color = get_neuron_status_color(neuron_row['Status'])
            col = dbc.Col(
                neuron_row['Status'][:COL_MAX_LEN],
                style={"color": color, }
            )
            row.append(col)

    bkg_color = '#DDD' if not neuron_row['highlighted'] else "#FFD700",
    return dbc.Row(
        id=str((neuron_row['layer_id'], neuron_row['neuron_id'])),
        children=row,
        style={
            "color": "black",
            "align": "center",
            "fontWeight": "bold",
            "fontFamily": "Monospace",
            'backgroundColor': bkg_color,
            'margin': '3px',
            'borderRadius': '10px',
            'minWidth': "12vw",
            'padding': '2px',
            'fontSize': '12px',
        }
    )


def get_layer_headings(layer_row) -> html.Div:
    heading = layer_row.layer_type
    heading += f"[id={layer_row.layer_id}]"
    sub_heading = ""

    if layer_row.layer_type == "Conv2d":
        sub_heading += f" [{layer_row.incoming}->"
        sub_heading += f"{layer_row.kernel_size}x"
        sub_heading += f"{layer_row.kernel_size}->"
        sub_heading += f"{layer_row.outgoing}]"
    if layer_row.layer_type == "Linear":
        sub_heading += f" [{layer_row.incoming}->"
        sub_heading += f"{layer_row.outgoing}]"

    return heading, sub_heading


def get_neuron_query_input_div(ui_state: UIState):
    del ui_state
    cols = []

    cols.append(
        dbc.Col(
            dbc.Input(
                id='neuron-query-input', type='text',
                placeholder='Enter query predicate to select neurons.',
                style={'width': '18vw'}
            ),
        )
    )
    cols.append(
        dbc.Col(
            dbc.Input(
                id='neuron-query-input-weight', type='number',
                placeholder='weight',
                style={'width': '6vw'}
            ),
        )
    )
    cols.append(
        dbc.Col(
            dcc.Dropdown(
                id="neuron-action-dropdown",
                options=[
                    {"label": "Highlight", "value": "highlight"},
                    {"label": "Delete", "value": "delete"},
                    {"label": "Reinitialize", "value": "reinitialize"},
                    {"label": "Freeze", "value": "freeze"},
                    {"label": "Add Neurons", "value": "add_neurons"}
                ],
                value="highlight",  # Default value
                placeholder="Select an action",  # Placeholder text
                style={'width': '8vw'}  # Style to control dropdown width
            ),
        )
    )
    cols.append(
        dbc.Col(
            dbc.Button(
                "Run", id='run-neuron-data-query', color='primary',
                n_clicks=0,
                style={'width': '3vw'}
            ),
        )
    )

    neuron_quering_row = dbc.Row(
        cols,
        style={
            "display": 'flex',
            "justifyContent": 'center',
            "alignItems": 'center',
            "align": "center",
            "width": "45vw",
        }
    )
    return dbc.Col(
        neuron_quering_row,
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
        }
    )


def convert_checklist_to_df_head(checklist_values):
    heading = []
    for checklist_value in checklist_values:
        if checklist_value == 'neuron_id':
            heading.append("neuron_id")
        if checklist_value == 'neuron_age':
            heading.append("Age")
        if checklist_value == 'trigger_rate_train':
            heading.append("RTrn")
        if checklist_value == 'trigger_rate_eval':
            heading.append("REval")
        if checklist_value == 'abs_diff':
            heading.append("ADiff")
        if checklist_value == 'rel_diff':
            heading.append("RDiff")
        if checklist_value == 'weight_diff':
            heading.append("WDiff")
        if checklist_value == 'bias_diff':
            heading.append("BDiff")
    for checklist_value in checklist_values:
        if checklist_value == 'frozen':
            heading.append("Frzn")
        if checklist_value == 'status':
            heading.append("Status")

    return heading


def format_values_df(df):
    formated_df = pd.DataFrame(columns=df.columns)
    for col in df.columns:  # Skip "Neuron" column
        if col == "neuron_id" or col == "Age":
            formated_df[col] = df[col].apply(format_large_int).astype(str)
        else:
            formated_df[col] = df[col].apply(format_value)
    return formated_df


def layer_div_width(checklist_values):
    layer_width = LYR_BASE_WIDTH + len(checklist_values) * WIDTH_PER_COLUMN
    return layer_width


def get_layer_div(
        layer_row,
        layer_neurons_df,
        ui_state,
        checklist_values,
):
    heading, sub_heading = get_layer_headings(layer_row)
    layer_width = layer_div_width(checklist_values)

    checklist_values = convert_checklist_to_df_head(checklist_values)
    try:
        neurons_view_df = format_values_df(layer_neurons_df[checklist_values])
    except Exception as e:
        return no_update

    fetch_filters_toggle = html.Div(
        dbc.Checkbox(
            id={'type': 'layer-heatmap-checkbox', 'layer_id': int(layer_row['layer_id'])},
            value=False,
            className="form-check-input",
        ),
        style={'display': 'inline-block', 'marginLeft': '6px', 'verticalAlign': 'middle'},
        title="Fetch filter weights/heatmaps for this layer (throttled & limited)"
    )

    neuron_range_input = html.Div(
        dbc.Input(
            id={'type': 'layer-neuron-range', 'layer_id': int(layer_row['layer_id'])},
            type='text',
            placeholder='[0-10]',
            style={'width': '9ch', 'marginLeft': '8px'},
            debounce=True
        ),
        style={'display': 'inline-block', 'verticalAlign': 'middle'},
        title="Limit rendered neurons: e.g. 0:23 (inclusive)"
    )

    linear_input_box = None
    if getattr(layer_row, "layer_type", "") == "Linear":
        linear_input_box = html.Div(
            dbc.Input(
                id={'type': 'linear-incoming-shape', 'layer_id': int(layer_row['layer_id'])},
                type='text',
                placeholder='CxHxW (e.g. 3x5x5)',
                style={'width': '12ch'}
            ),
            id={'type': 'linear-shape-wrapper', 'layer_id': int(layer_row['layer_id'])},
            style={'display': 'none', 'marginTop': '4px'}
        )

    table_wrap = html.Div(
        id={'type': 'layer-table-wrap', 'layer_id': int(layer_row['layer_id'])},
        children=[
            dash_table.DataTable(
                id={"type": "layer-data-table", "layer_id": layer_row['layer_id']},
                data=neurons_view_df.to_dict("records"),
                columns=[{"name": col, "id": col} for col in neurons_view_df.columns],
                sort_action="native",
                row_selectable='single',
                style_table={
                    "maxHeight": "300px",
                    "overflowY": "scroll",
                    "width": "100%",
                    "minWidth": "100%",
                },
                style_cell={
                    "minWidth": "60px",
                    "width": "60px",
                    "maxWidth": f"{WIDTH_PER_COLUMN}px",
                },
                style_data={"whiteSpace": "normal", "height": "auto"},
                style_data_conditional=[
                    {"if": {"filter_query": "{Status} = 'OVRFT'"},
                    "color": "red", "fontWeight": "bold"},
                    {"if": {"filter_query": "{Status} = 'BAD'"},
                    "color": "orange"},
                    {"if": {"filter_query": "{Status} = 'GREAT'"},
                    "color": "green", "fontWeight": "bold"},
                    {"if": {"filter_query": "{Status} = 'DEAD'"},
                    "color": "black", "backgroundColor": "#e0e0e0"},
                    {"if": {"filter_query": "{Status} = 'FROZEN'"},
                    "color": "#B7C9E2"},
                    {"if": {"filter_query": "{Status} = 'N/A'"},
                    "color": "#a0a0a0", "fontStyle": "italic"},
                ],
            ),
        ],
        style={"flex": "1 1 50%", "minWidth": "50%"}
    )

    side_panel = html.Div(
        id={'type': 'layer-side-panel', 'layer_id': int(layer_row['layer_id'])},
        children=[
            html.Div(
                [
                    html.Span("", style={'flex': '1 1 auto'}),
                    html.Small("Fetch filters", style={'marginRight': '6px'}),
                    fetch_filters_toggle,
                    neuron_range_input
                ],
                style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end',
                    'gap': '6px', 'marginBottom': '4px'}
            ),
            linear_input_box,
            html.Div(
                id={'type': 'layer-activation', 'layer_id': int(layer_row['layer_id'])},
                children=[],
                style={'display': 'none', 'marginBottom': '8px'}
            ),
            html.Div(
                id={'type': 'layer-heatmap', 'layer_id': int(layer_row['layer_id'])},
                children=[],
                style={'display': 'none'}
            ),
        ],
        style={
            'display': 'none',
            'maxHeight': '300px',
            'overflowY': 'auto',
            'flex': '1 1 50%',
        }
    )

    return html.Div(
        id={"type": "layer-div", "layer_id": layer_row['layer_id']},
        children=[
            html.H3(heading, style={'textAlign': 'center'}),
            html.H4(
                id={"type": "layer-sub-heading", "layer_id": layer_row['layer_id']},
                children=sub_heading, style={'textAlign': 'center'}
            ),

            html.Div(
                children=[table_wrap, side_panel],
                style={'display': 'flex', 'alignItems': 'flex-start', 'gap': '8px'}
            ),

            get_layer_ops_buttons(layer_row['layer_id']),
        ],
        style={
            "minWidth": f"{layer_width}px",
            "padding": "10px",
            "margin": "5px",
            'border': '2px solid #666',
            'borderRadius': '15px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
        },
    )


def interactable_layers(
    ui_state: UIState,
    checklist_values : Tuple[str] = _DEFAULT_CHECKLIST_VALUES,
):
    children = []
    for _, layer_row in ui_state.layers_df.iterrows():
        try:
            layer_neurons_df = ui_state.get_neurons_df().loc[
                layer_row.layer_id].copy()
            layer_neurons_df = layer_neurons_df.reset_index()
            layer_neurons_df['layer_id'] = layer_row['layer_id']

            children.append(
                get_layer_div(
                    layer_row, layer_neurons_df, ui_state, checklist_values))
        except Exception:
            logger.exception(
                "layer_render_failed",
                extra={"layer_id": int(layer_row.get("layer_id", -1))}
            )
            continue

    return html.Div(
        id="layer-weights",
        children=children,
        style={
            'display': 'flex',
            'overflowX': 'scroll',
            "padding": "10px",
            "border": "1px solid #ccc",
        }
    )


def stats_display_checklist(ui_state: UIState):
    del ui_state
    checklist = dcc.Checklist(
        id='neuron_stats-checkboxes',
        options=[
            {'label': 'Neuron Id', 'value': 'neuron_id'},
            {'label': 'Neuron Age', 'value': 'neuron_age'},
            {'label': 'Train Set Trigger Rate', 'value': 'trigger_rate_train'},
            {'label': 'Eval Set Trigger Rate', 'value': 'trigger_rate_eval'},
            {'label': 'Absolute Difference between rates', 'value': 'abs_diff'},
            {'label': 'Relative Difference between rates', 'value': 'rel_diff'},
            {'label': 'Health State', 'value': 'status'},
            {'label': 'Show activation maps', 'value': 'show_activation_maps'},
            {'label': 'Show filter/weights', 'value': 'show_filter_heatmaps'},
        ],
        value=['neuron_id', 'neuron_age', 'trigger_rate_train', 'status'],
        inline=True,
        labelStyle={'marginRight': '10px'},
    )
    return dbc.Col(
        children=[checklist],
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
        }
    )


def get_data_query_input_div(ui_state: UIState):
    cols = []

    cols.append(
        dbc.Col(
            dbc.Input(
                id='train-data-query-input', type='text',
                placeholder='Enter train data query',
                style={'width': '18vw'}
            ),
        )
    )
    cols.append(
        dbc.Col(
            dbc.Input(
                id='data-query-input-weight', type='number',
                placeholder='weight',
                style={'width': '4vw'}
            ),
        )
    )
    cols.append(
        dbc.Col(
            dbc.Button(
                "Run", id='run-train-data-query', color='primary',
                n_clicks=0,
                style={'width': '3vw'}
            ),
        )
    )

    return dbc.Row(cols)


def get_weights_modal(ui_state: UIState):
    return dbc.Modal(
        [
            dbc.ModalHeader("Edit Weights"),
            dbc.ModalBody([
                dcc.Graph(id='weight-heatmap', config={'displayModeBar': False}),
            ]),
            dbc.ModalFooter(
                dbc.Button("Apply Changes", id="apply-weights", color="primary", className="ml-auto")
            ),
        ],
        id="modal-weights-edit",
        size="lg",  # Make the modal large
        is_open=False,  # Initially closed
    )

def zerofy_checklist():
    checklist = dcc.Checklist(
        id='zerofy-options-checklist',
        options=[
            {'label': 'zerofy with frozen', 'value': 'frozen'},
            {'label': 'zerofy with older',  'value': 'older'},
        ],
        value=[],
        inline=True,
        labelStyle={'marginRight': '10px'},
    )
    return dbc.Col(
        children=[checklist],
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
        }
    )

def activation_controls():
    return html.Div(
        id='activation-controls',
        children=dbc.InputGroup(
            [
                dbc.InputGroupText("Activations"),
                html.Div(
                    dcc.RadioItems(
                        id='activation-origin',
                        options=[
                            {'label': 'Eval',  'value': 'eval'},
                            {'label': 'Train', 'value': 'train'},
                        ],
                        value='eval',
                        inline=True,
                        inputStyle={'marginRight': '4px'},
                        labelStyle={'marginRight': '12px', 'marginBottom': '0'}
                    ),
                    style={'display': 'flex', 'alignItems': 'center', 'padding': '0 6px'}
                ),
                dbc.InputGroupText("Sample"),
                dbc.Input(
                    id='activation-sample-id',
                    type='number',
                    min=0,
                    value=0,
                    style={'width': '90px'}
                ),
                dbc.InputGroupText(
                    id='activation-sample-count',
                    children="ID range: (loading…)",
                    style={'minWidth': 'fit-content'}
                ),
            ],
            size="sm",
            className="justify-content-center d-flex align-items-center flex-nowrap gap-2"
        ),
        style={
            'display': 'none',
            'margin': '6px auto 2px',
            'maxWidth': '720px'
        }
    )

def get_weights_div(ui_state: UIState):
    return html.Div(
        id="model-architecture-div",
        children=[
            stats_display_checklist(ui_state),
            zerofy_checklist(),
            activation_controls(),
            interactable_layers(ui_state),
            get_neuron_query_input_div(ui_state),
            get_weights_modal(ui_state),
        ],
    )

def _downsample_strip(z_1xN: np.ndarray, max_len: int = 256) -> np.ndarray:
    N = z_1xN.shape[1]
    if N <= max_len:
        return z_1xN
    bucket = int(np.ceil(N / max_len))
    trim = (N // bucket) * bucket
    if trim == 0:
        return z_1xN
    v = z_1xN[0, :trim].reshape(-1, bucket).mean(axis=1)
    return v.reshape(1, -1)

def _parse_chw(s):
        if not s or not isinstance(s, str):
            return None
        import re
        m = re.match(r'^\s*(\d+)\s*[x×]\s*(\d+)\s*[x×]\s*(\d+)\s*$', s, re.I)
        C, H, W = map(int, m.groups())
        return (C, H, W)

def _renorm_to_unit(vals, eps=1e-8, robust=False, p=99.0):
    if robust:
        scale = np.nanpercentile(np.abs(vals), p)
    else:
        scale = np.nanmax(np.abs(vals))
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(vals)
    out = vals / scale
    return np.clip(out, -1.0, 1.0)

def _make_heatmap_figure(z, zmin=None, zmax=None):
    fig = go.Figure(
        data=[go.Heatmap(
            z=z, zmid=0, zmin=zmin, zmax=zmax,
            colorscale=[[0.0, 'red'], [0.5, 'white'], [1.0, 'green']],
            showscale=False
        )]
    )
    fig.update_yaxes(autorange='reversed')
    return fig.update_layout(
        margin=dict(l=2, r=2, t=2, b=2),
        xaxis_showgrid=False, yaxis_showgrid=False,
        xaxis_visible=False, yaxis_visible=False,
    )

def _parse_neuron_range(text):
    if not text or not isinstance(text, str):
        return None
    s = text.strip().strip('[](){}')
    if not s:
        return None
    parts = [p.strip() for p in s.split(',')] if ',' in s else [s]
    out = set()
    import re
    for p in parts:
        m = re.match(r'^(\d+)\s*[:-]\s*(\d+)$', p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b: a, b = b, a
            out.update(range(a, b + 1))
        elif re.match(r'^\d+$', p):
            out.add(int(p))
        else:
            return None
    return sorted(out) if out else None


def _make_out_ids(C_out: int, selected_neuron_ids):
    if selected_neuron_ids is None:
        return list(range(C_out))
    return [i for i in selected_neuron_ids if 0 <= i < C_out]

def get_plots_div():
    experiment_checkboxes = dcc.Checklist(
        id='experiment_checklist',
        options=[],
        value=[],
        labelStyle={
            'display': 'block'
        },
        style={
            'overflowY': 'auto',
            "margin": "8px",
        }
    )

    experiment_management = dbc.Col(
        id="experiment_management",
        children=[
            # experiment_smoothness,
            # experiment_smooth_sld,
            experiment_checkboxes,
        ],
        width=1,
    )

    experiment_plots_div = dbc.Col(
        id="experiment_plots_div",
        children=[
            #dcc.Graph(id='experiment-plot'),
        ],
        width=11,
    )

    return dbc.Row(
        id="plots-panel",
        children=[
            experiment_management,
            experiment_plots_div,
        ],
        style={
            'marginTop': '20px',  # Add top margin to create space
            'paddingTop': '10px'  # Add top padding for additional space
        }
    )


def sample_statistics_to_data_records(
        sample_statistics: pb2.SampleStatistics
):
    data_records = []
    for sample_id in range(sample_statistics.sample_count):
        data_record = {
            "SampleId": sample_id,
            "Label": sample_statistics.sample_label[sample_id],
            "Prediction": sample_statistics.sample_prediction[sample_id],
            "LastLoss": sample_statistics.sample_last_loss[sample_id],
            "Encounters": sample_statistics.sample_encounters[sample_id],
            "Discarded": sample_statistics.sample_discarded[sample_id]
        }
        data_records.append(data_record)
    return data_records

def _build_table_columns(df: pd.DataFrame) -> list[dict]:
    base = [c for c in _DISPLAY_COLUMNS if c in df.columns]
    extra = sorted([c for c in df.columns if c not in base and c not in {"Encounters", "Discarded"}])
    cols = []

    for col in base + extra:
        spec = {"name": col, "id": col}
        if col == "LastLoss":
            spec["type"] = "numeric"
            spec["format"] = Format(precision=2, scheme=Scheme.fixed)
        elif col.startswith('pred/'):  # Multi-task prediction columns
            spec["type"] = "numeric"
            spec["format"] = Format(precision=0, scheme=Scheme.fixed)
        else:
            dtype = df[col].dtype if col in df.columns else None
            if pd.api.types.is_float_dtype(dtype):
                spec["type"] = "numeric"
                spec["format"] = Format(precision=4, scheme=Scheme.fixed)
            elif pd.api.types.is_integer_dtype(dtype):
                spec["type"] = "numeric"
                spec["format"] = Format(precision=0, scheme=Scheme.fixed)
            else:
                spec["type"] = "any"
        cols.append(spec)
    return cols

def get_data_tab(ui_state: UIState):
    train_cols = _build_table_columns(ui_state.samples_df)
    eval_cols  = _build_table_columns(ui_state.eval_samples_df)
    grid_preset_dropdown = dcc.Dropdown(
        id='grid-preset-dropdown',
        options=[
            {'label': str(x * x), 'value': x * x} for x in [3, 4, 5, 6, 7, 10]
        ],
        value=9,
        clearable=False,
        style={'width': '6vw'}
    )

    eval_grid_dropdown = dcc.Dropdown(
        id='eval-grid-preset-dropdown',
        options=[{'label': str(x * x), 'value': x * x} for x in [3, 4, 5, 6, 7, 10]],
        value=9,
        clearable=False,
        style={'width': '6vw'}
    )

    train_table = dash_table.DataTable(
        id='train-data-table',
        data=ui_state.samples_df.to_dict('records'),
        columns=train_cols,
        sort_action="native",
        page_action="native",
        page_size=16,
        row_selectable='multi',
        style_data_conditional=[],
        row_deletable=True,
        editable=True,
        virtualization=True,
        style_table={
            'height': 'auto',
            'overflowY': 'auto',
            'width': 'auto',
            'margin': '2px',
            'padding': '2px'
        },
        style_cell={'textAlign': 'left', 'minWidth': '4vw', 'maxWidth': '4.5vw'}
    )

    eval_table = dash_table.DataTable(
        id='eval-data-table',
        data=ui_state.eval_samples_df.to_dict('records'),
        columns=eval_cols,
        sort_action="native",
        page_action="native",
        page_size=16,
        row_selectable='multi',
        style_data_conditional=[],
        row_deletable=True,
        editable=True,
        virtualization=True,
        style_table={
            'height': 'auto',
            'overflowY': 'auto',
            'width': 'auto',
            'margin': '2px',
            'padding': '2px'
        },
        style_cell={'textAlign': 'left', 'minWidth': '4vw', 'maxWidth': '4.5vw'}
    )

    train_controls = html.Div([
        dcc.Checklist(
            id='table-refresh-checkbox',
            options=[
                {'label': 'Refresh regularly', 'value': 'refresh_regularly'},
                {'label': 'Discard by flag flip', 'value': 'discard_by_flag_flip'}
            ],
            value=['refresh_regularly', 'discard_by_flag_flip'],
            inline=True,
            labelStyle={'marginRight': '5px'}
        ),
        dcc.Checklist(
            id='sample-inspect-checkboxes',
            options=[{'label': 'Inspect on click', 'value': 'inspect_sample_on_click'}],
            value=[], inline=True,
            labelStyle={'marginRight': '5px'}
        ),
        html.Div(grid_preset_dropdown, style={'marginLeft': '1vw'})
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1vw'})

    eval_controls = html.Div([
        dcc.Checklist(
            id='eval-table-refresh-checkbox',
            options=[
                {'label': 'Refresh regularly', 'value': 'refresh_regularly'},
                {'label': 'Discard by flag flip', 'value': 'discard_by_flag_flip'}
            ],
            value=['refresh_regularly', 'discard_by_flag_flip'],
            inline=True,
            labelStyle={'marginRight': '5px'}
        ),
        dcc.Checklist(
            id='eval-sample-inspect-checkboxes',
            options=[{'label': 'Inspect on click', 'value': 'inspect_sample_on_click'}],
            value=[], inline=True,
            labelStyle={'marginRight': '5px'}
        ),
        html.Div(eval_grid_dropdown, style={'marginLeft': '1vw'})
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1vw'})


    train_query_div = dbc.Row([
        dbc.Col(
            dbc.Input(
                id='train-data-query-input', type='text',
                placeholder='Enter train data query',
                style={'width': '18vw'}
            ),
        ),
        dbc.Col(
            dbc.Checklist(
                id='train-query-discard-toggle',
                options=[{'label': 'Un-discard', 'value': 'undiscard'}],
                value=[],
                inline=True
            ),
        ),
        dbc.Col(
            dbc.Checklist(
                id='train-denylist-accumulate-checkbox',
                options=[{'label': 'Accumulate', 'value': 'accumulate'}],
                value=[],
                inline=True,
                style={'marginLeft': '1vw'}
            ),
        ),
        dbc.Col(
            dbc.Input(
                id='data-query-input-weight', type='number',
                placeholder='weight',
                style={'width': '4vw'}
            ),
        ),
        dbc.Col(
            dbc.Button(
                "Run", id='run-train-data-query', color='primary',
                n_clicks=0,
                style={'width': '3vw'}
            ),
        ),
    ])

    eval_query_div = dbc.Row([
        dbc.Col(
            dbc.Input(
                id='eval-data-query-input', type='text',
                placeholder='Enter eval data query',
                style={'width': '18vw'}
            ),
        ),
        dbc.Col(
            dbc.Checklist(
                id='eval-query-discard-toggle',
                options=[{'label': 'Un-discard', 'value': 'undiscard'}],
                value=[],
                inline=True
            ),
        ),
        dbc.Col(
            dbc.Checklist(
                id='eval-denylist-accumulate-checkbox',
                options=[{'label': 'Accumulate', 'value': 'accumulate'}],
                value=[],
                inline=True,
                style={'marginLeft': '1vw'}
            ),
        ),
        dbc.Col(
            dbc.Input(
                id='eval-data-query-weight', type='number',
                placeholder='weight',
                style={'width': '4vw'}
            ),
        ),
        dbc.Col(
            dbc.Button(
                "Run", id='run-eval-data-query', color='primary',
                n_clicks=0,
                style={'width': '3vw'}
            ),
        ),
    ])

    tabs = dcc.Tabs(
        id='data-tabs',
        value='train',
        children=[
            dcc.Tab(label='Train Dataset', value='train', children=[
                html.Div([
                    html.H2("Train Dataset", id="train-dataset-header"),
                    train_controls,
                    html.Div([
                        html.Div([train_table], style={
                            'flex': '0 0 35vw',
                            'minWidth': '35vw',
                            'height': '100%'
                        }),
                        html.Div([
                            html.Div(id='train-sample-panel')
                        ], style={
                            'flex': '1',
                            'minWidth': '400px',
                            'height': 'auto',
                            'display': 'flex',
                            'overflow': 'auto',
                            'alignItems': 'flex-start',
                            'justifyContent': 'center'
                        })
                    ], style={
                        'display': 'flex',
                        'alignItems': 'stretch',
                        'gap': '1vw',
                        'width': '100%'
                    }),
                    train_query_div
                ])

            ]),
            dcc.Tab(label='Eval Dataset', value='eval', children=[
                html.Div([
                    html.H2("Eval Dataset", id="eval-dataset-header"),
                    eval_controls,
                    html.Div([
                        html.Div([eval_table], style={
                            'flex': '0 0 35vw',
                            'minWidth': '35vw',
                            'height': '100%'

                        }),
                        html.Div([
                            html.Div(id='eval-sample-panel')
                        ], style={
                            'flex': '1',
                            'minWidth': '400px',
                            'height': 'auto',
                            'overflow': 'auto',
                            'display': 'flex',
                            'alignItems': 'flex-start',
                            'justifyContent': 'center'
                        })
                    ], style={
                        'display': 'flex',
                        'alignItems': 'stretch',
                        'gap': '1vw',
                        'width': '100%'
                    }),
                    eval_query_div
                ])

            ])
        ]
    )

    return html.Div(tabs, style={
        'margin': '4vw', 'padding': '2vw',
        'borderRadius': '15px', 'border': '2px solid #666',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
        'width': '87vw'
    })

def label_below_img(img_component, last_loss, img_size):
    return html.Div([
        img_component,
        html.Div(
            f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -",
            style={'fontSize': '11px', 'lineHeight': '15px', 'textAlign': 'center', 'marginTop': '2px'}
        )
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': f'{img_size}px'})


def render_unified_triplet(sample, sample_row, task_type, is_selected, img_size, is_discarded, sid=None, last_loss=None):
    """Unified triplet display for both segmentation and reconstruction"""

    if task_type == "segmentation":
        left_label, left_border = "Input", "#888"
        middle_label, middle_border = "Target", "green"
        right_label, right_border = "Prediction", "blue"

        left_b64 = base64.b64encode(sample.raw_data).decode('utf-8')
        middle_b64 = base64.b64encode(sample.mask).decode('utf-8') if sample.mask else ""
        right_b64 = base64.b64encode(sample.prediction).decode('utf-8') if sample.prediction else ""

    else:  # reconstruction
        left_label, left_border = "Input", "#888"
        middle_label, middle_border = "Target", "green"
        right_label, right_border = "Reconstruction", "blue"

        left_b64 = base64.b64encode(sample.raw_data).decode('utf-8')
        middle_b64 = base64.b64encode(sample.raw_data).decode('utf-8')  # Target ≈ Input for reconstruction
        right_b64 = base64.b64encode(sample.prediction).decode('utf-8') if sample.prediction else ""

    def img_component(src_b64, label, border_color):
        if src_b64:
            return html.Div([
                html.Img(
                    src=f'data:image/png;base64,{src_b64}',
                    width=img_size,
                    height=img_size,
                    style={
                        'width': f'{img_size}px',
                        'height': f'{img_size}px',
                        'border': f'2px solid {border_color}',
                        'contentVisibility': 'auto'
                    }
                ),
                html.Div(label, style={'fontSize': 10, 'textAlign': 'center', 'marginTop': '2px'})
            ])
        else:
            return html.Div([
                html.Div("No data", style={
                    'width': f'{img_size}px', 'height': f'{img_size}px',
                    'border': f'2px dashed {border_color}', 'display': 'flex',
                    'alignItems': 'center', 'justifyContent': 'center',
                    'fontSize': '9px', 'color': '#666'
                }),
                html.Div(label, style={'fontSize': 10, 'textAlign': 'center', 'marginTop': '2px'})
            ])

    # Get classification info if available (for multi-task)
    cls_info = None
    if sample_row is not None:
        target = sample_row.get('Target', '?')
        prediction = sample_row.get('Prediction', '?')
        if isinstance(target, list) and len(target) > 0:
            target = target[0]
        if isinstance(prediction, list) and len(prediction) > 0:
            prediction = prediction[0]
        cls_info = f"Cls: T{target}/P{prediction}"

    return html.Div([
        # Metadata overlays
        html.Div(
            f"ID: {sid}", style={
                'position': 'absolute', 'top': '2px', 'left': '4px',
                'background': 'rgba(0,0,0,0.55)', 'color': 'white',
                'fontSize': '10px', 'padding': '1px 5px', 'borderRadius': '3px'
            }
        ) if sid is not None else None,
        html.Div(
            f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -",
            style={
                'position': 'absolute', 'top': '2px', 'right': '4px',
                'background': 'rgba(0,0,0,0.55)', 'color': 'white',
                'fontSize': '10px', 'padding': '1px 5px', 'borderRadius': '3px'
            }
        ) if last_loss is not None else None,

        # Classification info for multi-task
        html.Div(
            cls_info, style={
                'position': 'absolute', 'bottom': '2px', 'left': '4px',
                'background': 'rgba(255,255,255,0.9)', 'color': 'black',
                'fontSize': '9px', 'padding': '1px 4px', 'borderRadius': '3px'
            }
        ) if cls_info else None,

        # Task type badge
        html.Div(
            task_type, style={
                'position': 'absolute', 'bottom': '2px', 'right': '4px',
                'background': 'rgba(0,0,0,0.7)', 'color': 'white',
                'fontSize': '8px', 'padding': '1px 4px', 'borderRadius': '3px',
                'textTransform': 'uppercase'
            }
        ),

        # Images
        html.Div([
            img_component(left_b64, left_label, left_border),
            img_component(middle_b64, middle_label, middle_border),
            img_component(right_b64, right_label, right_border)
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '4px'})

    ], style={
        'position': 'relative',
        'marginBottom': '8px',
        'border': '4px solid red' if is_selected else 'none',
        'transition': 'border 0.3s, opacity 0.3s',
        'opacity': 0.25 if is_discarded else 1.0
    })

def render_images(ui_state: UIState, stub, sample_ids, origin,
                    discarded_ids=None, selected_ids=None):
    selected_ids = set(selected_ids or [])
    task_type = getattr(ui_state, "task_type", "classification")
    imgs = []
    num_images = len(sample_ids)
    cols = isqrt(num_images) or 1
    rows = cols
    img_size = int(512 / max(cols, rows))

    df = ui_state.samples_df if origin == "train" else ui_state.eval_samples_df
    id_to_loss = {int(r["SampleId"]): r.get("LastLoss", None) for _, r in df.iterrows()}
    discarded_ids = set(discarded_ids or set())

    def base_img_style(sid, is_discarded):
        return {
            'width': f'{img_size}px',
            'height': f'{img_size}px',
            'margin': '0.1vh',
            'border': '1px solid #ccc',
            'boxSizing': 'border-box',
            'objectFit': 'contain',
            'imageRendering': 'auto',
            'opacity': 0.25 if is_discarded else 1.0,
            'transition': 'box-shadow 0.06s, opacity 0.1s',
            'boxShadow': '0 0 0 3px rgba(255,45,85,0.95)' if sid in selected_ids else 'none',
            'contentVisibility': 'auto',
            'containIntrinsicSize': f'{img_size}px {img_size}px'
        }

    try:
        if task_type == "classification":
            for sid in sample_ids:
                sid = int(sid)
                is_discarded = sid in discarded_ids
                img_style = base_img_style(sid, is_discarded)
                url = f"/img/{origin}/{sid}?w={img_size}&h={img_size}&fmt=webp"

                img = html.Img(
                    id={'type': 'sample-img-el', 'origin': origin, 'sid': sid},
                    src=url,
                    width=img_size,
                    height=img_size,
                    style=img_style,
                    n_clicks=0
                )

                clickable = html.Div(
                    label_below_img(img, id_to_loss.get(sid, None), img_size),
                    id={'type': 'sample-img', 'origin': origin, 'sid': sid},
                    n_clicks=0,
                    style={'cursor': 'pointer'}
                )
                imgs.append(clickable)

        else:
            # Unified triplet for both segmentation and reconstruction
            batch_response = stub.GetSamples(pb2.BatchSampleRequest(
                sample_ids=sample_ids,
                origin=origin,
                resize_width=img_size,
                resize_height=img_size
            ))

            for sample in batch_response.samples:
                sid = int(sample.sample_id)
                is_discarded = sid in discarded_ids
                last_loss = id_to_loss.get(sid, None)

                df = ui_state.samples_df if origin == "train" else ui_state.eval_samples_df
                sample_row = df[df['SampleId'] == sid].iloc[0] if not df.empty else None

                task_type = "segmentation" if sample.mask else "reconstruction"

                triplet = render_unified_triplet(
                    sample, sample_row, task_type,
                    sid in selected_ids, img_size, is_discarded,
                    sid, last_loss
                )

                clickable = html.Div(
                    [triplet],
                    id={'type': 'sample-img', 'origin': origin, 'sid': sid},
                    n_clicks=0,
                    style={'cursor': 'pointer'}
                )
                imgs.append(clickable)

    except Exception:
        logger.exception("sample_render_failed", extra={"origin": origin})
        return no_update


    return html.Div(children=imgs, style={
        'display': 'grid',
        'gridTemplateColumns': f'repeat({cols}, 1fr)',
        'columnGap': '0.1vw',
        'rowGap': '0.1vh',
        'width': '100%',
        'height': 'auto',
        'maxWidth': 'calc(100vw - 40vw)',
        'boxSizing': 'border-box',
        'justifyItems': 'center',
        'alignItems': 'center',
        'paddingLeft': '0.01vw'
    })

def parse_sort_info(query):
    if not query or 'sortby' not in query.lower():
        return None
    match = re.search(r'sortby\s+([a-zA-Z0-9_, \s]+)', query, re.IGNORECASE)
    if not match:
        return None
    cols, dirs = [], []
    for part in match.group(1).split(','):
        tokens = part.strip().split()
        if not tokens:
            continue
        col = tokens[0]
        direction = tokens[1].lower() if len(tokens) > 1 and tokens[1].lower() in ['asc', 'desc'] else 'asc'
        cols.append(col)
        dirs.append(direction == 'asc')
    return {'cols': cols, 'dirs': dirs} if cols else None

def format_for_table(val, task_type):
    if val is None:
        return "-"
    if task_type == "segmentation":
        return str(val)
    else:
        if isinstance(val, list):
            try:
                return int(val[0])
            except Exception:
                return str(val)
        return str(val)


def rewrite_query_for_lists(query, task_type, ui_state: UIState):
    if task_type == "segmentation" and query:
        pattern = re.compile(r"(\d+)\s+in\s+(Target|Prediction)")
        matches = pattern.findall(query)
        if matches:
            def filter_fn(df):
                mask = None
                for val, col in matches:
                    this_mask = df[f"{col}ClassesStr"].str.split(",").apply(lambda x: any(xx == val for xx in x))
                    if mask is None:
                        mask = this_mask
                    else:
                        mask &= this_mask
                return df[mask]
            return filter_fn
    return None

def get_query_context(tab_type, ui_state: UIState):
    if tab_type == "train":
        df = ui_state.samples_df.copy()
        remove_op = "remove_from_denylist_operation"
        deny_op = "deny_samples_operation"
    else:
        df = ui_state.eval_samples_df.copy()
        remove_op = "remove_eval_from_denylist_operation"
        deny_op = "deny_eval_samples_operation"
    return df, remove_op, deny_op

def _rwg_rgb_from_signed(z: np.ndarray) -> np.ndarray:
    a = np.asarray(z, dtype=np.float32)
    if a.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    max_abs = float(np.max(np.abs(a)))
    if max_abs <= 1e-12:
        max_abs = 1.0
    t = np.clip(a / max_abs, -1.0, 1.0)

    r = np.empty_like(t)
    g = np.empty_like(t)
    b = np.empty_like(t)

    neg = t < 0
    r[neg] = 1.0
    g[neg] = 1.0 + t[neg]
    b[neg] = 1.0 + t[neg]

    pos = ~neg
    r[pos] = 1.0 - t[pos]
    g[pos] = 1.0
    b[pos] = 1.0 - t[pos]

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _png_data_uri_from_rgb(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _tile_img_component(z: np.ndarray) -> html.Img:
    z = np.asarray(z, dtype=np.float32)
    rgb = _rwg_rgb_from_signed(z)

    if z.ndim == 2 and z.shape[0] == 1:  # strip 1xN
        target_w = int(max(40, min(10 * z.shape[1], 600)))
        style = {
            'height': '16px',
            'width': f'{target_w}px',
            'imageRendering': 'pixelated',
        }
    else:
        style = {
            'height': '36px',
            'width': '36px',
            'imageRendering': 'pixelated',
        }

    uri = _png_data_uri_from_rgb(rgb)
    return html.Img(src=uri, draggable="false", style=style)

def _render_spatial_grid(resp):
    graphs = []
    is_spatial = ("Conv2d" in (resp.layer_type or "")) or ("BatchNorm2d" in (resp.layer_type or ""))
    if not is_spatial or resp.neurons_count == 0:
        return graphs
    for i in range(resp.neurons_count):
        amap = resp.activations[i]
        vals = np.array(amap.values, dtype=float).reshape(amap.H, amap.W)
        # renormalize to [-1, 1]
        vals = _renorm_to_unit(vals, robust=False)
        fig = _make_heatmap_figure(vals, zmin=-1.0, zmax=+1.0)
        graphs.append(
            html.Div(
                dcc.Graph(figure=fig, config={'displayModeBar': False},
                            style={'height': '40px', 'width': '40px'}),
                style={'display': 'inline-block'}
            )
        )
    return graphs

def get_ui_app_layout(ui_state: UIState) -> html.Div:
    layout_children = [
        dcc.Store(id='train-image-selected-ids', data=[]),
        dcc.Store(id='eval-image-selected-ids', data=[]),
        dcc.Interval(id='weights-render-freq', interval=1*1000, n_intervals=0),
        dcc.Interval(id='weights-fetch-freq', interval=1000, n_intervals=0),
        dcc.Interval(id='datatbl-render-freq', interval=10*1000, n_intervals=0),
        dcc.Interval(id='graphss-render-freq', interval=10*1000, n_intervals=0),
    ]
    layout_children.append(get_header_hyper_params_div(ui_state))
    layout_children.append(get_plots_div())
    layout_children.append(get_weights_div(ui_state))
    layout_children.append(get_data_tab(ui_state))

    return html.Div(children=layout_children)


def parse_args():
    parser = argparse.ArgumentParser(
        description="WeightsLAb Dash UI")

    parser.add_argument(
        "--root_directory",
        type=str,
        required=True,
        help="Path to the directory"
    )
    parser.add_argument(
        "--grpc_host",
        type=str,
        required=False,
        help="gRPC host address",
        default="localhost:50051"
    )
    parser.add_argument(
        "--ui_host",
        type=str,
        required=False,
        help="UI port",
        default="localhost:8050"
    )
    args = parser.parse_args()

    return args


def check_host_available(ui_host: str, timeout: float = 5.0) -> bool:
    """
    Check if the gRPC ui_host is available before attempting to connect.

    Args:
        ui_host: Host address in format 'hostname:port' or 'ip:port'
        timeout: Connection timeout in seconds

    Returns:
        True if ui_host is reachable, False otherwise
    """
    import socket

    try:
        # Parse ui_host and port
        if ':' in ui_host:
            hostname, port_str = ui_host.rsplit(':', 1)
            port = int(port_str)
        else:
            hostname = ui_host
            port = 50051  # Default gRPC port

        # Try to establish a TCP connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((hostname, port))
        sock.close()

        if result == 0:
            logger.info("host_available", extra={"ui_host": ui_host})
            return True
        else:
            logger.warning("host_unavailable", extra={"ui_host": ui_host, "error_code": result})
            return False

    except socket.gaierror as e:
        logger.error("host_resolution_failed", extra={"ui_host": ui_host, "error": str(e)})
        return False
    except ValueError as e:
        logger.error("invalid_host_format", extra={"ui_host": ui_host, "error": str(e)})
        return False
    except Exception as e:
        logger.error("host_check_failed", extra={"ui_host": ui_host, "error": str(e)})
        return False


def main(root_directory, ui_host: int = 8050, grpc_host: str = 'localhost:50051', **_):
    # Sanity checks
    # # Check if root directory exists
    if not os.path.isdir(root_directory):
        try:
            os.makedirs(root_directory, exist_ok=True)
        except Exception:
            logger.info("created_root_directory", extra={"root_directory": root_directory})
            sys.exit(1)

    channel = grpc.insecure_channel(
        grpc_host,
        options=[('grpc.max_receive_message_length', 32 * 1024 * 1024)]
    )
    stub = pb2_grpc.ExperimentServiceStub(channel)
    ui_state = UIState(root_directory=root_directory)
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.ZEPHYR],
        prevent_initial_callbacks='initial_duplicate')

    app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .config-card:hover {
                        border-color: #4267B2 !important;
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(66, 103, 178, 0.3);
                    }
                    .play-pause-btn:hover {
                        transform: scale(1.1);
                        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

    setup_logging()

    logging.getLogger("dash").disabled = True
    dash._utils.show_server = lambda *a, **k: None
    get_initial_state_request = pb2.TrainerCommand(
        get_hyper_parameters=True,
        get_interactive_layers=True,
        get_data_records="train",
    )
    logger.info("\nAbout Fetching initial state.")

    try:
        with ScopeTimer(tag="initial_state_fetch_and_update") as t:
            initial_state_response = stub.ExperimentCommand(
                get_initial_state_request)
            logger.info("Fetched initial state.", extra={"scope_timer": str(t)})
            ui_state.update_from_server_state(initial_state_response)
    except Exception as e:
        logger.error(f"Error fetching initial state: {e}")
        return

    # Generate app layout
    app.layout = get_ui_app_layout(ui_state)
    server = app.server

    _IMAGE_CACHE = {}

    @server.route("/img/<origin>/<int:sid>")
    def serve_img(origin, sid):
        try:
            w = int(request.args.get("w", "128"))
            h = int(request.args.get("h", "128"))
            fmt = request.args.get("fmt", "webp")
            if origin not in ("train","eval"): abort(404)

            key = (origin, sid, w, h, fmt)
            if key in _IMAGE_CACHE:
                data, mime, etag = _IMAGE_CACHE[key]
            else:
                batch = stub.GetSamples(pb2.BatchSampleRequest(
                    sample_ids=[sid], origin=origin, resize_width=w, resize_height=h
                ))
                if not batch.samples: abort(404)
                raw_png = batch.samples[0].raw_data or batch.samples[0].data

                im = Image.open(io.BytesIO(raw_png)).convert("RGB")
                buf = io.BytesIO()
                if fmt == "webp":
                    im.save(buf, format="WEBP", quality=78, method=4)
                    mime = "image/webp"
                else:
                    im.save(buf, format="JPEG", quality=80, optimize=True)
                    mime = "image/jpeg"
                data = buf.getvalue()

                etag = hashlib.md5(data).hexdigest()
                _IMAGE_CACHE[key] = (data, mime, etag)

            inm = request.headers.get("If-None-Match")
            if inm and inm == etag:
                return Response(status=304)

            resp = Response(data, mimetype=mime)
            resp.headers["Cache-Control"] = "public, no-cache"
            resp.headers["ETag"] = etag
            return resp
        except Exception as e:
            print("img route error:", e)
            abort(404)

    @server.route("/api/root-log-dir")
    def get_root_log_dir():
        """REST endpoint to get the root log directory.

        Used by Weights Studio UI to resolve image paths.
        Returns JSON with root_log_dir path.
        """
        try:
            # Get root_log_dir from the gRPC stub's servicer
            # The ExperimentServiceServicer has access to root_log_dir through its ExperimentService
            root_log_dir = stub.ExperimentCommand(pb2.TrainerCommand(
                get_hyper_parameters=False,
                get_interactive_layers=False,
            )).response  # This just gets command response, not what we want

            # Instead, try to get it from ui_state (root_directory was passed to main())
            root_log_dir = root_directory

            return jsonify({
                'success': True,
                'root_log_dir': root_log_dir
            })
        except Exception as e:
            logger.error(f"Error getting root_log_dir: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    def fetch_server_state_and_update_ui_state():
        while True:
            try:
                for dataset in ["train", "eval"]:
                    req = pb2.TrainerCommand(
                        get_hyper_parameters=(dataset == "train"),
                        get_interactive_layers=(dataset == "train"),
                        get_data_records=dataset
                    )
                    state = stub.ExperimentCommand(req)
                    ui_state.update_from_server_state(state)
            except Exception as e:
                print("Error updating UI state:", e)
            # Sleep to avoid hammering the gRPC server and causing GIL contention
            time.sleep(0.5)  # Update every 0.5 second - adjust as needed

    consistency_thread = threading.Thread(
        target=fetch_server_state_and_update_ui_state, name='fetch_server_state_and_update_ui_state', daemon=True)
    consistency_thread.start()

    def retrieve_training_statuses():
        nonlocal ui_state, stub
        for status in stub.StreamStatus(pb2.Empty()):
            ui_state.update_metrics_from_server(status)
    status_thread = threading.Thread(
        target=retrieve_training_statuses, name='retrieve_training_statuses', daemon=True)
    status_thread.start()

    @app.callback(
        Output('resume-pause-train-btn', 'children', allow_duplicate=True),
        Input({"type": "hyper-params-input", "idx": ALL}, "value"),
        Input('resume-pause-train-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def send_to_controller_hyper_parameters_on_change(
            hyper_param_values, resume_pause_clicks):
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        logger.info(
            "[UI] Hyper parameters on change",
            extra={
                "hyper_param_values": hyper_param_values,
                "resume_pause_clicks": resume_pause_clicks,
            }
        )

        button_children = no_update
        prop_id = ctx.triggered[0]['prop_id']
        hyper_parameter = pb2.HyperParameters()

        if "resume-pause-train-btn" in prop_id:
            is_training = resume_pause_clicks % 2
            hyper_parameter.is_training = is_training
            if is_training:
                button_children = "⏸"
                hyper_parameter.training_steps_to_do = hyper_param_values[5]
            else:
                button_children = "▶"
                hyper_parameter.training_steps_to_do = 0
        else:
            btn_dict = eval(prop_id.split('.')[0])
            hyper_parameter_id = btn_dict['idx']

            if hyper_parameter_id == "experiment_name":
                hyper_parameter.experiment_name = hyper_param_values[3]
            elif hyper_parameter_id == "training_left":
                hyper_parameter.training_steps_to_do = hyper_param_values[5]
            elif hyper_parameter_id == "learning_rate":
                hyper_parameter.learning_rate = hyper_param_values[4]
            elif hyper_parameter_id == "batch_size":
                hyper_parameter.batch_size = hyper_param_values[0]
            elif hyper_parameter_id == "eval_frequency":
                hyper_parameter.full_eval_frequency = hyper_param_values[2]
            elif hyper_parameter_id == "checkpooint_frequency":
                hyper_parameter.checkpont_frequency = hyper_param_values[1]

        request = pb2.TrainerCommand(
            hyper_parameter_change=pb2.HyperParameterCommand(
                hyper_parameters=hyper_parameter))
        stub.ExperimentCommand(request)
        return button_children

    # @app.callback(
    #     Output('layer-weights', 'children'),
    #     Input('weights-render-freq', 'n_intervals'),
    #     Input('neuron_stats-checkboxes', 'value'),
    #     # Input('refresh-weights-div-store', 'data'),
    #     State('layer-weights', 'children'),
    #     prevent_initial_call=True
    # )
    # def update_weights_div(_, checklist_values, children):
    #     print(f"[UI] WeightsLab.update_weights_div {checklist_values},")
    #     nonlocal ui_state
    #     nonlocal stub

    #     ctx = dash.callback_context
    #     if not ctx.triggered:
    #         trigger = 'No trigger'
    #     else:
    #         trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    #     print("[UI] WeightsLab.update_weights_div.trigger:", trigger)
    #     if 'weights' in trigger:
    #         print("[UI] WeightsLab.update_weights_div.refreshing.")
    #         if children:
    #             return children

    #         if not checklist_values:
    #             return []

    #     if 'neuron_stats-checkboxes' in trigger:
    #         print("[UI] WeightsLab.update_weights_div.checklist_values.")

    #         children = []
    #         for _, layer_row in ui_state.layers_df.iterrows():
    #             layer_neurons_df = ui_state.neurons_df.loc[layer_row.layer_id]
    #             layer_neurons_df = layer_neurons_df.reset_index()
    #             layer_neurons_df['layer_id'] = layer_row['layer_id']

    #             children.append(get_layer_div(
    #                 layer_row, layer_neurons_df, ui_state, checklist_values))
    #         return children
    #     return no_update

    @app.callback(
        Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'columns'),
        Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'data'),
        Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'style_data_conditional'),
        Output({'type': 'layer-div', 'layer_id': MATCH}, 'style'),
        Output({"type": "layer-sub-heading", "layer_id": MATCH}, 'children'),
        Input('weights-render-freq', 'n_intervals'),
        Input('neuron_stats-checkboxes', 'value'),
        State({'type': 'layer-data-table', 'layer_id': MATCH}, 'id'),
        State({'type': 'layer-data-table', 'layer_id': MATCH}, 'style_data_conditional'),
        State({'type': 'layer-div', 'layer_id': MATCH}, 'style'),
    )
    def update_layer_data_table(
            _, checklist_values, neuron_dt_div_id, style_data_conditional,
            layer_div_style
    ):
        # print(f"[UI] WeightsLab.update_layer_data_table.", neuron_dt_div_id)

        layer_id = neuron_dt_div_id['layer_id']
        if layer_id not in ui_state.get_neurons_df().index.get_level_values(0):
            # print('layer_id not updated:', layer_id)

            return no_update
        layer_neurons_df = ui_state.get_neurons_df().loc[layer_id].copy()
        layer_neurons_df = layer_neurons_df.reset_index()
        layer_neurons_df['layer_id'] = layer_id
        layer_row = ui_state.get_layer_df_row_by_id(layer_id)

        checklist_values = convert_checklist_to_df_head(checklist_values)
        neurons_view_df = format_values_df(layer_neurons_df[checklist_values])
        columns = [{"name": col, "id": col} for col in neurons_view_df.columns]

        highlight_conditional = []
        selected_ids = ui_state.selected_neurons[layer_row['layer_id']]

        if selected_ids:
            # Create a filter_query with multiple OR conditions
            filter_query = ' || '.join([
                f"{{neuron_id}} = '{id}'" for id in selected_ids])
            highlight_conditional = [
                {
                    "if": {"filter_query": filter_query},
                    "backgroundColor": "#ffefcc",
                }
            ]

        if len(style_data_conditional) > 6:
            style_data_conditional = style_data_conditional[-6:]
        new_cond_style = highlight_conditional + style_data_conditional

        # Handle layer_div children and style
        # We just update the second childe since it contain information about
        # the layer.
        _, sub_heading = get_layer_headings(layer_row)
        layer_width = layer_div_width(checklist_values)
        layer_div_style['minWidth'] = f"{layer_width}px"
        records = neurons_view_df.to_dict("records")

        return columns, records, new_cond_style, layer_div_style, sub_heading

    @app.callback(
        Input({"type": "layer-rem-btn", "layer_id": ALL}, "n_clicks"),
    )
    def on_layer_remove_neuron_callback(_,):
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        layer_details = ui_state.get_layer_df_row_by_id(layer_id)
        with ui_state.lock:
            layer_row_idx = ui_state.layer_id_to_df_row_idx[layer_id]
            ui_state.layers_df.loc[layer_row_idx]['outgoing'] -= 1

        weight_operation = pb2.WeightOperation(
            op_type=pb2.WeightOperationType.REMOVE_NEURONS
        )

        neuron_id = ui_state.selected_neurons[layer_id]
        if len(neuron_id) > 0:
            neuron_id = neuron_id[0]
        if neuron_id is None:
            neuron_id = layer_details.outgoing - 1
        removed_neuron_id = pb2.NeuronId(
            layer_id=layer_id,
            neuron_id=neuron_id
        )
        weight_operation.neuron_ids.extend([removed_neuron_id])

        request = pb2.WeightsOperationRequest(weight_operation=weight_operation)
        response = stub.ManipulateWeights(request)
        logger.info("Removed neuron", extra={
            "layer_id": int(layer_id),
            "neuron_id": int(neuron_id),
            "resp_message": getattr(response, "message", "")[:200]
        })

    @app.callback(
        Input({"type": "layer-add-btn", "layer_id": ALL}, "n_clicks"),
        State('zerofy-options-checklist', 'value')
    )
    def on_layer_add_neurons_callback(n_clicks, zerofy_opts):
        nonlocal ui_state
        if not ctx.triggered:
            return dash.no_update
        triggered = ctx.triggered_id
        if not triggered:
            return dash.no_update

        layer_id = triggered['layer_id']
        n_add = 1

        next_layer_id = _get_next_layer_id(ui_state, layer_id)
        old_incoming = _get_incoming_count(ui_state, next_layer_id) if next_layer_id is not None else None

        add_op = pb2.WeightOperation(
            op_type=pb2.WeightOperationType.ADD_NEURONS,
            layer_id=layer_id,
            neurons_to_add=n_add
        )
        stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=add_op))
        logger.info("Added neurons", extra={"layer_id": int(layer_id), "count": int(n_add)})

        state = stub.ExperimentCommand(pb2.TrainerCommand(
            get_interactive_layers=True,
            get_hyper_parameters=False,
        ))
        ui_state.update_from_server_state(state)

        if next_layer_id is None or old_incoming is None:
            return

        new_incoming = _get_incoming_count(ui_state, next_layer_id)
        if new_incoming <= old_incoming:
            return

        new_from_ids = list(range(old_incoming, new_incoming))

        selected_to_ids = ui_state.selected_neurons.get(next_layer_id, []) or []
        predicates = []
        if zerofy_opts:
            if 'frozen' in zerofy_opts:
                predicates.append(pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_FROZEN)
            if 'older' in zerofy_opts:
                predicates.append(pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_OLDER)


        if not selected_to_ids and not predicates:
            return

        zerofy_op = pb2.WeightOperation(
            op_type=pb2.WeightOperationType.ZEROFY,
            layer_id=next_layer_id,
            zerofy_from_incoming_ids=new_from_ids,
            zerofy_to_neuron_ids=selected_to_ids
        )
        if predicates:
            zerofy_op.zerofy_predicates.extend(predicates)

        resp = stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=zerofy_op))
        logger.info("Applied zerofy", extra={
            "to_layer_id": int(next_layer_id),
            "from_count": int(len(new_from_ids)),
            "to_selected_count": len(selected_to_ids),
            "predicates": [int(p) for p in (predicates or [])],
            "resp_message": getattr(resp, "message", "")[:200]
        })

        ui_state.update_from_server_state(
            stub.ExperimentCommand(pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers=True,
            ))
        )
        return

    @app.callback(
        Input({"type": "layer-freeze-btn", "layer_id": ALL}, "n_clicks"),
    )
    def on_layer_freeze_neuron_callback(_):
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        weight_operation=pb2.WeightOperation(
            op_type=pb2.WeightOperationType.FREEZE,
            layer_id=layer_id
        )
        neuron_id = ui_state.selected_neurons[layer_id]
        if len(neuron_id) > 0:
            neuron_id = neuron_id[0]
        if neuron_id is None:
            removed_neuron_id = pb2.NeuronId(
                layer_id=layer_id,
                neuron_id=neuron_id
            )
            weight_operation.neuron_ids.extend([removed_neuron_id])

        request = pb2.WeightsOperationRequest(
            weight_operation=weight_operation)
        response = stub.ManipulateWeights(request)
        logger.info("Froze layer", extra={
            "layer_id": int(layer_id),
            "neuron_id": int(neuron_id),
            "resp_message": getattr(response, "message", "")[:200]
        })

    @app.callback(
        Input({"type": "layer-reset-btn", "layer_id": ALL}, "n_clicks"),
    )
    def on_layer_reset_neuron_callback(_):
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        weight_operation=pb2.WeightOperation(
            op_type=pb2.WeightOperationType.REINITIALIZE,
            layer_id=layer_id
        )
        neuron_id = ui_state.selected_neurons[layer_id]
        if len(neuron_id) > 0:
            neuron_id = neuron_id[0]
        if neuron_id is None:
            removed_neuron_id = pb2.NeuronId(
                layer_id=layer_id,
                neuron_id=neuron_id
            )
            weight_operation.neuron_ids.extend([removed_neuron_id])

        request = pb2.WeightsOperationRequest(
            weight_operation=weight_operation)
        response = stub.ManipulateWeights(request)
        logger.info("Froze layer", extra={
            "layer_id": int(layer_id),
            "neuron_id": int(neuron_id),
            "resp_message": getattr(response, "message", "")[:200]
        })

    @app.callback(
        Output('modal-weights-edit', 'is_open'),
        Input({"type": "layer-see-btn", "layer_id": ALL}, "n_clicks"),
        prevent_initial_call=True
    )
    def inspect_neurons_weights_by_btn(n_clicks):
        if not any(click and click > 0 for click in n_clicks):
            return no_update

        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        logger.info("Inspecting weights of layer", extra={"layer_id": int(layer_id)})

        return True

    @app.callback(
        # State({"type": "layer-data-table", "layer_id": MATCH}, 'data'),
        Input({"type": "layer-data-table", "layer_id": ALL}, 'selected_rows'),
        prevent_initial_call=True  # Skip initial execution
    )
    def display_weights_of_neuron(selected_rows):
        nonlocal ui_state

        if not selected_rows:
            return no_update

        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        selected_row_index = selected_rows[layer_id]
        if selected_row_index is not None:
            ui_state.selected_neurons[layer_id] = selected_row_index  # Only one line selected ?
        logger.info("Selected neuron row indices", extra={"layer_id": int(layer_id), "indices": list(map(int, selected_row_index))})

        # row = data[selected_row_index]
        # print("Selected row: ", row)

    @app.callback(
        Input('run-neuron-data-query', 'n_clicks'),
        State('neuron-query-input', 'value'),
        State('neuron-query-input-weight', 'value'),
        State('neuron-action-dropdown', "value"),
        State('zerofy-options-checklist', 'value')
    )
    def run_query_on_neurons(_, query, weight, action, zerofy_opts):
        nonlocal ui_state
        if weight is None:
            weight = 1.0

        selected_neurons = collections.defaultdict(lambda: [])
        try:
            selected_neurons_df = ui_state.get_neurons_df().query(query)
        except Exception as e:
            logger.warning("Neuron query failed", extra={"query": str(query), "error": repr(e)})
            return

        sample_params = {}
        if weight <= 1.0:
            sample_params["frac"] = weight
        else:
            sample_params["n"] = weight

        selected_neurons_df = selected_neurons_df.sample(**sample_params)
        selected_neurons_df = selected_neurons_df.reset_index()

        if action == "highlight":
            for _, row in selected_neurons_df.iterrows():
                selected_neurons[row["layer_id"]].append(row["neuron_id"])
            ui_state.selected_neurons = selected_neurons
            logger.info("Selected neurons", extra={
                "query": str(query),
                "count": int(len(selected_neurons_df)),
            })
            return

        weight_operation = None
        if action == "delete":
            weight_operation=pb2.WeightOperation(
                op_type=pb2.WeightOperationType.REMOVE_NEURONS)
        elif action == "reinitialize":
            weight_operation=pb2.WeightOperation(
                op_type=pb2.WeightOperationType.REINITIALIZE)
        elif action == "freeze":
            weight_operation=pb2.WeightOperation(
                op_type=pb2.WeightOperationType.FREEZE)

        elif action == "add_neurons":
            selected_df = ui_state.get_layers_df().query(query)
            for _, row in selected_df.iterrows():
                layer_id = int(row['layer_id'])
                outgoing_neurons = int(row['outgoing'])

                if isinstance(weight, float) and 0 < weight < 1:
                    neurons_to_add = max(1, int(round(outgoing_neurons * weight)))
                elif isinstance(weight, int) and weight >= 1:
                    neurons_to_add = int(weight)
                else:
                    logger.warning("Invalid weight for add neurons", extra={"weight": weight})
                    continue

                next_layer_id = _get_next_layer_id(ui_state, layer_id)
                old_incoming = _get_incoming_count(ui_state, next_layer_id) if next_layer_id is not None else None

                add_op = pb2.WeightOperation(
                    op_type=pb2.WeightOperationType.ADD_NEURONS,
                    layer_id=layer_id,
                    neurons_to_add=neurons_to_add
                )
                stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=add_op))
                logger.info("Added neurons by query", extra={
                    "layer_id": int(layer_id),
                    "count": int(neurons_to_add)
                })

                if next_layer_id is None or old_incoming is None:
                    continue

                new_from_ids = list(range(old_incoming, old_incoming + neurons_to_add))

                selected_to_ids = ui_state.selected_neurons[next_layer_id] or []
                predicates = []
                if zerofy_opts:
                    if 'frozen' in zerofy_opts:
                        predicates.append(pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_FROZEN)
                    if 'older' in zerofy_opts:
                        predicates.append(pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_OLDER)

                if not selected_to_ids and not predicates:
                    print("[UI][query add] No ZEROFY targets (no selection & no predicates).")
                    continue

                zerofy_op = pb2.WeightOperation(
                    op_type=pb2.WeightOperationType.ZEROFY,
                    layer_id=next_layer_id,
                    zerofy_from_incoming_ids=new_from_ids,
                    zerofy_to_neuron_ids=selected_to_ids
                )
                if predicates:
                    zerofy_op.zerofy_predicates.extend(predicates)

                resp = stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=zerofy_op))
                logger.info("Applied zerofy by query", extra={
                    "to_layer_id": int(next_layer_id),
                    "from_count": int(len(new_from_ids)),
                    "to_selected_count": len(selected_to_ids),
                    "predicates": [int(p) for p in (predicates or [])],
                    "resp_message": getattr(resp, "message", "")[:200]
                })
                print(resp.message)

            ui_state.update_from_server_state(
                stub.ExperimentCommand(pb2.TrainerCommand(
                    get_hyper_parameters=True,
                    get_interactive_layers=True,
                ))
            )
            return

        if weight_operation:
            for idx, row in selected_neurons_df.reset_index().iterrows():
                logger.debug("Selected neuron row", extra={
                    "index": int(idx),
                    "layer_id": int(row['layer_id']),
                    "neuron_id": int(row['neuron_id'])
                })
                neuron_id = pb2.NeuronId(
                    layer_id=row['layer_id'],
                    neuron_id=row['neuron_id'])
                weight_operation.neuron_ids.extend([neuron_id])

        if weight_operation:
            request = pb2.WeightsOperationRequest(
                weight_operation=weight_operation)
            response = stub.ManipulateWeights(request)
            logger.info("Applied neuron operation", extra={
                "operation": pb2.WeightOperationType.Name(weight_operation.op_type),
                "count": int(len(weight_operation.neuron_ids)),
                "resp_message": getattr(response, "message", "")[:200]
            })

    @app.callback(
        Output({'type': 'layer-side-panel', 'layer_id': MATCH}, 'style'),
        Input('neuron_stats-checkboxes', 'value'),
        State({'type': 'layer-side-panel', 'layer_id': MATCH}, 'style'),
    )
    def toggle_side_panel(checklist_values, style):
        values = checklist_values or []
        show = ('show_activation_maps' in values) or ('show_filter_heatmaps' in values) or ('show_heatmaps' in values)
        style = dict(style or {})
        style['display'] = 'block' if show else 'none'
        style['maxHeight'] = '300px'
        style['overflowY'] = 'auto'
        return style

    @app.callback(
        Output('activation-controls', 'style'),
        Input('neuron_stats-checkboxes', 'value'),
        State('activation-controls', 'style'),
    )
    def toggle_activation_controls(values, style):
        style = dict(style or {})
        values = values or []
        style['display'] = 'block' if 'show_activation_maps' in values else 'none'
        style['whiteSpace'] = 'nowrap'
        style['overflow'] = 'hidden'
        return style

    @app.callback(
        Output({'type': 'linear-shape-wrapper', 'layer_id': MATCH}, 'style'),
        Input('neuron_stats-checkboxes', 'value'),
        State({'type': 'linear-shape-wrapper', 'layer_id': MATCH}, 'style'),
        prevent_initial_call=True
    )
    def toggle_linear_shape_input(values, style):
        style = dict(style or {})
        values = values or []
        if 'show_filter_heatmaps' in values:
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style


    @app.callback(
        Output({'type': 'layer-activation', 'layer_id': MATCH}, 'children'),
        Output({'type': 'layer-activation', 'layer_id': MATCH}, 'style'),
        # Input('weights-render-freq', 'n_intervals'),
        Input('neuron_stats-checkboxes', 'value'),
        Input('activation-sample-id', 'value'),
        Input('activation-origin', 'value'),
        State({'type': 'layer-activation', 'layer_id': MATCH}, 'id'),
        prevent_initial_call = True
    )
    def render_layer_activation(checklist_values, sample_value, origin_value, act_id):
        values = checklist_values or []
        if 'show_activation_maps' not in values:
            return dash.no_update, {'display': 'none'}

        layer_id = int(act_id['layer_id'])
        sample_id = int(sample_value) if sample_value is not None else 0
        origin = origin_value or "eval"

        resp_pre = stub.GetActivations(pb2.ActivationRequest(
            layer_id=layer_id, sample_id=sample_id, origin=origin
        ))

        is_conv = "Conv2d" in (getattr(resp_pre, "layer_type", "") or "")
        resp_post = None
        if is_conv:
            bn_layer_id = layer_id + 1
            resp_post = stub.GetActivations(pb2.ActivationRequest(
                layer_id=bn_layer_id, sample_id=sample_id, origin=origin
            ))

        pre_grid = _render_spatial_grid(resp_pre)
        post_grid = _render_spatial_grid(resp_post) if resp_post else []
        columns = []
        pre_label = "pre-BN" if is_conv else ""
        columns.append(
            html.Div([
                html.Small(pre_label, style={'opacity': 0.7}),
                html.Div(pre_grid, style={'display': 'grid', 'gap': '4px', 'paddingRight': '6px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'rowGap': '4px'})
        )
        if post_grid:
            columns.append(
                html.Div([
                    html.Small("post-BN", style={'opacity': 0.7}),
                    html.Div(post_grid, style={'display': 'grid', 'gap': '4px', 'paddingRight': '6px'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'rowGap': '4px'})
            )

        block = html.Div(
            columns,
            style={
                'display': 'grid',
                'gridTemplateColumns': f'repeat({len(columns)}, auto)',
                'columnGap': '12px',
                'borderTop': '1px solid #eee',
                'paddingTop': '6px'
            }
        )
        return [block], {'display': 'block'}


    @app.callback(
        Output('activation-sample-id', 'max'),
        Output('activation-sample-count', 'children'),
        Input('activation-origin', 'value'),
    )
    def update_sample_bounds(origin):
        try:
            resp = stub.ExperimentCommand(pb2.TrainerCommand(get_data_records=origin))
            n = int(resp.sample_statistics.sample_count or 0)
            max_id = max(n - 1, 0)
            return max_id, f"ID range: 0–{max_id} ({origin})"
        except Exception as e:
            return no_update, f"(couldn’t fetch sample count: {e})"

    @app.callback(
        Output({'type': 'layer-heatmap', 'layer_id': MATCH}, 'children'),
        Output({'type': 'layer-heatmap', 'layer_id': MATCH}, 'style'),
        Input('weights-fetch-freq', 'n_intervals'),
        Input('neuron_stats-checkboxes', 'value'),
        Input({'type': 'layer-heatmap-checkbox', 'layer_id': ALL}, 'value'),
        Input({'type': 'layer-neuron-range', 'layer_id': ALL}, 'n_submit'),
        State({'type': 'layer-heatmap', 'layer_id': MATCH}, 'id'),
        State({'type': 'linear-incoming-shape', 'layer_id': ALL}, 'id'),
        State({'type': 'linear-incoming-shape', 'layer_id': ALL}, 'value'),
        State({'type': 'layer-heatmap-checkbox', 'layer_id': ALL}, 'id'),
        State({'type': 'layer-neuron-range', 'layer_id': ALL}, 'id'),
        State({'type': 'layer-neuron-range', 'layer_id': ALL}, 'value'),
    )
    def render_layer_heatmap(
            _, checklist_values, checkbox_values, submit_counts, heatmap_id,
            all_linear_ids, all_linear_values, all_cb_ids, all_range_ids, all_range_vals
    ):
        values = checklist_values or []
        global_heatmap_enabled = ('show_filter_heatmaps' in values) or ('show_heatmaps' in values)
        if not global_heatmap_enabled:
            return no_update, {'display': 'none'}

        if not heatmap_id or 'layer_id' not in heatmap_id:
            return no_update, {'display': 'none'}

        layer_id = int(heatmap_id['layer_id'])

        is_layer_checked = False
        try:
            for cid, cval in zip(all_cb_ids or [], checkbox_values or []):
                lid = int(cid.get('layer_id')) if isinstance(cid, dict) else None
                if lid == layer_id:
                    is_layer_checked = bool(cval)
                    break
        except Exception:
            is_layer_checked = False
        if not is_layer_checked:
            return no_update, {'display': 'none'}

        linear_shape_text = None
        if all_linear_ids and all_linear_values:
            try:
                id_to_val = {
                    (i.get('layer_id') if isinstance(i, dict) else None): v
                    for i, v in zip(all_linear_ids, all_linear_values)
                }
                linear_shape_text = id_to_val.get(layer_id, None)
            except Exception:
                linear_shape_text = None

        neuron_range_text = None
        if all_range_ids and all_range_vals:
            try:
                id_to_val = {
                    (i.get('layer_id') if isinstance(i, dict) else None): v
                    for i, v in zip(all_range_ids, all_range_vals)
                }
                neuron_range_text = id_to_val.get(layer_id, None)
            except Exception:
                neuron_range_text = None
        selected_neuron_ids = _parse_neuron_range(neuron_range_text)

        resp = stub.GetWeights(pb2.WeightsRequest(
            neuron_id=pb2.NeuronId(layer_id=layer_id, neuron_id=-1)
        ))
        if not resp.success:
            msg = getattr(resp, "error_message", "Unknown error")
            block = html.Div([html.Small(msg)],
                                style={'borderTop': '1px solid #eee', 'paddingTop': '6px'})
            return [block], {'display': 'block'}

        layer_type = (resp.layer_type or "").strip()
        C_in, C_out = int(resp.incoming), int(resp.outgoing)
        w = np.array(resp.weights, dtype=np.float32)

        out_ids = _make_out_ids(C_out, selected_neuron_ids)

        tiles_by_neuron: list[list[np.ndarray]] = []

        if "Conv2d" in layer_type:
            K = int(resp.kernel_size or 0)
            expected = C_out * C_in * K * K
            if K <= 0 or w.size != expected:
                msg = (f"Unexpected weight shape: got {w.size}, expected {expected} "
                        f"(C_out={C_out}, C_in={C_in}, K={K})")
                block = html.Div([html.Small(msg)],
                                    style={'borderTop': '1px solid #eee', 'paddingTop': '6px'})
                return [block], {'display': 'block'}

            w = w.reshape(C_out, C_in, K, K)
            for out_id in out_ids:
                tiles_by_neuron.append([w[out_id, in_id] for in_id in range(C_in)])

        else:
            expected = C_out * C_in
            if w.size != expected:
                msg = (f"Unexpected weight shape: got {w.size}, expected {expected} "
                        f"(C_out={C_out}, C_in={C_in})")
                block = html.Div([html.Small(msg)],
                                    style={'borderTop': '1px solid #eee', 'paddingTop': '6px'})
                return [block], {'display': 'block'}

            w = w.reshape(C_out, C_in)  # (out, in)
            CHW = _parse_chw(linear_shape_text)
            can_reshape = CHW is not None and (CHW[0] * CHW[1] * CHW[2] == C_in)

            if can_reshape:
                C, H, W = CHW
                for out_id in out_ids:
                    vol = w[out_id, :].reshape(C, H, W)      # (C,H,W)
                    tiles_by_neuron.append([vol[c] for c in range(C)])  # one H×W per input channel
            else:
                for out_id in out_ids:
                    strip = _downsample_strip(w[out_id, :].reshape(1, C_in), max_len=256)
                    tiles_by_neuron.append([strip])

        rows = []
        for out_id, row_tiles in zip(out_ids, tiles_by_neuron):
            id_badge = html.Div(
                str(out_id),
                style={
                    'minWidth': '32px', 'textAlign': 'right', 'marginRight': '6px',
                    'fontFamily': 'monospace', 'fontSize': '12px', 'color': '#666'
                }
            )

            row_imgs = [
                html.Div(_tile_img_component(z),
                            style={'display': 'inline-block', 'marginRight': '4px'})
                for z in row_tiles
            ]
            tiles_scroller = html.Div(
                row_imgs,
                style={'whiteSpace': 'nowrap', 'overflowX': 'auto', 'marginBottom': '6px'}
            )

            rows.append(
                html.Div([id_badge, tiles_scroller],
                            style={'display': 'flex', 'alignItems': 'center'})
            )

        block = html.Div(rows, style={
            'borderTop': '1px solid #eee', 'paddingTop': '6px', 'paddingRight': '6px'
        })
        return [block], {'display': 'block'}


    @app.callback(
        Output('train-data-table', 'data'),
        Output('train-data-table', 'columns'),
        Output('train-dataset-header', 'children'),
        Input('datatbl-render-freq', 'n_intervals'),
        Input('run-train-data-query', 'n_clicks'),
        Input('train-image-selected-ids', 'data'),
        State('table-refresh-checkbox', 'value'),
        State('train-data-query-input', 'value'),
    )
    def update_train_data_table(_, __, train_selected_ids, chk, query):
        if 'refresh_regularly' not in chk:
            return no_update

        df = ui_state.samples_df.copy()
        if getattr(ui_state, "task_type") == "segmentation":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: format_for_table(v, "segmentation"))
        elif getattr(ui_state, "task_type") == "classification":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: format_for_table(v, "classification"))

        sort_info = parse_sort_info(query)
        if sort_info:
            try:
                df = df.sort_values(by=sort_info['cols'], ascending=sort_info['dirs'])
            except Exception:
                logger.warning("Train table sort failed", extra={"query": str(query)})


        num_available_samples = (~df["Discarded"]).sum()
        selected_count = len(train_selected_ids or [])
        return (
            df.to_dict('records'),
            _build_table_columns(df),
            f"Train Dataset #{num_available_samples} samples | {selected_count} selected"
        )



    @app.callback(
        Output('eval-data-table', 'data'),
        Output('eval-data-table', 'columns'),
        Output('eval-dataset-header', 'children'),
        Input('datatbl-render-freq', 'n_intervals'),
        Input('run-eval-data-query', 'n_clicks'),
        Input('eval-image-selected-ids', 'data'),
        State('eval-table-refresh-checkbox', 'value'),
        State('eval-data-query-input', 'value'),
    )
    def update_eval_data_table(_, __, eval_selected_ids, chk, query):
        if 'refresh_regularly' not in chk:
            return no_update

        df = ui_state.eval_samples_df.copy()
        if getattr(ui_state, "task_type", "classification") == "segmentation":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: format_for_table(v, "segmentation"))
        elif getattr(ui_state, "task_type", "classification") == "classification":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: format_for_table(v, "classification"))

        sort_info = parse_sort_info(query)
        if sort_info:
            try:
                df = df.sort_values(by=sort_info['cols'], ascending=sort_info['dirs'])
            except Exception:
                logger.warning("Eval table sort failed", extra={"query": str(query)})

        num_available_samples = (~df["Discarded"]).sum()
        selected_count = len(eval_selected_ids or [])
        return (
            df.to_dict('records'),
            _build_table_columns(df),
            f"Eval Dataset #{num_available_samples} samples | {selected_count} selected"
        )

    @app.callback(
        Output('train-sample-panel', 'children', allow_duplicate=True),
        Output('eval-sample-panel', 'children', allow_duplicate=True),
        Input('run-train-data-query', 'n_clicks'),
        Input('run-eval-data-query', 'n_clicks'),
        State('train-data-query-input', 'value'),
        State('eval-data-query-input', 'value'),
        State('data-query-input-weight', 'value'),
        State('eval-data-query-weight', 'value'),
        State('train-query-discard-toggle', 'value'),
        State('eval-query-discard-toggle', 'value'),
        State('train-denylist-accumulate-checkbox', 'value'),
        State('eval-denylist-accumulate-checkbox', 'value'),
        State('train-image-selected-ids', 'data'),
        State('eval-image-selected-ids', 'data'),
        State('train-data-table', 'derived_viewport_data'),
        State('eval-data-table', 'derived_viewport_data'),
        State('sample-inspect-checkboxes', 'value'),
        State('eval-sample-inspect-checkboxes', 'value'),
        State('data-tabs', 'value'),

        prevent_initial_call=True
    )
    def run_query_on_dataset(
        train_click, eval_click,
        train_query, eval_query,
        train_weight, eval_weight,
        train_toggle, eval_toggle,
        train_accumulate, eval_accumulate,
        train_selected_ids, eval_selected_ids,
        train_viewport, eval_viewport,
        train_flags, eval_flags, active_tab
    ):
        nonlocal ui_state, stub

        if not ctx.triggered:
            return no_update, no_update
        trig = ctx.triggered_id

        if trig == 'run-train-data-query':
            tab_type = 'train'
            query, weight = train_query, (train_weight or 1.0)
            toggle_values, accumulate_values = train_toggle, train_accumulate
            selected_ids_store = train_selected_ids or []
            viewport_rows = train_viewport or []
            inspect_on = 'inspect_sample_on_click' in (train_flags or [])
        else:
            tab_type = 'eval'
            query, weight = eval_query, (eval_weight or 1.0)
            toggle_values, accumulate_values = eval_toggle, eval_accumulate
            selected_ids_store = eval_selected_ids or []
            viewport_rows = eval_viewport or []
            inspect_on = 'inspect_sample_on_click' in (eval_flags or [])

        un_discard = 'undiscard' in (toggle_values or [])
        accumulate = 'accumulate' in (accumulate_values or [])

        if isinstance(query, str) and query.strip().lower() == 'selected':
            sample_ids = list(map(int, selected_ids_store or []))
            if not sample_ids:
                return no_update, no_update
            if isinstance(weight, float) and 0 < weight < 1 and len(sample_ids) > 1:
                k = max(1, int(round(len(sample_ids) * weight)))
                sample_ids = random.sample(sample_ids, k)
            elif isinstance(weight, int) and weight >= 1 and len(sample_ids) > weight:
                sample_ids = random.sample(sample_ids, weight)

        else:
            if not query or ('sortby' in query.lower()):
                return no_update, no_update

            df, remove_op_key, deny_op_key = get_query_context(tab_type, ui_state)
            task_type = getattr(ui_state, "task_type", "classification")
            if task_type == "classification":
                for col in ["Prediction", "Target"]:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda v: v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else v
                        )
            elif task_type == "segmentation":
                for col in ["Prediction", "Target"]:
                    if col in df.columns:
                        df[col + "ClassesStr"] = df[col].apply(lambda arr: ",".join([str(x) for x in arr]))
            filter_fn = rewrite_query_for_lists(query, task_type, ui_state)
            if filter_fn:
                query_dataframe = filter_fn(df)
            else:
                query_dataframe = df.query(query)

            if weight <= 1.0:
                query_dataframe = query_dataframe.sample(frac=weight)
            elif isinstance(weight, int):
                query_dataframe = query_dataframe.sample(n=weight)

            sample_ids = query_dataframe['SampleId'].to_list()

        deny_op = pb2.DenySamplesOperation()
        deny_op.sample_ids.extend(sample_ids)
        request = pb2.TrainerCommand()

        if un_discard:
            if tab_type == "train":
                request.remove_from_denylist_operation.CopyFrom(deny_op)
            else:
                request.remove_eval_from_denylist_operation.CopyFrom(deny_op)
        else:
            if tab_type == "train":
                request.deny_samples_operation.CopyFrom(deny_op)
                request.deny_samples_operation.accumulate = accumulate
            else:
                request.deny_eval_samples_operation.CopyFrom(deny_op)
                request.deny_eval_samples_operation.accumulate = accumulate
        resp = stub.ExperimentCommand(request)
        logger.info("Updated samples", extra={
            "tab": tab_type,
            "action": ("undiscard" if un_discard else "deny"),
            "count": int(len(sample_ids)),
            "accumulate": bool(accumulate),
            "resp_msg": getattr(resp, "message", "")[:200]
        })

        with ui_state.lock:
            if tab_type == "train":
                mask = ui_state.samples_df['SampleId'].isin(sample_ids)
                ui_state.samples_df.loc[mask, 'Discarded'] = (not un_discard)
                df_now = ui_state.samples_df
            else:
                mask = ui_state.eval_samples_df['SampleId'].isin(sample_ids)
                ui_state.eval_samples_df.loc[mask, 'Discarded'] = (not un_discard)
                df_now = ui_state.eval_samples_df

        if not inspect_on:
            return (no_update, no_update)

        visible_ids = [int(r['SampleId']) for r in (viewport_rows or [])]
        if not visible_ids:
            return (no_update, no_update)

        discarded_ids = set(df_now.loc[df_now['Discarded'], 'SampleId'].astype(int).tolist())

        selected_ids = train_selected_ids if tab_type == 'train' else eval_selected_ids
        selected_ids = selected_ids or []

        if tab_type == 'train' and active_tab == 'train':
            panel_train = render_images(ui_state, stub, visible_ids, origin='train',
                                        discarded_ids=discarded_ids,
                                        selected_ids=selected_ids)
            return panel_train, no_update
        elif tab_type == 'eval' and active_tab == 'eval':
            panel_eval = render_images(ui_state, stub, visible_ids, origin='eval',
                                    discarded_ids=discarded_ids,
                                    selected_ids=selected_ids)
            return no_update, panel_eval
        else:
            return no_update, no_update


    @app.callback(
        Output('train-data-div', 'style', allow_duplicate=True),
        Input('sample-inspect-checkboxes', 'value'),
        State('train-data-div', 'style'),
    )
    def update_train_data_div_style(inspect_checkboxes, old_div_style):
        width_percent = 45
        width_percent_delta = (90 - width_percent) // 2

        total_new_width_percent = width_percent + \
            len(inspect_checkboxes) * width_percent_delta
        style = dict(old_div_style)

        style.update({
            'width': f'{total_new_width_percent}vw',
            'maxWdith': f'{total_new_width_percent+2}vw',
        })

        return style

    @app.callback(
        Output('train-data-table', 'page_size'),
        Input('grid-preset-dropdown', 'value')
    )
    def update_page_size(grid_count):
        return grid_count

    @app.callback(
        Output('eval-data-table', 'page_size'),
        Input('eval-grid-preset-dropdown', 'value')
    )
    def update_eval_page_size(grid_count):
        return grid_count


    @app.callback(
        Output('train-sample-panel', 'children'),
        Output('eval-sample-panel', 'children'),
        Input('train-data-table', 'derived_viewport_data'),
        Input('eval-data-table', 'derived_viewport_data'),
        Input('sample-inspect-checkboxes', 'value'),
        Input('eval-sample-inspect-checkboxes', 'value'),
        Input('data-tabs', 'value'),
        State('train-image-selected-ids', 'data'),
        State('eval-image-selected-ids', 'data'),
        prevent_initial_call=True
    )
    def render_samples(
        train_viewport, eval_viewport, train_flags, eval_flags, tab,
        train_selected_ids, eval_selected_ids
    ):
        panels = [no_update, no_update]
        nonlocal ui_state, stub

        if tab == 'train' and 'inspect_sample_on_click' in (train_flags or []) and train_viewport:
            ids = [row['SampleId'] for row in train_viewport]
            discarded_ids = set(ui_state.samples_df.loc[ui_state.samples_df['Discarded'], 'SampleId'])
            panels[0] = render_images(ui_state, stub, ids, origin='train',
                                    discarded_ids=discarded_ids,
                                    selected_ids=(train_selected_ids or []))


        elif tab == 'eval' and 'inspect_sample_on_click' in (eval_flags or []) and eval_viewport:
            ids = [row['SampleId'] for row in eval_viewport]
            discarded_ids = set(ui_state.eval_samples_df.loc[ui_state.eval_samples_df['Discarded'], 'SampleId'])
            panels[1] = render_images(ui_state, stub, ids, origin='eval',
                                    discarded_ids=discarded_ids,
                                    selected_ids=(eval_selected_ids or []))
        return panels

    @app.callback(
        Output({'type': 'sample-img-el', 'origin': 'train', 'slot': ALL}, 'src'),
        Output({'type': 'sample-img-label', 'origin': 'train', 'slot': ALL}, 'children'),
        Input('train-data-table', 'derived_viewport_data'),
        State('grid-preset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_train_slots(viewport_rows, grid_count):
        n = grid_count
        urls = [""] * n
        labels = [""] * n
        if not viewport_rows:
            return urls, labels

        rows = viewport_rows[:n]
        img_size = int(512 / max(isqrt(n) or 1, isqrt(n) or 1))
        for i, row in enumerate(rows):
            sid = row['SampleId']
            last_loss = row.get('LastLoss', None)
            urls[i] = f"/img/train/{sid}?w={img_size}&h={img_size}&fmt=webp"
            labels[i] = f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -"
        return urls, labels

    @app.callback(
        Output({'type': 'sample-img-el', 'origin': 'eval', 'slot': ALL}, 'src'),
        Output({'type': 'sample-img-label', 'origin': 'eval', 'slot': ALL}, 'children'),
        Input('eval-data-table', 'derived_viewport_data'),
        State('eval-grid-preset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_eval_slots(viewport_rows, grid_count):
        n = grid_count
        urls = [""] * n
        labels = [""] * n
        if not viewport_rows:
            return urls, labels

        rows = viewport_rows[:n]
        img_size = int(512 / max(isqrt(n) or 1, isqrt(n) or 1))
        for i, row in enumerate(rows):
            sid = row['SampleId']
            last_loss = row.get('LastLoss', None)
            urls[i] = f"/img/eval/{sid}?w={img_size}&h={img_size}&fmt=webp"
            labels[i] = f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -"
        return urls, labels



    @app.callback(
        Output('train-image-selected-ids', 'data', allow_duplicate=True),
        Input({'type': 'sample-img', 'origin': 'train', 'sid': ALL}, 'n_clicks'),
        State('train-image-selected-ids', 'data'),
        prevent_initial_call=True
    )
    def on_train_image_click(n_clicks_list, current_ids):
        if not n_clicks_list or not any(n_clicks_list):
            return dash.no_update
        trig = ctx.triggered_id   # {'type':'sample-img','origin':'train','sid':...}
        if not trig or 'sid' not in trig:
            return dash.no_update
        sid = int(trig['sid'])
        current_ids = current_ids or []
        if sid in current_ids:
            return [x for x in current_ids if x != sid]
        return current_ids + [sid]


    @app.callback(
        Output('train-data-table', 'selected_rows', allow_duplicate=True),
        Input('train-image-selected-ids', 'data'),
        Input('train-data-table', 'data'),
        prevent_initial_call=True
    )
    def restore_train_selected_rows(selected_ids, data):
        if not data:
            return []
        selected_ids = selected_ids or []
        id2idx = {row['SampleId']: idx for idx, row in enumerate(data)}
        return [id2idx[sid] for sid in selected_ids if sid in id2idx]

    @app.callback(
        Output('eval-data-table', 'selected_rows', allow_duplicate=True),
        Input('eval-image-selected-ids', 'data'),
        Input('eval-data-table', 'data'),
        prevent_initial_call=True
    )
    def restore_eval_selected_rows(selected_ids, data):
        if not data:
            return []
        selected_ids = selected_ids or []
        id2idx = {row['SampleId']: idx for idx, row in enumerate(data)}
        return [id2idx[sid] for sid in selected_ids if sid in id2idx]

    @app.callback(
        Output('eval-image-selected-ids', 'data', allow_duplicate=True),
        Input({'type': 'sample-img', 'origin': 'eval', 'sid': ALL}, 'n_clicks'),
        State('eval-image-selected-ids', 'data'),
        prevent_initial_call=True
    )
    def on_eval_image_click(n_clicks_list, current_ids):
        if not n_clicks_list or not any(n_clicks_list):
            return dash.no_update
        trig = ctx.triggered_id
        if not trig or 'sid' not in trig:
            return dash.no_update
        sid = int(trig['sid'])
        current_ids = current_ids or []
        if sid in current_ids:
            return [x for x in current_ids if x != sid]
        return current_ids + [sid]

    @app.callback(
        Output('train-data-table', 'data', allow_duplicate=True),
        Input('train-data-table', 'data'),
        State('train-data-table', 'data_previous'),
        State('table-refresh-checkbox', 'value')
    )
    def denylist_deleted_rows_sample_ids(
        current_data, previous_data, table_checkboxes):
        if previous_data is None or len(previous_data) == 0:
            return no_update

        previous_sample_ids = set([row["SampleId"] for row in previous_data])
        current_sample_ids = set([row["SampleId"] for row in current_data])

        diff_sample_ids = previous_sample_ids - current_sample_ids

        for row in previous_data:
            if row["SampleId"] in diff_sample_ids:
                row['Discarded'] = True

        sample_deny_request = pb2.TrainerCommand(
            deny_samples_operation=pb2.DenySamplesOperation(
                sample_ids=list(diff_sample_ids))
        )
        sample_deny_response = stub.ExperimentCommand(sample_deny_request)
        del sample_deny_response

        if "discard_by_flag_flip" in table_checkboxes:
            return previous_data
        return current_data

    @app.callback(
        Output('train-image-selected-ids', 'data', allow_duplicate=True),
        Input('train-data-table', 'derived_virtual_selected_rows'),
        State('train-data-table', 'derived_virtual_data'),
        State('train-image-selected-ids', 'data'),
        prevent_initial_call=True
    )
    def sync_selection_from_table(sel_rows, vdata, prev_ids):
        if sel_rows is None or vdata is None:
            return no_update
        ids = [vdata[i]['SampleId'] for i in sel_rows if 0 <= i < len(vdata)]
        return ids

    @app.callback(
        Output('eval-image-selected-ids', 'data', allow_duplicate=True),
        Input('eval-data-table', 'derived_virtual_selected_rows'),
        State('eval-data-table', 'derived_virtual_data'),
        State('eval-image-selected-ids', 'data'),
        prevent_initial_call=True
    )
    def sync_eval_selection_from_table(sel_rows, vdata, prev_ids):
        if sel_rows is None or vdata is None:
            return no_update
        ids = [vdata[i]['SampleId'] for i in sel_rows if 0 <= i < len(vdata)]
        return ids



    @app.callback(
        Output({'type': 'sample-img-el', 'origin': 'train', 'sid': ALL}, 'style'),
        Input('train-image-selected-ids', 'data'),
        Input('train-sample-panel', 'children'),
        State({'type': 'sample-img-el', 'origin': 'train', 'sid': ALL}, 'id'),
        State({'type': 'sample-img-el', 'origin': 'train', 'sid': ALL}, 'style'),
        prevent_initial_call=True
    )
    def update_train_img_highlight(selected_ids, _children, ids, styles):
        sel = set(selected_ids or [])
        out = []
        for cid, st in zip(ids or [], styles or []):
            s = dict(st or {})
            try:
                sid = int(cid.get('sid'))
                s['boxShadow'] = '0 0 0 3px rgba(255,45,85,0.95)' if sid in sel else 'none'
                s.setdefault('transition', 'box-shadow 0.06s, opacity 0.1s')
            except Exception:
                pass
            out.append(s)
        return out


    @app.callback(
        Output({'type': 'sample-img-el', 'origin': 'eval', 'sid': ALL}, 'style'),
        Input('eval-image-selected-ids', 'data'),
        Input('eval-sample-panel', 'children'),
        State({'type': 'sample-img-el', 'origin': 'eval', 'sid': ALL}, 'id'),
        State({'type': 'sample-img-el', 'origin': 'eval', 'sid': ALL}, 'style'),
        prevent_initial_call=True
    )
    def update_eval_img_highlight(selected_ids, _children, ids, styles):
        sel = set(selected_ids or [])
        out = []
        for cid, st in zip(ids or [], styles or []):
            s = dict(st or {})
            try:
                sid = int(cid.get('sid'))
                s['boxShadow'] = '0 0 0 3px rgba(255,45,85,0.95)' if sid in sel else 'none'
                s.setdefault('transition', 'box-shadow 0.06s, opacity 0.1s')
            except Exception:
                pass
            out.append(s)
        return out


    @app.callback(
        Output('experiment_checklist', 'options', allow_duplicate=True),
        Output('experiment_checklist', 'value', allow_duplicate=True),
        Input('graphss-render-freq', 'n_intervals'),
    )
    def update_experiments_checklist(n_intervals):
        nonlocal ui_state

        experiment_names = list(ui_state.exp_names)
        options = [
            {'label': experiment_name, 'value': experiment_name}
            for experiment_name in experiment_names]
        return options, experiment_names

    @app.callback(
        Output("experiment_plots_div", "children"),
        Input("graphss-render-freq", "n_intervals"),
        State("experiment_plots_div", "children")
    )
    def add_graphs_to_div(_, existing_children):
        # print(f"UI.add_graphs_to_div")
        nonlocal ui_state

        graph_names = sorted(ui_state.met_names)

        if len(graph_names) == len(existing_children):
            return existing_children
        if len(graph_names) == 0:
            return no_update

        graph_divs = []
        for graph_name in graph_names:
            graph_divs.append(
                dcc.Graph(
                    id={"type": "graph", "index": graph_name},
                    config={"displayModeBar": False},
                )
            )
        return graph_divs

    @app.callback(
        Output({'type': "graph", "index": MATCH}, "figure",
               allow_duplicate=True),
        Input("graphss-render-freq", "n_intervals"),
        State({'type': "graph", "index": MATCH}, "id"),
        State('experiment_checklist', "value"),
        # State("plot-smoothness-slider", "value"),
        prevent_initial_call=True,
    )

    def update_graph(_, graph_id, checklist):
        # print("update_graph", graph_id, checklist)
        nonlocal ui_state

        metric_name = graph_id["index"]
        data = []

        for experiment_name in checklist:
            data.extend(ui_state.get_plots_for_exp_name_metric_name(
                metric_name, experiment_name)
            )
        if ui_state.plot_name_2_curr_head_point[metric_name] is not None:
            curr_point = ui_state.plot_name_2_curr_head_point[metric_name]
            data.append(
                go.Scattergl(
                    x=[curr_point.x],
                    y=[curr_point.y],
                    mode='markers',
                    name="Current Model",
                    marker_symbol="star-diamond-open-dot",
                    marker=dict(color='red', size=16)
                )
            )

        select_graph = go.Scattergl(
            x=[None],
            y=[None],
            mode='markers',
            name="",
            marker_symbol="diamond",
            marker=dict(color='cyan', size=16, opacity=0.8)
        )

        figure = {
            'data': data + [select_graph],
            'layout': go.Layout(
                title=metric_name,
                xaxis={'title': 'Seen Samples'},
                yaxis={'title': "Value"},
            )
        }
        return figure

    @app.callback(
        Output({'type': "graph", "index": MATCH}, "figure", allow_duplicate=True),
        Output({'type': "graph", "index": MATCH}, "ClickData"),
        [
            Input({'type': "graph", "index": MATCH}, 'hoverData'),
            Input({'type': "graph", "index": MATCH}, 'clickData'),
        ],
        State({'type': "graph", "index": MATCH}, "figure"),
        prevent_initial_call=True
    )

    def update_selection_of_checkpoint(hoverData, clickData, figure):
        nonlocal stub
        nonlocal ui_state
        # print("update_selection_of_checkpoint", hoverData, clickData, figure)

        ctx = dash.callback_context
        if not ctx.triggered:
            return figure, no_update

        if hoverData is None or 'points' not in hoverData:
            return no_update

        cursor_x = hoverData['points'][0]['x']
        cursor_y = hoverData['points'][0]['y']

        x_min, y_min, t_min, i_min, min_dist = None, None, None, None, 1e10

        if 'data' not in figure:
            return no_update

        for t_idx, trace_data in enumerate(figure['data']):
            if "ckpt" not in trace_data['name']:
                continue
            x_data = np.array(trace_data['x'])
            y_data = np.array(trace_data['y'])

            for i, val in enumerate(x_data):
                x_data[i] = 0 if val is None else val

            for i, val in enumerate(y_data):
                y_data[i] = 0 if val is None else val

            if len(y_data) < len(x_data):
                x_data = x_data[:-1]
            elif len(x_data) < len(y_data):
                y_data = y_data[:-1]

            if x_data is None or y_data is None or x_data.size == 0 or \
                    y_data.size == 0 or cursor_x is None or cursor_y is None:
                continue

            # replace None in x_data and y_data with 0
            x_data = np.nan_to_num(x_data)
            try:
                distances = np.sqrt(
                    (x_data - cursor_x) ** 2 + (y_data - cursor_y) ** 2)
                min_index = np.argmin(distances)  # Index of the closest point
                if distances[min_index] < min_dist:
                    x_min, y_min, t_min, i_min, min_dist = (
                        x_data[min_index], y_data[min_index], t_idx, min_index,
                        distances[min_index])
            except Exception as e:
                logger.warning("Error while updating checkpoint selection", extra={"error": repr(e)})
                continue

        checkpoint_id_to_load = None
        if t_min is not None and i_min is not None:
            figure['data'][-1]['x'] = [x_min]
            figure['data'][-1]['y'] = [y_min]

            if i_min < len(figure['data'][t_min]["customdata"]):
                checkpoint_id_to_load = \
                    figure['data'][t_min]["customdata"][i_min]

        trigger = ctx.triggered[0]["prop_id"]
        if "clickData" in trigger and clickData:
            load_checkpoint_op = pb2.LoadCheckpointOperation(
                checkpoint_id=checkpoint_id_to_load)
            load_checkpoint_request = pb2.TrainerCommand(
                load_checkpoint_operation=load_checkpoint_op)
            ckpt_load_result = stub.ExperimentCommand(
                load_checkpoint_request)

            logger.info("Loaded checkpoint", extra={
                "checkpoint_id": str(checkpoint_id_to_load),
                "ok": getattr(ckpt_load_result, "success", None),
                "resp_msg": getattr(ckpt_load_result, "message", "")[:200]
            })
            ui_state.current_run_id += 1

            update_request = pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers=True
            )
            updated_state = stub.ExperimentCommand(update_request)
            ui_state.update_from_server_state(updated_state)

            if checkpoint_id_to_load is not None:
                logger.debug("Figure data for loaded checkpoint", extra={"trace_index": int(t_min)})

            return figure, None

        return figure, no_update

    # Run the Dash app
    host_addr, host_port = ui_host.split(":")
    app.run(debug=False, host=host_addr, port=int(host_port), use_reloader=False)


def ui_serve(root_directory: str = None, ui_host: str = "localhost", ui_port: int = 8050, grpc_host: str = 'localhost', grpc_port: int = 50051, **_):
    """Launch the UI in a separate subprocess to avoid GIL contention."""
    import subprocess
    import sys

    ui_host = os.environ.get("WEIGHTSLAB_UI_HOST", ui_host)
    ui_port = int(os.environ.get("WEIGHTSLAB_UI_PORT", ui_port))
    grpc_host = os.environ.get("GRPC_BACKEND_HOST", grpc_host)
    grpc_port = int(os.environ.get("GRPC_BACKEND_PORT", grpc_port))

    # Build command to run this file as a subprocess
    cmd = [
        sys.executable,  # Use the same Python interpreter
        __file__,        # This file (weightslab_ui.py)
        "--root_directory", str(root_directory),
        "--ui_host", f'{ui_host}:{ui_port}',
        "--grpc_host", f'{grpc_host}:{grpc_port}'
    ]

    logger.info("ui_subprocess_starting", extra={
        "command": " ".join(cmd),
        "ui_host": ui_host,
        "ui_port": ui_port,
        "grpc_host": grpc_host,
        "grpc_port": grpc_port,
        "root_directory": root_directory
    })

    # Launch UI as subprocess - output goes to parent console
    ui_process = subprocess.Popen(
        cmd,
        stdout=None,  # Inherit parent's stdout (console)
        stderr=None,  # Inherit parent's stderr (console)
        stdin=subprocess.DEVNULL
    )

    logger.info("ui_subprocess_started", extra={
        "pid": ui_process.pid,
        "ui_host": ui_host,
        "ui_port": ui_port,
        "grpc_host": grpc_host,
        "grpc_port": grpc_port,
        "root_directory": root_directory
    })

    return ui_process


if __name__ == '__main__':
    args = parse_args()
    main(root_directory=args.root_directory, ui_host=args.ui_host, grpc_host=args.grpc_host)