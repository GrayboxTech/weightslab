import subprocess
import unittest
from unittest.mock import patch

import graphviz
import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from weightslab.utils.plot_graph import plot_fx_graph_with_details


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestPlotGraphRenderUnit(unittest.TestCase):
    def test_plot_fx_graph_calls_render(self):
        traced = symbolic_trace(_TinyNet())
        traced.name = "tiny"

        with patch("weightslab.utils.plot_graph.graphviz.Digraph.render") as render_mock:
            plot_fx_graph_with_details(traced, custom_dependencies=[], filename="dummy_plot")

        render_mock.assert_called_once()

    def test_plot_fx_graph_handles_render_error(self):
        traced = symbolic_trace(_TinyNet())
        traced.name = "tiny"

        called_err = graphviz.backend.execute.CalledProcessError(returncode=1, cmd="dot")

        with patch("weightslab.utils.plot_graph.graphviz.Digraph.render", side_effect=called_err), \
             patch("weightslab.utils.plot_graph.logger.error") as log_error_mock:
            plot_fx_graph_with_details(traced, custom_dependencies=[], filename="dummy_plot")

        log_error_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
