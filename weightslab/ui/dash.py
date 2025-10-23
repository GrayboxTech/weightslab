"""Class that interface scalar and annotation reporting"""

import pandas as pd
import os


class Dash:
    def __init__(self, root_directory: str) -> None:
        self.root_directory = root_directory

        self.graph_n_lines = pd.DataFrame(
            columns=[
                "step", "graph_name", "line_name", "line_value"]
        )
        self.lines_2_annot = pd.DataFrame(
            columns=[
                "step", "graph_name", "line_name", "annotation",
                "line_value", "metadata"]
        )
        self.hidden_line_names = []

        self._load()

    def _dump(self):
        if not os.path.exists(self.root_directory):
            os.mkdir(self.root_directory)
        self.graph_n_lines.to_pickle(
            os.path.join(self.root_directory, 'graph_n_lines.pkl'))
        self.lines_2_annot.to_pickle(
            os.path.join(self.root_directory, 'lines_2_annot.pkl'))

    def _load(self):
        if not os.path.exists(self.root_directory):
            print("Dash._load ", self.root_directory, " does not exist")
            return
        graph_n_lines_path = os.path.join(
            self.root_directory, 'graph_n_lines.pkl')
        lines_2_annot_path = os.path.join(
            self.root_directory, 'lines_2_annot.pkl')
        if not os.path.exists(graph_n_lines_path) or \
                not os.path.exists(lines_2_annot_path):
            print(
                f"Dash._load {graph_n_lines_path} or {lines_2_annot_path} "
                f"does not exist.")
            return

        self.graph_n_lines = pd.read_pickle(graph_n_lines_path)
        self.lines_2_annot = pd.read_pickle(lines_2_annot_path)

    def get_graph_names(self):
        """Returns a list of graph names that have been reported."""
        return self.graph_n_lines["graph_name"].unique()

    def get_line_names(self):
        """Returns a list of experiment(line names) that have been reported."""
        line_names = self.graph_n_lines["line_name"].unique()

        return [line_name for line_name in line_names if line_name not in self.hidden_line_names]

    def _get_value_closest_to_step(self, graph_name, line_name, step):
        closest_match_value = None
        line_and_graph_df = self.graph_n_lines[
            (self.graph_n_lines["graph_name"] == graph_name) &
            (self.graph_n_lines["line_name"] == line_name)
        ]
        try:
            closest_match = \
                (line_and_graph_df["step"] - step).abs().argsort().iloc[0]

            closest_match_value = \
                line_and_graph_df.iloc[closest_match]["line_value"]
        except IndexError:
            pass

        return closest_match_value

    def add_scalars(self, graph_name, name_2_value, global_step: int):
        """"Add a scalar value to the dashboard.
        Args:
            graph_name: The name of the graph to which the scalar belongs.
            name_2_value: A dictionary with the name of the scalar as key and
                the value as value.
            global_step: The global step of the experiment.
        """
        data_frame_lines = len(self.graph_n_lines)

        for line_name, line_value in name_2_value.items():
            data_frame_row = [global_step, graph_name]
            data_frame_row.append(line_name)
            data_frame_row.append(line_value)
            self.graph_n_lines.loc[data_frame_lines] = data_frame_row

    def add_annotations(
            self, graph_names, line_name, annotation, global_step,
            metadata=None):
        """Add an annotation to the dashboard. The annotation is a mark on the
        plot line signaling that an event happened such as for instance, a
        checkpoint has been saved.
        Args:
            graph_names: The names of the graphs to which the annotation
                belongs.
            line_name: The name of the line to which the annotation belongs.
            annotation: The annotation message.
            global_step: The global step of the experiment.
            metadata: Additional information to be stored with the annotation. 
        """

        if len(graph_names) == 0:
            return

        for graph_name in graph_names:
            annotation_line_value = self._get_value_closest_to_step(
                graph_name, line_name, global_step
            )

            self.lines_2_annot.loc[len(self.lines_2_annot)] = [
                global_step, graph_name, line_name, annotation,
                annotation_line_value, metadata]
        self._dump()

    def hide_line(self, line_name: str):
        self.hidden_line_names.append(line_name)

    def remove_line(self, line_name: str):
        if line_name is None:
            return
        self.graph_n_lines = self.graph_n_lines[
            self.graph_n_lines["line_name"] != line_name]
        self.lines_2_annot = self.lines_2_annot[
            self.lines_2_annot["line_name"] != line_name]
        self._dump()

    def get_experiment_name_for(self, checkpoint_id: int):
        if checkpoint_id is None:
            return
        checkpoint_line = self.lines_2_annot[
            self.lines_2_annot.metadata == checkpoint_id]
        line_name = checkpoint_line.iloc[0].line_name
        return line_name

    def remove_data_after_checkpoint(self, checkpoint_id: int):
        if checkpoint_id is None:
            return
        checkpoint_line = self.lines_2_annot[
            self.lines_2_annot.metadata == checkpoint_id]
        self.graph_n_lines = self.graph_n_lines[
            self.graph_n_lines["step"] < checkpoint_line.iloc[0].step]
        self.lines_2_annot = self.lines_2_annot[
            self.lines_2_annot["step"] < checkpoint_line.iloc[0].step]

        self._dump()
