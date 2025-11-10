import queue


class Dash:
    def __init__(self, root_directory: str) -> None:
        self.queue = queue.Queue()
        self.graph_names = set()

    def get_graph_names(self):
        return list(self.graph_names)

    def add_scalars(self, graph_name, name_2_value, global_step: int):
        self.graph_names.add(graph_name)
        for line_name, line_value in name_2_value.items():
            self.queue.put({
                "experiment_name": line_name,
                "model_age": global_step,
                "metric_name": graph_name,
                "metric_value": float(line_value),
            })

    def add_annotations(
            self, graph_names, line_name, annotation, global_step,
            metadata=None):

        self.queue.put({
            "experiment_name": line_name,
            "model_age": global_step,
            "annotation": annotation,
            "metadata": metadata,
        })
