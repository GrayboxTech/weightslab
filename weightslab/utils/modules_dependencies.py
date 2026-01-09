import collections
import enum
import logging

import torch.nn as nn
from typing import List, Set

logger = logging.getLogger(__name__)


class DepType(str, enum.Enum):
    """E.g: layer1.prune triggers layer2.prune_incoming."""
    INCOMING = "INCOMING"
    """E.g: layer1.insert triggers layer2.insert."""
    SAME = "SAME"
    NONE = "NONE"
    REC = "REC"


class _ModulesDependencyManager():
    """
    Instead of awkwardly checking indexes in order to update dependent
    layer, we are keeping them into a dictionary in order to quickly
    look them up.
    """

    def __init__(self) -> None:
        # layer_id -> layer_ref
        self.id_2_layer = collections.defaultdict(lambda: None)
        # what kind of dependency is there
        self.dependency_2_id_2_id = collections.defaultdict(
            lambda: collections.defaultdict(lambda: []))
        #
        self.reversed_dep_type = {k: False for k in DepType}

    def __str__(self):
        return \
            "ModulesDependencyManager: " + \
            f"{self.id_2_layer} {self.dependency_2_id_2_id}"

    def register_module(self, id: int, module: nn.Module):
        """Register the model submodules.

        Args:
            id (int): The id of the module.
            module (nn.Module): The module associated with the id.
        """
        self.id_2_layer[id] = module

    def _register_dependency(
            self, id1: int, id2: int,
            dep_type: DepType = DepType.NONE):
        self.dependency_2_id_2_id[dep_type][id1].append(id2)

    def register_rec_dependency(self, id1, id2):
        """Marks the dependency between two modules with id1 and id2 as REC,
        in the sense that id1.operation1 triggers id2.operation1. Useful for
        when after a Linear there is a BatchNorm so adding neurons in the first
        layer triggers adding neuron the second layer.

        Args:
            id1 (int): The id of the module being depended on.
            id2 (int): The id of the module dependent on first module.
        """
        self._register_dependency(id1, id2, DepType.REC)

    def register_same_dependency(self, id1, id2):
        """Marks the dependency between two modules with id1 and id2 as SAME,
        in the sense that id1.operation1 triggers id2.operation1. Useful for
        when after a Linear there is a BatchNorm so adding neurons in the first
        layer triggers adding neuron the second layer.

        Args:
            id1 (int): The id of the module being depended on.
            id2 (int): The id of the module dependent on first module.
        """
        self._register_dependency(id1, id2, DepType.SAME)

    def register_incoming_dependency(self, id1, id2):
        """Marks the dependency between two modules with id1 and id2 as
        INCOMING. Useful for when after a Linear there is a Linear so adding
        neurons in the first layer triggers adding incoming neurons the second
        layer.

        Args:
            id1 (int): The id of the module being depended on.
            id2 (int): The id of the module dependent on first module.
        """
        self._register_dependency(id1, id2, DepType.INCOMING)

    def get_child_ids(self, idd: int, dep_type: DepType):
        """Get the ids of the modules that are dependent on the module with the
        given id.

        Args:
            id (int): The id of the module.
            dep_type (DependencyType): The type of dependency.

        Returns:
            List[int]: The ids of the dependent modules.
        """
        return list(self.dependency_2_id_2_id[dep_type][idd]) if idd in \
            self.dependency_2_id_2_id[dep_type] else []

    def get_parent_ids(self, child_id: int, dep_type: DepType) -> List[int]:
        parents = []
        for parent_id, children in self.dependency_2_id_2_id[dep_type].items():
            if child_id in children:
                parents.append(parent_id)
        return parents
