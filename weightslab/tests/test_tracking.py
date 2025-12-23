""" Test tracking related interfaces and classes. """
import time
import shutil
import unittest
import tempfile
import warnings; warnings.filterwarnings("ignore")
import torch as th

from os import path

from weightslab.components.tracking import (
    add_tracked_attrs_to_input_tensor,
    TriggersTracker,
    TriggersTrackerClazzAndSampleID
)


# Set Global Default Settings
DEVICE = 'cpu' if not th.cuda.is_available() else 'cuda'
th.manual_seed(42)  # Set SEED


class TriggersTrackerTest(unittest.TestCase):
    """
        Tests the TriggersTracker for the 4 neuron ops operation and persistent
        storage capability.
    """
    def setUp(self):
        print(f"\n--- Start {self._testMethodName} ---\n")
        self.batch_size = 2
        self.device = DEVICE
        self.triggers_tracker = TriggersTracker(2, self.device)
        self.test_dir = '/tmp/utests/'
        self.stamp = time.time()

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        shutil.rmtree(self.test_dir)
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

    def test_triggers_tracker_one_update(self):
        # [batch_size x neuron_count]
        processed_activation_map = th.tensor([[4, 6], [0, 2]]).to(self.device)
        self.triggers_tracker.update(processed_activation_map)
        # Check if each neuron trackers has seen exactly 2 samples, since
        # batch size is 2.

        # check first neuron
        self.assertEqual(self.triggers_tracker.get_neuron_age(0), 2)
        # check second neuron
        self.assertEqual(self.triggers_tracker.get_neuron_age(1), 2)
        # Check if each neuron has the correct number of triggers in the
        # counters.
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(0), 4)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(1), 8)

    def test_triggers_tracker_two_updates(self):
        # [batch_size x neuron_count]
        processed_activation_map1 = th.tensor([[4, 6], [0, 2]]).to(self.device)
        processed_activation_map2 = th.tensor(
            [[0, 1], [7, 2], [3, 3], [12, 0]]).to(self.device)
        self.triggers_tracker.update(processed_activation_map1)
        self.triggers_tracker.update(processed_activation_map2)
        self.assertEqual(self.triggers_tracker.get_neuron_age(0), 6)
        self.assertEqual(self.triggers_tracker.get_neuron_age(1), 6)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(0), 26)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(1), 14)

    def test_triggers_tracker_one_update_and_prune(self):
        processed_activation_map = th.tensor([[4, 6], [0, 2]]).to(self.device)
        self.triggers_tracker.update(processed_activation_map)
        self.assertEqual(self.triggers_tracker.get_neuron_number(), 2)
        self.triggers_tracker.prune({0})
        self.assertEqual(self.triggers_tracker.get_neuron_number(), 1)
        self.assertEqual(self.triggers_tracker.get_neuron_age(0), 2)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(0), 8)

    def test_triggers_tracker_one_update_and_reset(self):
        processed_activation_map = th.tensor([[4, 6], [0, 2]]).to(self.device)
        self.triggers_tracker.update(processed_activation_map)
        self.triggers_tracker.reset({0})
        self.assertEqual(self.triggers_tracker.get_neuron_age(0), 0)
        self.assertEqual(self.triggers_tracker.get_neuron_age(1), 2)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(0), 0)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(1), 8)

    def test_triggers_tracker_one_update_and_add_neurons_update(self):
        processed_activation_map = th.tensor([[4, 6], [0, 2]]).to(self.device)
        self.triggers_tracker.update(processed_activation_map)
        self.triggers_tracker.add_neurons(2)
        self.assertEqual(self.triggers_tracker.get_neuron_number(), 4)
        processed_activation_map_2 = th.tensor([
            [1, 2, 1, 2], [2, 7, 7, 19], [1, 1, 3, 1], [7, 0, 12, 0]]).to(
                self.device)
        self.triggers_tracker.update(processed_activation_map_2)
        self.assertEqual(self.triggers_tracker.get_neuron_age(0), 6)
        self.assertEqual(self.triggers_tracker.get_neuron_age(1), 6)
        self.assertEqual(self.triggers_tracker.get_neuron_age(2), 4)
        self.assertEqual(self.triggers_tracker.get_neuron_age(3), 4)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(0), 15)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(1), 18)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(2), 23)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(3), 22)

    def test_triggers_tracker_all_ops(self):
        processed_activation_map = th.tensor([[4, 6], [0, 2]]).to(self.device)
        self.triggers_tracker.update(processed_activation_map)
        self.triggers_tracker.add_neurons(2)
        self.assertEqual(self.triggers_tracker.get_neuron_number(), 4)
        raw_values = [[1, 2, 1, 2], [2, 7, 7, 19], [1, 1, 3, 1], [7, 0, 12, 0]]
        processed_activation_map_2 = th.tensor(raw_values).to(self.device)
        self.triggers_tracker.update(processed_activation_map_2)
        self.triggers_tracker.reset({3})
        self.triggers_tracker.prune({2})
        self.assertEqual(self.triggers_tracker.get_neuron_number(), 3)
        self.assertEqual(self.triggers_tracker.get_neuron_age(0), 6)
        self.assertEqual(self.triggers_tracker.get_neuron_age(1), 6)
        self.assertEqual(self.triggers_tracker.get_neuron_age(2), 0)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(0), 15)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(1), 18)
        self.assertEqual(self.triggers_tracker.get_neuron_triggers(2), 0)

    def test_triggers_tracker_all_ops_plus_save_and_load(self):
        replica_triggers_tracker = TriggersTracker(2, self.device)

        processed_activation_map = th.tensor([[4, 6], [0, 2]]).to(self.device)
        self.triggers_tracker.update(processed_activation_map)
        self.triggers_tracker.add_neurons(2)
        self.assertEqual(self.triggers_tracker.get_neuron_number(), 4)
        raw_values = [[1, 2, 1, 2], [2, 7, 7, 19], [1, 1, 3, 1], [7, 0, 12, 0]]
        processed_activation_map_2 = th.tensor(raw_values).to(self.device)
        self.triggers_tracker.update(processed_activation_map_2)
        self.triggers_tracker.reset({3})
        self.triggers_tracker.prune({2})

        state_dict_file_path = path.join(self.test_dir, 'triggers_tracker.txt')
        th.save(self.triggers_tracker.state_dict(), state_dict_file_path)
        state_dict = th.load(state_dict_file_path)

        self.assertNotEqual(self.triggers_tracker, replica_triggers_tracker)
        replica_triggers_tracker.load_state_dict(state_dict, strict=False)
        self.assertEqual(self.triggers_tracker, replica_triggers_tracker)


class TriggersTrackerClazzAndSampleIDTest(unittest.TestCase):
    """ Tests the TriggersTrackerClazzAndSampleID for the neuron operations and
        persistency.
    """
    def setUp(self):
        self.batch_size = 2
        self.device = th.device("cpu")
        self.triggers_tracker = TriggersTrackerClazzAndSampleID(2, self.device)
        self.test_dir = '/tmp/utests/'

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_update(self):
        processed_activation_map = th.tensor([[0, 1], [0, 1]])

        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([12, 789]),
            label_batch=th.tensor([0, 4])
        )

        self.triggers_tracker.update(processed_activation_map)

        neuron0_stats = self.triggers_tracker.get_neuron_stats(0)
        self.assertListEqual(neuron0_stats[1], [])
        self.assertListEqual(neuron0_stats[2], [])

        neuron1_stats = self.triggers_tracker.get_neuron_stats(1)
        self.assertListEqual(neuron1_stats[1], [12, 789])
        self.assertListEqual(neuron1_stats[2], [(0, 1), (4, 1)])

    def test_update_and_prune(self):
        processed_activation_map = th.tensor([[1, 1], [0, 1]])

        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([12, 789]),
            label_batch=th.tensor([0, 4])
        )
        self.triggers_tracker.update(processed_activation_map)
        self.triggers_tracker.prune({0})
        self.assertTrue(self.triggers_tracker.number_of_neurons.item() == 1)

        processed_activation_map = th.tensor([[5], [1]])

        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([55, 234]),
            label_batch=th.tensor([2, 0])
        )

        self.triggers_tracker.update(processed_activation_map)

        neuron_stats = self.triggers_tracker.get_neuron_stats(0)
        self.assertListEqual(neuron_stats[1], [234, 12, 789, 55])
        self.assertListEqual(neuron_stats[2], [(0, 2), (4, 1), (2, 1)])

    def test_update_and_reset(self):
        processed_activation_map = th.tensor([[1, 1], [0, 1]])
        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([12, 789]),
            label_batch=th.tensor([0, 4])
        )
        self.triggers_tracker.update(processed_activation_map)
        self.triggers_tracker.reset({1})
        self.assertTrue(self.triggers_tracker.number_of_neurons.item() == 2)

        neuron0_stats = self.triggers_tracker.get_neuron_stats(0)
        self.assertListEqual(neuron0_stats[1], [12])
        self.assertListEqual(neuron0_stats[2], [(0, 1)])

        neuron1_stats = self.triggers_tracker.get_neuron_stats(1)
        self.assertListEqual(neuron1_stats[1], [])
        self.assertListEqual(neuron1_stats[2], [])

        processed_activation_map = th.tensor([[1, 8], [0, 2]])
        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([345, 2324]),
            label_batch=th.tensor([4, 7])
        )
        self.triggers_tracker.update(processed_activation_map)

        neuron0_stats = self.triggers_tracker.get_neuron_stats(0)
        self.assertListEqual(neuron0_stats[1], [345, 12])
        self.assertListEqual(neuron0_stats[2], [(0, 1), (4, 1)])

        neuron1_stats = self.triggers_tracker.get_neuron_stats(1)
        self.assertListEqual(neuron1_stats[1], [345, 2324])
        self.assertListEqual(neuron1_stats[2], [(4, 1), (7, 1)])

    def test_update_and_add_neurons(self):
        processed_activation_map = th.tensor([[1, 1], [0, 1]])

        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([12, 789]),
            label_batch=th.tensor([0, 4])
        )
        self.triggers_tracker.update(processed_activation_map)
        self.triggers_tracker.add_neurons(1)
        self.assertTrue(self.triggers_tracker.number_of_neurons.item() == 3)
        with self.assertRaises(RuntimeError):
            self.triggers_tracker.update(processed_activation_map)
        processed_activation_map = th.tensor([[1, 8, 3], [0, 2, 4]])

        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([345, 2324]),
            label_batch=th.tensor([4, 7])
        )
        self.triggers_tracker.update(processed_activation_map)

        neuron0_stats = self.triggers_tracker.get_neuron_stats(0)
        self.assertListEqual(neuron0_stats[1], [345, 12])
        self.assertListEqual(neuron0_stats[2], [(0, 1), (4, 1)])

        neuron1_stats = self.triggers_tracker.get_neuron_stats(1)
        self.assertListEqual(neuron1_stats[1], [345, 2324, 12, 789])
        self.assertListEqual(neuron1_stats[2], [(4, 2), (0, 1), (7, 1)])

        neuron2_stats = self.triggers_tracker.get_neuron_stats(2)
        self.assertListEqual(neuron2_stats[1], [345, 2324])
        self.assertListEqual(neuron2_stats[2], [(4, 1), (7, 1)])

    def test_store_and_load(self):
        replica_triggers_tracker = TriggersTrackerClazzAndSampleID(
            2, self.device)

        processed_activation_map = th.tensor([[1, 1], [0, 1]])
        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([12, 789]),
            label_batch=th.tensor([0, 4]))
        self.triggers_tracker.update(processed_activation_map)

        self.triggers_tracker.add_neurons(1)

        processed_activation_map = th.tensor([[1, 8, 3], [0, 2, 4]])
        add_tracked_attrs_to_input_tensor(
            processed_activation_map,
            in_id_batch=th.tensor([345, 2324]),
            label_batch=th.tensor([4, 7]))
        self.triggers_tracker.update(processed_activation_map)

        state_dict_file_path = path.join(self.test_dir, 'triggers_tracker.txt')
        th.save(self.triggers_tracker.state_dict(), state_dict_file_path)

        state_dict = th.load(state_dict_file_path)

        self.assertNotEqual(self.triggers_tracker, replica_triggers_tracker)
        replica_triggers_tracker.load_state_dict(state_dict, strict=False)
        self.assertEqual(self.triggers_tracker, replica_triggers_tracker)


if __name__ == '__main__':
    unittest.main()
