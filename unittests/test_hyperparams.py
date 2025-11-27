import os
import tempfile
import yaml
import unittest

from weightslab.backend.ledgers import GLOBAL_LEDGER, register_hyperparams, get_hyperparams, set_hyperparam


class HyperparamsTests(unittest.TestCase):
    def setUp(self):
        # clear global ledger to ensure isolation between tests
        GLOBAL_LEDGER.clear()

    def test_register_and_get_hyperparams(self):
        params = {
            'a': 1,
            'b': {'c': 2, 'd': 3},
            'e': [4, 5]
        }
        register_hyperparams('test_exp', params)
        loaded = get_hyperparams('test_exp')
        self.assertEqual(loaded['a'], 1)
        self.assertEqual(loaded['b']['c'], 2)
        self.assertEqual(loaded['e'], [4, 5])

    def test_set_hyperparam_dot_path(self):
        params = {'x': {'y': {'z': 10}}}
        register_hyperparams('test_exp2', params)
        set_hyperparam('test_exp2', 'x.y.z', 42)
        loaded = get_hyperparams('test_exp2')
        self.assertEqual(loaded['x']['y']['z'], 42)

    def test_reload_from_yaml(self):
        # Simulate YAML reload by writing a temp file and registering it
        td = tempfile.TemporaryDirectory()
        try:
            yaml_path = os.path.join(td.name, 'config.yaml')
            params = {'foo': 123, 'bar': {'baz': 456}}
            with open(yaml_path, 'w') as f:
                yaml.dump(params, f)
            with open(yaml_path, 'r') as f:
                loaded = yaml.safe_load(f)
            register_hyperparams('yaml_exp', loaded)
            self.assertEqual(get_hyperparams('yaml_exp')['foo'], 123)
            self.assertEqual(get_hyperparams('yaml_exp')['bar']['baz'], 456)
        finally:
            td.cleanup()


if __name__ == '__main__':
    unittest.main()
