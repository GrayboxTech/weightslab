import time
import warnings; warnings.filterwarnings("ignore")
import unittest
import traceback
import torch as th

from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType
from weightslab.backend.model_interface import ModelInterface
from weightslab.utils.tools import model_op_neurons, \
    get_model_parameters_neuronwise
from weightslab.utils.logs import print
from weightslab.baseline_models.pytorch.models import ALL_MODEL_CLASSES


# Set Global Default Settings
DEVICE = 'cpu' if not th.cuda.is_available() else 'cuda'
th.manual_seed(42)  # Set SEED


# --- Test Class 1: Dynamic Inference and Shape Checks ---
class TestAllModelInference(unittest.TestCase):
    """
        Dynamically generated tests to check forward pass and output
        structure/shapes.
    """
    def setUp(self):
        print(f"\n--- Start {self._testMethodName} ---\n")
        self.stamp = time.time()

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")


def create_inference_test(ModelClass):
    """
        Helper to dynamically generate a test method
        for inference verification.
    """

    def _test_inference(self, model, dummy_input, op=None):
        # Infer
        try:
            with th.no_grad():
                output = model(dummy_input)
        except Exception as e:
            print(f"Error during inference: {e}")
            output = None
            traceback.print_exc()
        # Test Inference
        self.assertNotEqual(
            output,
            None,
            f"[{model.get_name()}] Inference fails." + "" if op is None else
            f"\nOperation was {op}"
            ) if self is not None else None

    def _check_model_architecture_neurons_consistency(self, model, debug=False):
        """
            For complicated models only (e.g., FCN50), check that the
            architecture neurons are consistent throughout the model.
            This layer (105) and its dependencies are the most complex as
            it implies both recurrent batchnorms and init convs.
        """
        if not debug:
            print = lambda x: None
        n = 105
        a = model.layers[90].out_neurons == model.layers[91].out_neurons == model.layers[91].in_neurons == model.layers[92].out_neurons == model.layers[93].out_neurons == model.layers[93].in_neurons == model.layers[98].out_neurons == model.layers[99].out_neurons == model.layers[99].in_neurons == model.layers[104].out_neurons == model.layers[105].out_neurons == model.layers[105].in_neurons == model.layers[106].in_neurons
        
        eq = True
        print(f'Operate initially on layer {n}')
        print(f'In/Out are good ?: {a}')

        # Last
        out_shapes = [(k, len(model.layers[106].dst_to_src_mapping_tnsrs[k])) for k in model.layers[106].dst_to_src_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(106:{eq}) Indexs maps are good ?: {out_shapes}')

        # Rec BN
        # src2dst 
        print('\n')
        out_shapes = [(k, len(model.layers[105].src_to_dst_mapping_tnsrs[k])) for k in model.layers[105].src_to_dst_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(105:{eq}) src2dst Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[99].src_to_dst_mapping_tnsrs[k])) for k in model.layers[99].src_to_dst_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(99:{eq}) src2dst Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[93].src_to_dst_mapping_tnsrs[k])) for k in model.layers[93].src_to_dst_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(93:{eq}) src2dst Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[91].src_to_dst_mapping_tnsrs[k])) for k in model.layers[91].src_to_dst_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(91:{eq}) src2dst Indexs maps are good ?: {out_shapes}')
        # dst2src 
        print('\n')
        out_shapes = [(k, len(model.layers[105].dst_to_src_mapping_tnsrs[k])) for k in model.layers[105].dst_to_src_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(105:{eq}) dst2src Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[99].dst_to_src_mapping_tnsrs[k])) for k in model.layers[99].dst_to_src_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(99:{eq}) dst2src Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[93].dst_to_src_mapping_tnsrs[k])) for k in model.layers[93].dst_to_src_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(93:{eq}) dst2src Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[91].dst_to_src_mapping_tnsrs[k])) for k in model.layers[91].dst_to_src_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(91:{eq}) dst2src Indexs maps are good ?: {out_shapes}')

        # Init CN
        print('\n')
        out_shapes = [(k, len(model.layers[104].src_to_dst_mapping_tnsrs[k])) for k in model.layers[104].src_to_dst_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(104:{eq}) Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[98].src_to_dst_mapping_tnsrs[k])) for k in model.layers[98].src_to_dst_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(98:{eq}) Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[92].src_to_dst_mapping_tnsrs[k])) for k in model.layers[92].src_to_dst_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(92:{eq}) Indexs maps are good ?: {out_shapes}')
        out_shapes = [(k, len(model.layers[90].src_to_dst_mapping_tnsrs[k])) for k in model.layers[90].src_to_dst_mapping_tnsrs]
        eq &= len(set([i[1] for i in out_shapes]))==1
        print(f'(90:{eq}) Indexs maps are good ?: {out_shapes}')

        # Assert all good
        self.assertTrue(
            eq,
            f"Model architecture neurons consistency check failed."
        )

    def model_test(self):
        # --- Setup ---
        # # Initialize model
        model = ModelClass()
        # # Create dummy input tensor
        dummy_input = th.randn(model.input_shape).to(DEVICE)
        # # Interface the model
        model = ModelInterface(
            model,
            dummy_input=dummy_input,
            print_graph=False,  
            use_onnx=True if 'VGG' not in ModelClass.__name__ else False
        )
        model.to(DEVICE)
        model.eval()
        model_name = ModelClass.__name__
        print(f"\n--- Running Inference Test: {model_name} ---")

        # --- Forward Pass Testing ---
        _test_inference(self, model, dummy_input)

        # #############################
        # ######## OP. mix ############
        # #############################
        print('Performing model parameters operations..', level='DEBUG')
        model_op_neurons(model, dummy_input=dummy_input, rand=False)
        _test_inference(self, model, dummy_input)

        # For one of the most complicated model, check IN OUT Neurons matching and indexing
        # TODO (GP): Implement something more relevant than just FCN50 excl.
        if ModelClass.__name__ == 'FCN50':
            print('Performing second model parameters operations..', level='DEBUG')
            _test_inference(self, model, dummy_input)
            _check_model_architecture_neurons_consistency(model)  # Check architecture neurons consistency

    # Set unique name for the test
    model_test.__name__ = f'test_{ModelClass.__name__}_inference_check'

    return model_test


# Add dynamic tests to TestAllModelInference
if ALL_MODEL_CLASSES and len(ALL_MODEL_CLASSES) > 0:
    for ModelClass in ALL_MODEL_CLASSES:
        test_method = create_inference_test(ModelClass)
        setattr(TestAllModelInference, test_method.__name__, test_method)


# --- Execution ---
if __name__ == '__main__':
    # Running unittest.main to execute all test classes
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
