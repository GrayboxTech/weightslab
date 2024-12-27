
# WeightsLab

A novel tool that allows you to inteligently reuse the weights from
previous experiments in your current experiment. To achieve this, we provide
a short but comprehensive list of operations on the weights and the datasets.

## Overview

All the features supported by this tool are available through the Experiment
class. A class that contains all the objects and functions to perform the
following operations. You can inspect the:
**main.py** file for an simple example.

```
exp = Experiment(
    model=ModelClass(),
    optimizer_class=optim.Adam,
    train_dataset=**train_dataset**, # a.k.a torch.data.Dataset instance
    eval_dataset=**eval_dataset**,
    device=device,
    learning_rate=1e-2,
    batch_size=256,
    name="x0" # a.k.a Experiment name, used for monitoring in tensorboard.
)

# Train one epoch and eval from time to time.
exp.train_n_steps_with_eval_full(len(exp.train_loader)) 
```

An simple example for a model would look something like this:
```
class MNISTModel(NetworkWithOps):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED
        self.layer0 = Conv2dWithNeuronOps(
            in_channels=1, out_channels=2, kernel_size=3)
        self.mpool0 = nn.MaxPool2d(2)
        self.layer1 = Conv2dWithNeuronOps(
            in_channels=2, out_channels=4, kernel_size=3)
        self.mpool1 = nn.MaxPool2d(2)
        self.layer2 = Conv2dWithNeuronOps(
            in_channels=4, out_channels=10, kernel_size=3)
        self.mpool2 = nn.MaxPool2d(2)
        self.out = LinearWithNeuronOps(in_features=10, out_features=10)
        self.layers = [self.layer0, self.layer1, self.layer2, self.out]

    def forward(self, input: th.Tensor):
        self.maybe_update_age(input)

        x = self.layer0(input)
        x = F.relu(x)
        x = self.mpool0(x)

        x = self.layer1(x)
        x = F.relu(x)
        x = self.mpool1(x)

        x = self.layer2(x)
        x = F.relu(x)
        x = self.mpool2(x)

        x = x.view(x.size(0), -1)
        output = self.out(x, skip_register=True)

        output = F.log_softmax(output, dim=1)
        return output
```

## Statistics about the data samples and neurons

In order to make informed decision w.r.t the supported operations the following
statistics are available and automatically computed at any point:
* For each neuron (note that a trigger mean activation > 0):
    * trigger rate: how many triggers over how many data points have been
      passed through the neuron
```
exp.display_stats()
# trigger_rate (triggers_count / number_of_seen_samples)
#Neuron 00000:  train: 477.3898 (  47146061 /      98758)
#            :  eval : 490.1269 (1171403380 /    2390000)
#Neuron 00001:  train: 487.6750 (  48161808 /      98758)
#            :  eval : 410.1002 ( 980139476 /    2390000)
#Neuron 00002:  train: 673.0587 (  66469933 /      98758)
#            :  eval : 671.2177 (1604210216 /    2390000)
#Neuron 00003:  train: 75.8755 (   7493316 /      98758) <------ NOT GREAT!!!
#            :  eval : 70.0397 ( 167394982 /    2390000)
#Neuron 00004:  train: 72.1667 (   7127041 /      98758) <------ NOT GREAT!!!
#            :  eval : 64.5702 ( 154322888 /    2390000)
#Neuron 00005:  train: 517.0625 (  51064055 /      98758)
#            :  eval : 420.9419 (1006051104 /    2390000)
#Neuron 00006:  train: 539.4792 (  53277883 /      98758)
#            :  eval : 537.7149 (1285138626 /    2390000)
#Neuron 00007:  train: 366.3466 (  36179656 /      98758)
#            :  eval : 297.7676 ( 711664610 /    2390000)
#Neuron 00008:  train: 346.1664 (  34186697 /      98758)
#            :  eval : 456.6625 (1091423392 /    2390000)
#Neuron 00009:  train: 0.0000 (         0 /      98758)  <------ BAD!!!
#            :  eval : 0.0000 (         0 /    2390000)
#Neuron 00010:  train: 84.5720 (   8352157 /      98758)
#            :  eval : 113.1915 ( 270527704 /    2390000)
#Neuron 00011:  train: 296.8443 (  29315754 /      98758)
#            :  eval : 375.7042 ( 897933094 /    2390000)
...

# In this example we can see that the neuron number 9 is not triggering at all
# as well as the fact that neuron 3, 4 trigger significantly less then other
# neurons. It may be a good idea to either reinitialize or prune them.
```
* For each data sample:
    * is denylisted: whether the data sample is still visible during training
    * exposure amount: how many times a given data sample has been passed
      through a model
    * prediction loss: what is the model loss on this example when the pass
      through happened
    * prediction age: how many data samples has the model seen when it made
      this prediction
```
exp.train_loader.dataset.get_dataframe()
#       prediction_age  prediction_loss  exposure_amount deny_listed
# 54378             0.0         2.137688              1.0       False
# 55741             0.0         2.985723              1.0       False
# 50058             0.0         3.413264              1.0       False
# 55867             0.0         3.070853              1.0       False
# 30114             0.0         3.381712              1.0       False
# ...               ...              ...              ...         ...
# 28291         59904.0         0.004244              1.0       False
# 1902          59904.0         0.073017              1.0       False
# 8621          59904.0         0.041011              1.0       False
# 14773         59904.0         0.979033              1.0       False
# 23576         59904.0         0.016847              1.0       False
```

## Supported operations

### Training and evaluation
You can kick start the training or evaluate partially or fully by using the
following commands:

```
# Train for one epoch and evaluate fully from time to time
exp.train_n_steps_with_eval_full(len(exp.train_loader))

# Only train for a given number of steps
exp.train_n_steps(25)

# Evaluate for a subset of the eval set, the loss and accuracy is returned
exp.eval_n_steps(20)

# Or evaluate on the whole set
exp.eval_full()
```

### Architecture operations (only for Conv2d and Linear)
  * Adding (N) neurons to a layer
  * Remove (N) neurons from a layer
  * Reinitialize (N) neurons in a layer
  * Individual learning rate for neurons
```
# Add 4 neurons to the first layer
exp.model.add_neurons(layer_index=0, neuron_count=4)

# Removes first, third and fifth neurons from the first layer
exp.model.prune(layer_index=0, neuron_indices={0, 2, 4}) 

# Reinitialize the weights of sixth and seventh neurons from the first layer
exp.model.reinit_neurons(layer_index=0, neuron_indices={5, 6})

# Pertrub the neuron weights and biases with a little noise
exp.model.reinit_neurons(layer_index=0, neuron_indices={5, 6}, perturbation_ratio=0.2)

# Set per neuron individual learning rate
exp.model.layers[0].set_per_neuron_learning_rate(set(range(0, 2)), lr=0.0)

# Set incoming neuron individual learning rate
exp.model.layers[0].set_per_incoming_neuron_learning_rate(set(range(0, 2)), lr=0.0)
```

### Dynamic layer \[un\]freezing
The layer of the model can be on demand frozen or un-frozen depending on the
circumstances.

```
# To freeze all the layers up to (including) a given index.
exp.model.freeze_layers_up_to_idx(1)

# To unfreeze all the layers from  (including) a given index.
exp.model.unfreeze_layers_from_idx(1)
```

### Data sample operations (for both train and eval set):
  * Denylist (S) samples
  * Allowlist (S) samples (if already denylisted)
```
# Create a dummy denylisting predicate based on predioction loss, for instance 
# remove data samples with low loss because its a signal tha the model learned
# them already.
denylist_fn = lambda id, age, loss, times, denied, pred, label: loss <= 1.5
exp.train_loader.dataset.deny_samples_with_predicate(denylist_fn)

# Denylist all the samples for which the predicate is true, but keep some of
# denied samples such that we keep the dataset balanced. The allowed to denied
# samples ratio is determined by allow_to_deny_factor.
exp.train_loader.dataset.deny_samples_and_sample_allowed_with_predicate(denylist_fn, allow_to_denied_factor=3)

# Allowlisting specific data samples that may or may not be denylisted
exp.train_loader.dataset.allowlist_samples({1, 2, ....})

# Or reinstate all data samples
exp.train_loader.dataset.allowlist_samples(None)
```
If you need to look at the dataset statistics in order to make sense what type
of thresholds to use, you can access the statistics in dataframe format by

```
exp.train_loader.dataset.get_dataframe()
```

### Checkpoints
  * Save a checkpoint: stores all relevant metadata for the:
    model, optimizer and the datasets
  * Load a checkpoint: loads all relevant metadata from a stored checkpoint
    into the current experiment
```
# Saves a checkpoint
exp.dump()

# All available checkpoints can be inspected with
print(exp.chkpt_manager)

# Reload a less trained version of the model, the id has to already exist
exp.load(checkpoint_id=0)
```
