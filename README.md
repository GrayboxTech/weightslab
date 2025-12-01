<div align="center">
  <img
    src="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/GitHub_banner_WL.png"
    alt="Graybox Logo"
    height="250"
  />
</div>
                                                     
</pred>
</pred style="font-style: italic;">
WeightsLab — Inspect, Edit and Evolve Neural Networks
By Graybx.
</pre>
</div>

## About WeightsLab
WeightsLab is a powerful tool for editing and inspecting data & AI model weights, during training.

### What Problems Does It Solve?
WeightsLab addresses critical training challenges:

* Overfitting and training plateau
* Dataset insights & optimization
* Over/Under parameterization

### Key Capabilities
The granular statistics and interactive paradigm enables powerful workflows:

* Monitor granular insights on data samples and weights parameters
* Discard low quality samples by click or query
* Create slices of data and discard them during training
* Iterative pruning or growing of the architectures by click or query


## Getting Started
### Watch our demo below:

<div style="position: relative; padding-bottom: 62.5%; height: 0;">
  <iframe src="https://www.loom.com/embed/5d04822a0933427d971d320f64687730?sid=8f3e4c5d-9b4e-4f3e-a5d1-2c3d4e5f6g7h" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

### Installation
Define a python environment
```bash
python -m venv weightslab_venv
./weightslab_venv/Scripts/activate
```
or install and use conda.

Clone and install the framework (CLI based interaction):

```bash
git clone https://github.com/GrayboxTech/weightslab.git
cd weightslab
pip install -e .
```

Clone the UI repository (UI based interaction; cf. loom video):
```bash
git clone git@github.com:GrayboxTech/weightslab_ui.git
cd weightslab_ui
pip install -r ./requirements.txt
```


### GrayBx Usage Example
Several code examples show different usage from toy to more complicated models (based with PyTorch).

Quickstart examples:
- [Toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/toy-pytorch_example)
- [Advanced (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/advanced-pytorch_example)

<!-- ### Documentation -->

### Community

Graybx is built by a wonderful community of researchers and engineers. 


### Citation

If you publish work that uses Graybx, please cite Graybx as follows:

```bibtex
@article{graybx2025,
  title={Graybox: A Friendly BlackBox interactive approach},
  author={Luigi, Alex, Marc, And Guillaume},
  year={2025}
}
```
