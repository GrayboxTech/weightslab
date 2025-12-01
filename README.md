<div align="center">
  <img
    src="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/GitHub_banner_WL.png"
    alt="Graybox Logo"
    height="250"
  />
  <h3>Place your AI anywhere between blackbox and whitebox — with insights and control</h3>
</div>


# WeightsLab: A Friendly BlackBox interactive approach

<div style="text-align: center;">
<pre style="background-color: transparent;">
<span style="color: red;">/WW      /WW</span>           /aa           /aa         /aa               <span style="color: green;">/aa</span>                 /aa      
<span style="color: red;">| WW  /W | WW</span>          |__/          | aa        | aa              <span style="color: green;">| aa</span>                | aa      
<span style="color: red;">| WW /WWW| WW</span>  /aaaaaa  /aa  /aaaaaa | aaaaaaa  /aaaaaa    /aaaaaaa<span style="color: green;">| aa</span>        /aaaaaa | aaaaaaa 
<span style="color: red;">| WW/WW WW WW</span> /aa__  aa| aa /aa__  aa| aa__  aa|_  aa_/   /aa_____/<span style="color: green;">| aa</span>       |____  aa| aa__  aa
<span style="color: red;">| WWWW_  WWWW</span>| aaaaaaaa| aa| aa  \ aa| aa  \ aa  | aa    |  aaaaaa <span style="color: green;">| aa</span>        /aaaaaaa| aa  \ aa
<span style="color: red;">| WWW/ \  WWW</span>| aa_____/| aa| aa  | aa| aa  | aa  | aa /aa \____  aa<span style="color: green;">| aa</span>       /aa__  aa| aa  | aa
<span style="color: red;">| WW/   \  WW</span>|  aaaaaaa| aa|  aaaaaaa| aa  | aa  |  aaaa/ /aaaaaaa/<span style="color: green;">| aaaaaaaa</span>  aaaaaaa| aaaaaaa/
<span style="color: red;">|__/     \__/</span> \_______/|__/ \____  aa|__/  |__/   \___/  |_______/ <span style="color: green;">|________/</span> \_______/|_______/ 
                            /aa  \ aa                                                            
                           |  aaaaaa/                                                            
                            \______/                                                             
</pred>
</pred style="font-style: italic;">
WeightsLab — Inspect, Edit and Evolve Neural Networks
By GrayBx.
</pre>
</div>


<p align="center">
    <a href="https://www.linkedin.com/company/graybx-com/posts/?feedView=all">Website</a> |
    <a href="https://www.linkedin.com/company/graybx-com/posts/?feedView=all">In</a> |
    <a href="https://www.linkedin.com/company/graybx-com/posts/?feedView=all">Blog</a> |
    <a href="https://www.linkedin.com/company/graybx-com/posts/?feedView=all">Docs</a> |
    <a href="https://www.linkedin.com/company/graybx-com/posts/?feedView=all">Slack</a>
    <br /><br />
</p>

[![GitHub license](https://github.com/GrayboxTech/weightslab/blob/main/LICENSE)](None)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](None)
![Build](https://github.com/adap/flower/actions/workflows/framework.yml/badge.svg)
[![Downloads](https://static.pepy.tech/badge/flwr)](None)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-flwr-blue)](None)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](None)

## Presentation
WeightsLab is a powerful tool for editing and inspecting AI model weights during training.
This early prototype helps you debug and fix common training issues through interactive weight manipulation and granular analysis.

### What Problems Does It Solve?
WeightsLab addresses critical training challenges:

* Overfitting and training plateaus
* Dataset insights and optimization
* Over/Under parameterization

### Key Capabilities
The granular statistics and interactive paradigm enables powerful workflows:

* Monitor granular insights on data samples and weights parameters
* Discard low quality samples by click or query
* Create slices of data and discard them during training
* Iterative pruning or growing of the architectures by click or query


## Getting Started
### 6-Minute Loom
[![Loom](https://img.shields.io/badge/loom%20-0000FF)](https://www.loom.com/share/5d04822a0933427d971d320f64687730)

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

Graybox is built by a wonderful community of researchers and engineers. 


### Citation

If you publish work that uses GrayBx, please cite Graybx as follows:

```bibtex
@article{graybx2025,
  title={Graybox: A Friendly BlackBox interactive approach},
  author={Luigi, Alex, Marc, And Guillaume},
  year={2025}
}
```
