# Novelty Detection with Autoencoders for System Health Monitoring in Industrial Environments

Predictive Maintenance (PdM) is the newest strategy for maintenance management in industrial contexts. 
It aims to predict the occurrence of a failure to minimize unexpected downtimes and maximize the useful life of components. 
In data-driven approaches, PdM makes use of Machine Learning (ML) algorithms to extract relevant features from signals, 
identify and classify possible faults (diagnostics), and predict the components’ remaining useful life (prognostics). 
The major challenge lies in the high complexity of industrial plants, where both operational conditions change over time 
and a large number of unknown modes occur. A solution to this problem is offered by novelty detection, 
where a representation of the machinery normal operating state is learned and compared with online measurements 
to identify new operating conditions. In this paper, a systematic study of autoencoder-based methods for novelty 
detection is conducted. We introduce an architecture template, which includes a classification layer 
to detect and separate the operative conditions, and a localizer for identifying the most influencing signals. 
Four implementations, with different deep learning models, are described and used to evaluate the approach 
on data collected from a test rig.

For a detailed description of the work please read our [paper](https://www.mdpi.com/2076-3417/12/10/4931). 
Please cite the paper if you use the code from this repository in your work.

```
@article{app12104931,
    author  = {Del Buono, Francesco and Calabrese, Francesca and Baraldi, Andrea and Paganelli, Matteo and Guerra, Francesco},
    title   = {Novelty Detection with Autoencoders for System Health Monitoring in Industrial Environments},
    jornual = {Applied Sciences},
    volume  = {12},
    year    = {2022},
    number  = {10},
}
```


## Library

### Requirements

- Python: Python 3.*
- Packages: requirements.txt

### Installation

```bash
$ cd source

$ virtualenv -p python3 venv

$ source venv/bin/activate

$ pip install -r requirements.txt

```

### Dataset

Please contact us to have access to the data, for research purposes.

### How to Use
Look at:
- *demo.py* is a python example of novelty detection application
- Note: you can run **[this notebook live in Google Colab](https://colab.research.google.com/github/softlab-unimore/AE4ND/blob/master/demo.ipynb)** and use free GPUs provided by Google.

Please feel free to contact me if you need any further information
