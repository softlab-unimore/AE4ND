# Big Data In & OUT services for Industry 4.0: from shopfloor to post-sales

SBDIO I4.0 is an industrial research project which aims to sustain 
local industries in their digitalization process, fostering transitions 
from a product economy to a services economy.

SBDIO I4.0 project stems from the needs of all the companies belonging 
to the industrial automation chain and based in the Emilia-Romagna Region, 
with the goal to simplify the transition from a product economy 
to a services economy.

The main objective of the project is the creation of a smart platform 
capable to offer a double servitization process for the industries: 
in the production (shopfloor) and in the post-sales services. 
This progressive transition can be developed by using an innovative 
approach based on Artificial Intelligence and Machine Learning 
algorithms applied on big data.

The project involves 5 industrial research laboratories belonging 
to 3 Universities and 1 research center in addition 
to the participation of 7 companies.

## Goal

- Big Data: to adopt new big data technology in industrial field
- Industry 4.0: transition from a product economy to a services economy
- Artificial Intelligence: servitization of both production and post-sales services
- Machine Learning: Predictive models to anticipate maintenance actions and breakdown


## Machine Learning

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

### How to Use

Inside the *timely* folder, there are the models and transformed implemented and tested in industrial use case, 
instead in *demo* there is an example of anomaly detection application

### Anomaly Detection Model
- CNN AutoEncoder
- LSTM AutoEncoder
- MLP AutoEncoder
- Setup Clustering
- Isolation Forest
- One Class SVM
- Local Outlier Factor (LOF)
- PCA Anomaly Detection
- One Threshold


[comment]: <> (### Evaluation)

[comment]: <> (Univariate Time Series)

[comment]: <> (| Model           	| Accuracy 	| Precision 	| Recall 	| F-score 	|)

[comment]: <> (|-----------------	|----------	|-----------	|--------   |---------  |)

[comment]: <> (| PCA             	| 90.53     | 99.76     	| 84.10     | 91.26     |)

[comment]: <> (| SetupClustering 	| 72.06     | 83.10        	| 65.90     | 73.51     |)

[comment]: <> (| OneClassSVM      	| 58.71     | 59.11       	| 96.70     | 73.37     |)

[comment]: <> (| Isolation Forest	| 66.24     | 68.52       	| 78.80    	| 73.30     |)

[comment]: <> (| LOF             	| 58.58     | 59.06       	| 96.50    	| 73.27     |)

[comment]: <> (Multivariate Time Series)

[comment]: <> (| Model           	| Accuracy 	| Precision 	| Recall 	| F-score 	|)

[comment]: <> (|-----------------	|----------	|-----------	|--------   |---------  |)

[comment]: <> (| PCA             	| 100.0     | 100.0     	| 100.0     | 100.0     |)

[comment]: <> (| SetupClustering 	| 91.17     | 100.0        	| 85.00     | 91.89     |)

[comment]: <> (| OneClassSVM      	| 61.76     | 60.60       	| 100.0     | 75.47     |)

[comment]: <> (| Isolation Forest	| 83.82     | 91.42       	| 80.00    	| 85.33     |)

[comment]: <> (| LOF             	| 60.29     | 59.70       	| 100.0    	| 74.77     |)