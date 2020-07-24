# SBDIO 

## Requirements

- Python: Python 3.*
- Packages: requirements.txt

## Installation

```bash
$ cd source

$ virtualenv -p python3 venv

$ source venv/bin/activate

$ pip install -r requirements.txt

```

## How to Use

Inside the *models* folder, there are the models already implemented and tested in fixed use case

- One Threshold
- Setup Clustering
- Isolation Forest
- One Class SVM
- Local Outlier Factor (LOF)
- PCA Anomaly Detection


## Evaluation

Univariate Time Series

| Model           	| Accuracy 	| Precision 	| Recall 	| F-score 	|
|-----------------	|----------	|-----------	|--------   |---------  |
| PCA             	| 90.53     | 99.76     	| 84.10     | 91.26     |
| SetupClustering 	| 72.06     | 83.10        	| 65.90     | 73.51     |
| OneClassSVM      	| 58.71     | 59.11       	| 96.70     | 73.37     |
| Isolation Forest	| 66.24     | 68.52       	| 78.80    	| 73.30     |
| LOF             	| 58.58     | 59.06       	| 96.50    	| 73.27     |

Multivariate Time Series

| Model           	| Accuracy 	| Precision 	| Recall 	| F-score 	|
|-----------------	|----------	|-----------	|--------   |---------  |
| PCA             	| 100.0     | 100.0     	| 100.0     | 100.0     |
| SetupClustering 	| 91.17     | 100.0        	| 85.00     | 91.89     |
| OneClassSVM      	| 61.76     | 60.60       	| 100.0     | 75.47     |
| Isolation Forest	| 83.82     | 91.42       	| 80.00    	| 85.33     |
| LOF             	| 60.29     | 59.70       	| 100.0    	| 74.77     |