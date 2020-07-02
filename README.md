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

Inside the *model* folder, there are the models already implemented and tested in fixed use case

- One Threshold
- Setup Clustering

### One Threshold
Tecnica base che consente di riconoscere variazioni nei dati specificando 
l’istante temporale dell’inserimento del setting e andando a individuare 
l’istante di tempo in cui si raggiunge la condizione di equilibrio

```bash
$ python one_threshold.py -h
usage: one_threshold.py [-h] --file FILE [--settings SETTINGS] [--sep SEP]
                        [--save]
Run "One Threshold Algorithm"

optional arguments:
  -h, --help           show this help message and exit
  --file FILE          file to analise
  --settings SETTINGS  settings to provide the setup starting time
  --sep SEP            table separator
  --save               to save the final algorithm result
```

```bash
python one_threshold.py --file data/ts_anomaly_setup1.CSV --settings data/settings_Ricette.CSV

anomaly:
+----+-----------+---------------------+---------------------+
|    | feature   | start               | end                 |
|----+-----------+---------------------+---------------------|
|  0 | params_2  | 2018-01-11 21:19:12 | 2018-01-11 21:54:03 |
|  1 | params_3  | 2018-01-11 21:19:12 | 2018-01-11 21:48:31 |
|  2 | params_4  | 2018-01-11 21:19:12 | 2018-01-11 22:00:28 |
...
```

### Clustering
Tecnica di estrazione di pattern dal comportamento normale del macchinario 
per l'individuazione di anomalie sulla base dello scostamento.

L’algoritmo impara pattern durante il normale funzionamento del componente, 
di conseguenza individua le condizioni di avviamento come anomalie, senza dover indicare 
l’inserimento del nuovo settaggio. Inoltre, il modello può essere 
aggiornato online con nuove condizioni operative imparando 
nuovi pattern, e individuare outlier in qualsiasi momento.

```bash
$ python demo.clustering.py -h
usage: clustering.py [-h] --test TEST --train TRAIN [--sep SEP]
                     [--features FEATURES [FEATURES ...]] [--save]

Run "Setup Clustering Algorithm"

optional arguments:
  -h, --help            show this help message and exit
  --test TEST           file to analyze and detect abnormal condition
  --train TRAIN         file used to learn the normal state
  --sep SEP             table separator to analyze input dataset
  --features FEATURES [FEATURES ...]
                        feature list where the algorithm is applied
  --save                to save the final algorithm result
```

```bash
$python clustering.py --train data/ts_normal1.CSV --test data/ts_anomaly_setup1.CSV --single

+----+-----------+---------------------+---------------------+
|    | feature   | start               | end                 |
|----+-----------+---------------------+---------------------|
|  0 | params_2  | 2018-01-11 21:19:12 | 2018-01-11 22:02:31 |
|  1 | params_3  | 2018-01-11 21:19:12 | 2018-01-11 21:22:31 |
|  2 | params_3  | 2018-01-11 21:32:32 | 2018-01-11 21:39:11 |
|  3 | params_4  | 2018-01-11 21:19:12 | 2018-01-11 22:05:51 |
...
```

Algorithm params are inside the *params* folder (params/params_clustering.json)

#### Tested Params
Sliding Window
```json
{
  "kernel": 200,
  "stride": 10
}
```

*Cosine* distance
```json
{
  "distance": "cosine",
  "max_distance": 0.001,
  "anomaly_threshold": 0.001,
}
```
