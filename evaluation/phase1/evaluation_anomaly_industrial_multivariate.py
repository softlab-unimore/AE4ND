import numpy as np
import pandas as pd

from timely.utils.tools import get_time_series_dataset, get_sliding_window_matrix, \
    get_settings, check_setup, lazy_check_setup, label_settings
from timely.utils.manage_file import get_files

from timely.models.anomaly_detection.setup_clustering import SetupClustering
from timely.models.anomaly_detection.isolation_forest import IsolationForest
from timely.models.anomaly_detection.one_class_svm import OneClassSVM
from timely.models.anomaly_detection.local_outlier_factor import LOF
from timely.models.anomaly_detection.pca import PCA

# Input folders
global_dir = '/Users/delbu/Projects/Notebooks/PHM/Dataset/Dataset_1/LOG1HZ'
settings_file = '/Users/delbu/Projects/PycharmProjects/SBDIO/data/settings_Ricette.CSV'

# Data preparation params
kernel = 200
stride = 10

# Empty list will select all input columns like features
# This is valid only for multi variate timely series anomaly detection
features_list = [
    'TEMP_E1',
    'TEMP_E2',
    'TEMP_E3',
    'TEMP_E4',
    'TEMP_E5',
]

# Model type
model_type = 'clustering'


def get_normal(files, setups):
    for file in files:
        if file not in setups:
            return file

    return None


constant_normal_files = {
    '0': r'/Users/delbu/Projects/Notebooks/PHM/Dataset/Dataset_1/LOG1HZ\File (5).CSV',
    '4': r'/Users/delbu/Projects/Notebooks/PHM/Dataset/Dataset_1/LOG1HZ\File (35).CSV',
}

normal_file_id_list = [
    '14', '17', '25', '27', '34', '35', '36', '37', '38', '39', '41', '42', '43', '44', '45', '46', '48', '50', '51',
    '56', '57', '58', '62', '63', '64', '67', '69', '5'
]


def main():
    # Get all .CSV files in global folder
    files = get_files(global_dir, ext='.CSV')
    print('Found {} files'.format(len(files)))

    # Get settings dataset, where each row represent a new setting entry
    ds_settings = get_time_series_dataset(settings_file, sep=';', col='DT')
    print('Found {} settings'.format(len(ds_settings)))

    # Identify settings label
    label_settings(ds_settings, ds_settings.columns[:13])
    ds_settings.ltime = pd.to_datetime(ds_settings.ltime)
    ds_settings.rtime = pd.to_datetime(ds_settings.rtime)
    print('Found {} unique settings'.format(len(np.unique(ds_settings.label))))

    settings_map = {}
    setup_files = []

    # Create settings map that associates a setting to each file
    print('\nSettings File identification')
    for file in files:
        # Read dataset
        ds = get_time_series_dataset(file, sep=';', col='DT')

        # Get nearest left setting
        setting = get_settings(ds, ds_settings)

        # Update settings_map
        if str(setting.label) not in settings_map:
            settings_map[str(setting.label)] = [file]
        else:
            settings_map[str(setting.label)] += [file]

        # Check if the setting start overlap with file timely interval
        if check_setup(ds, setting):
            print('Found setup {}: {} - {} in ds {} - {}'.format(setting.label, setting.ltime, setting.rtime,
                                                                 ds.index.min(), ds.index.max()))
            setup_files += [file]

        elif lazy_check_setup(ds, setting):
            print('Found lazy setup {}: {} - {} in ds {} - {}'.format(setting.label, setting.ltime, setting.rtime,
                                                                      ds.index.min(), ds.index.max()))
            setup_files += [file]

    print('Number of timely series with setup: {}'.format(len(setup_files)))

    y_pred_multi = {}
    y_true_multi = {}

    normal_files = {}

    # Save settings_map and setup_files list
    # with open('../results/settings_map.json', 'w') as outfile:
    #     json.dump(settings_map, outfile)
    #
    # with open('../results/setup_files.json', 'w') as outfile:
    #     json.dump(setup_files, outfile)

    # For each state we train a models with a "normal" file and predict anomalies
    print('\nTraining and Testing - {}'.format(model_type))
    for k, val in settings_map.items():
        print('\nState {} has {} files'.format(k, len(val)))

        # Get normal file from constant_normal_files dictionary
        if k not in constant_normal_files:
            print('No normal files founded')
            continue

        normal_file = constant_normal_files[k]
        normal_files[k] = normal_file

        if normal_file is None:
            print('Impossible get normal file')
            return

        # Training
        ds_train = get_time_series_dataset(filename=normal_file, sep=';', col='DT')
        # Check train
        if ds_train is None:
            print('Impossible read train file')
            return

        y_pred_multi[k] = []
        y_true_multi[k] = []

        x_train = ds_train[features_list]
        x_train = get_sliding_window_matrix(x_train.values, kernel, stride)

        # Selected models
        if model_type == 'pca':
            model = PCA(n_components=0.95, threshold=100, c_alpha=3.2905)
        elif model_type == 'clustering':
            model = SetupClustering(distance="cosine", max_dist=0.001, anomaly_threshold=0.0001)
        elif model_type == 'svm':
            model = OneClassSVM(nu=0.001, tol=0.001, kernel="rbf", gamma="scale")
        elif model_type == 'lof':
            model = LOF(n_neighbors=50, algorithm='auto', metric='minkowski', contamination='auto')
        elif model_type == 'if':
            model = IsolationForest(n_estimators=200, max_samples=512, contamination=0.0003, max_features=0.8)
        else:
            print("Select the wrong models")
            return

        # Training
        print("Training... state {}".format(k))
        model.fit(x_train)

        print("Testing...")
        for file in val:
            # y_true_single is useless
            # setup_files doesn't have value for label
            if file in setup_files:
                y_true_multi[k].append(1)
            else:
                y_true_multi[k].append(0)

            x_test = get_time_series_dataset(filename=file, sep=';', col='DT')

            # Check test
            if x_test is None:
                print('Impossible read test file')
                return

            # Create testing values
            x_test = x_test[features_list]
            x_test = get_sliding_window_matrix(x_test.values, kernel, kernel)

            # Testing
            y_pred = model.predict(x_test)

            # Save number of detected anomalies
            y_pred_multi[k].append(len(y_pred[y_pred == 1]))

        # break

    print('\nSelected normal files:')
    for k, file in normal_files.items():
        print("State {} -> {}".format(k, file))

    # Create result dataset
    y_pred = []
    y_true = []
    files = []
    states = []

    for k in y_pred_multi.keys():
        i = 0
        for pred, true in zip(y_pred_multi[k], y_true_multi[k]):
            y_pred.append(pred)
            y_true.append(true)
            files.append(settings_map[k][i])
            states.append(k)
            i += 1

    res_ds = pd.DataFrame({
        'file': files,
        'states': states,
        'y_pred': y_pred,
        'y_true': y_true
    })

    # Create real ground truth
    res_ds['file'] = res_ds['file'].apply(lambda x: x.split('\\')[-1])
    normal_file_list = ["File ({}).CSV".format(x) for x in normal_file_id_list]
    res_ds['y_true'] = 1
    res_ds.loc[res_ds['file'].isin(normal_file_list), 'y_true'] = 0

    # Save results
    res_ds.to_csv('../results/{}_evaluation.CSV'.format(model_type), sep=';', index=False)

    # Evaluation
    print("\nEvaluation")
    true_positive = len(res_ds[(res_ds['y_pred'] > 0) & (res_ds['y_true'] > 0)])
    false_positive = len(res_ds[(res_ds['y_pred'] > 0) & (res_ds['y_true'] == 0)])
    true_negative = len(res_ds[(res_ds['y_pred'] <= 0) & (res_ds['y_true'] == 0)])
    false_negative = len(res_ds[(res_ds['y_pred'] <= 0) & (res_ds['y_true'] > 0)])

    acc = 100 * (true_positive + true_negative) / len(res_ds)
    print("Accuracy: {}".format(acc))

    precision = 100 * true_positive / (true_positive + false_positive)
    print("Precision: {}".format(precision))
    recall = 100 * true_positive / (true_positive + false_negative)
    print("Recall: {}".format(recall))
    f_score = 2 * precision * recall / (precision + recall)
    print("F-score: {}".format(f_score))


if __name__ == '__main__':
    main()
