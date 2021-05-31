import statistics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer



from src.feature_selection import get_features_from_patients, get_baseline_from_patients
from src.feature_extraction import get_patient_hrv_features, get_patient_hrv_baseline_features
from src.detection import train_test_svm_wrapper, create_dataset
import src.feature_extraction.io

def decision_algorithm (patient:int, crisis:int, state:str):

    features= get_patient_hrv_features(patient, crisis)
    features_baseline = get_baseline_from_patients([patient], state)
    features=get_features_from_patients([patient], [crisis], state)
    #dataset_x, dataset_y= create_dataset(features)

    model = src.feature_extraction.io.__load_model(patient)
    labels= src.feature_extraction.io.__read_labels(patient)

    def checklist(y_prediction):
        for i in range(len(y_prediction) - 1):
            if y_prediction[i] == 1 and y_prediction[i + 1] == 1:
                print('There is crisis')
                return True
        print('There is no crisis')
        return False
    results=[]
    for p in features.keys():
        for c in features[p].keys():
            for column in features[p][c].columns:
                if column not in labels:
                    features[p][c] = features[p][c].drop(column, axis=1)
            y_prediction = model.predict(features[p][c].to_numpy().tolist())
            print("Y_predicted = ", y_prediction)
            count_crisis_zeroes = y_prediction.tolist().count(0)
            count_crisis_ones = y_prediction.tolist().count(1)
            print('Zeros in crisis:', count_crisis_zeroes, 'Ones in crisis:', count_crisis_ones)
            results.append(checklist(y_prediction))
    for p in features_baseline.keys():
        for column in features_baseline[p].columns:
            if column not in labels:
                features_baseline[p] = features_baseline[p].drop(column, axis=1)
        y_prediction_baseline = model.predict(features_baseline[p].to_numpy().tolist())
        print('y-predicted baseline', y_prediction_baseline)
        count_baseline_zeroes = y_prediction_baseline.tolist().count(0)
        count_baseline_ones = y_prediction_baseline.tolist().count(1)
        print('Zeros in baseline:', count_baseline_zeroes, 'Ones in baseline:', count_baseline_ones)
        results.append(checklist(y_prediction_baseline))

            # print('Y real=', dataset_y)
            #
            # crisis_onset_index = dataset_y.index(1)
            # crisis_end_index = len(dataset_y) - dataset_y[::-1].index(1) - 1
            #
            # count_crisis_zeroes = y_prediction[crisis_onset_index:crisis_end_index].tolist().count(0)
            # count_crisis_ones = y_prediction[crisis_onset_index:crisis_end_index].tolist().count(1)
            #
            # count_no_crisis_zeroes = y_prediction[:crisis_onset_index].tolist().count(0) + y_prediction[crisis_end_index:].tolist().count(0)
            # count_no_crisis_ones = y_prediction[:crisis_onset_index].tolist().count(1) + y_prediction[crisis_end_index:].tolist().count(1)
            #
            # ratio_crisis = count_crisis_ones / count_crisis_zeroes
            # ratio_no_crisis = count_no_crisis_ones / count_no_crisis_zeroes

            #print("Crisis: ", ratio_crisis)
            #print("No crisis: ", ratio_no_crisis)
    return results



def test_model (patient:int, crisis:int, state:str):

    features= get_patient_hrv_features(patient, crisis)
    features_baseline= get_patient_hrv_baseline_features(patient, state)
    features=get_features_from_patients([patient], [crisis], state)
    dataset_x, dataset_y= create_dataset(features)

    model = src.feature_extraction.io.__load_model(109)
    labels= src.feature_extraction.io.__read_labels(109)

    for p in features.keys():
        for c in features[p].keys():
            for column in features[p][c].columns:
                if column not in labels:
                    features[p][c] = features[p][c].drop(column, axis=1)
            y_prediction = model.predict(features[p][c].to_numpy().tolist())
            print("Y_predicted = ", y_prediction)

            def classification_report_with_f1_score(dataset_x, dataset_y):
                print(classification_report(dataset_x, dataset_y))  # print classification report
                return f1_score(dataset_x, dataset_y)  # return accuracy score

            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(model, dataset_x, dataset_y, scoring=make_scorer(classification_report_with_f1_score),
                                       cv=cv, n_jobs=-1, error_score='raise')
            n_scores_mean = statistics.mean(n_scores)
            n_scores_stdev = statistics.stdev(n_scores)
            print('F1: %.3f (%.3f)' % (n_scores_mean, n_scores_stdev))
            # print classification report
            print(classification_report(dataset_y, y_prediction))
            cm = confusion_matrix(dataset_y, y_prediction, labels=[1, 0])
            print('Confusion matrix : \n', cm)
            tp, fn, fp, tn = confusion_matrix(dataset_y, y_prediction, labels=[1, 0]).reshape(-1)
            print('Outcome values : \n', 'true positive:', tp, 'false negative:', fn, 'false positive:', fp,
                  'true negative:', tn)
            print('False positive rate:', fp / (fp + tn))
    return


def test_model_baseline (patient:int, state:str):

    features=get_baseline_from_patients([patient], state)

    def dataset_baseline (features):

        dataset_x, dataset_y = [], []

        feature_label = ['sampen',
                         'cosen', 'lf', 'hf', 'lf_hf', 'hf_lf', 'sd1', 'sd2',
                         'csi', 'csv', 's', 'rec', 'det', 'lmax', 'nn50', 'pnn50', 'sdnn',
                         'rmssd', 'mean', 'var', 'hr', 'maxhr', 'katz_fractal_dim']

        for patient in features.keys():
            print(features[patient].columns)
            if any(feature in feature_label for feature in features[patient].columns):
                dataset_y = [0] * len(features[patient])
                features[patient].drop(columns=features[patient].columns[0])
                dataset_x = features[patient]
                print('X', dataset_x)
                print('Y', dataset_y)
                assert len(dataset_y) == len(dataset_x)


        return dataset_x, dataset_y





    dataset_x, dataset_y= dataset_baseline(features)

    model = src.feature_extraction.io.__load_model(patient)
    labels= src.feature_extraction.io.__read_labels(patient)

    for p in features.keys():
        for column in features[p].columns:
            print('col', column)
            print('labels', labels)
            if column not in labels:
                features[p] = features[p].drop(column, axis=1)
        y_prediction = model.predict(features[p].to_numpy().tolist())
        print("Y_predicted = ", y_prediction)
        print(len(y_prediction))

        f1=f1_score(dataset_y, y_prediction)
        print('f1score:', f1)
        # print classification report
        print(classification_report(dataset_y, y_prediction))
        cm = confusion_matrix(dataset_y, y_prediction, labels=[1, 0])
        print('Confusion matrix : \n', cm)
        tp, fn, fp, tn = confusion_matrix(dataset_y, y_prediction, labels=[1, 0]).reshape(-1)
        print('Outcome values : \n', 'true positive:', tp, 'false negative:', fn, 'false positive:', fp,
              'true negative:', tn)
        print('False positive rate:', fp / (fp + tn))
    return






