import time
import mlflow
# from mlflow import log_metric, log_param, log_artifacts

# https://medium.com/analytics-vidhya/integrate-mlflow-model-logging-to-scikit-learn-pipeline-6f74e5c601c3

def log_model(search):
    clock_time = time.ctime().replace(' ', '-')
    model_run = str(search.get_params()['estimator__steps'][2][1])
    run_name = clock_time + '_' + model_run

    score_fit_time = search.cv_results_['mean_fit_time'].item()
    score_accuracy = search.cv_results_['mean_test_score_accuracy'].item()
    score_roc_auc = search.cv_results_['mean_test_score_roc_auc'].item()
    score_fbeta = search.cv_results_['mean_test_score_fbeta'].item()

    run_metrics = {
        'score_fit_time': score_fit_time,
        'score_accuracy': score_accuracy,
        'score_roc_auc': score_roc_auc,
        'score_fbeta': score_fbeta
    }

    mlflow.set_experiment(run_name)
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.end_run()
    with mlflow.start_run(run_name=run_name):
        # if not run_params == None:
        #     for name in run_params:
        #         mlflow.log_param(name, run_params[name])
        for name in run_metrics:
            mlflow.log_metric(name, run_metrics[name])

    print('Run is logged')
