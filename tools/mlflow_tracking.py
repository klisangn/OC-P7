import time
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import roc_curve, fbeta_score

def log_model(search, X, y, X_test, y_test):
    clock_time = time.strftime("%H:%M:%S")
    model_run = str(search.get_params()['estimator__steps'][2][1])
    run_name = model_run + '_' + clock_time
    experiment_name = time.strftime("%Y-%m-%d")

    score_fit_time = search.cv_results_['mean_fit_time'].mean()
    score_accuracy = search.cv_results_['mean_test_score_accuracy'].mean()
    score_roc_auc = search.cv_results_['mean_test_score_roc_auc'].mean()
    score_f1 = search.cv_results_['mean_test_score_f1'].mean()
    score_fbeta = search.cv_results_['mean_test_score_fbeta'].mean()
    
    start_time = time.time()
    search.predict(X)
    score_pred_time = time.time() - start_time

    run_metrics = {
        'score_fit_time': score_fit_time,
        'score_accuracy': score_accuracy,
        'score_roc_auc': score_roc_auc,
        'score_f1': score_f1,
        'score_fbeta': score_fbeta,
        'score_pred_time': score_pred_time
    }

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f'Active experiment name "{experiment_name}" - id "{experiment_id}"')
    except :
        mlflow.set_experiment(experiment_name=experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f'Create experiment name "{experiment_name}" - id "{experiment_id}"')
    
    # with mlflow.start_run(run_name=run_name):
    mlflow.start_run(run_name=run_name)
    print(f"Logging run: {run_name}")
    mlflow.set_tracking_uri('http://localhost:5000')

    model = search.best_estimator_
    mlflow.sklearn.log_model(model, "model")
    print(f"- model logged")

    for metric in run_metrics:
        mlflow.log_metric(metric, run_metrics[metric])
        print(f"- {metric} logged")

    for param in search.best_params_:
        mlflow.log_param(param.replace('model__', ''), search.best_params_[param])
        print(f"- {param} logged")
    
    y_pred = search.predict(X)
    fpr, tpr, _ = roc_curve(y, y_pred)
    y_pred_proba = search.predict_proba(X)[::,1]
    fpr, tpr, _ = roc_curve(y,  y_pred_proba)
    fig = plt.figure(figsize=(6, 4));
    plt.plot(fpr,tpr);
    mlflow.log_figure(figure=fig, artifact_file='roc_curve.png');
    print(f"- roc_curve.png logged")

    search.best_estimator_.fit(X_test, y_test)
    y_test_pred = search.best_estimator_.predict(X_test)
    score_fbeta_test = fbeta_score(y_test, y_test_pred, beta=2)
    mlflow.log_metric('score_fbeta_test', score_fbeta_test)
    print(f"- score_fbeta_test logged")

    print(f"Finished logging")
    mlflow.end_run()


def log_model_wo_pipe(search, X, y, X_test, y_test):
    clock_time = time.strftime("%H:%M:%S")
    model_run = str(search.get_params()['estimator__steps'][0][1])
    run_name = model_run + '_' + clock_time
    experiment_name = time.strftime("%Y-%m-%d")

    score_fit_time = search.cv_results_['mean_fit_time'].mean()
    score_accuracy = search.cv_results_['mean_test_score_accuracy'].mean()
    score_roc_auc = search.cv_results_['mean_test_score_roc_auc'].mean()
    score_f1 = search.cv_results_['mean_test_score_f1'].mean()
    score_fbeta = search.cv_results_['mean_test_score_fbeta'].mean()
    
    start_time = time.time()
    search.predict(X)
    score_pred_time = time.time() - start_time

    run_metrics = {
        'score_fit_time': score_fit_time,
        'score_accuracy': score_accuracy,
        'score_roc_auc': score_roc_auc,
        'score_f1': score_f1,
        'score_fbeta': score_fbeta,
        'score_pred_time': score_pred_time
    }

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f'Active experiment name "{experiment_name}" - id "{experiment_id}"')
    except :
        mlflow.set_experiment(experiment_name=experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f'Create experiment name "{experiment_name}" - id "{experiment_id}"')
    
    # with mlflow.start_run(run_name=run_name):
    mlflow.start_run(run_name=run_name)
    print(f"Logging run: {run_name}")
    mlflow.set_tracking_uri('http://localhost:5000')

    model = search.best_estimator_
    mlflow.sklearn.log_model(model, "model")
    print(f"- model logged")

    for metric in run_metrics:
        mlflow.log_metric(metric, run_metrics[metric])
        print(f"- {metric} logged")

    for param in search.best_params_:
        mlflow.log_param(param.replace('model__', ''), search.best_params_[param])
        print(f"- {param} logged")
    
    y_pred = search.predict(X)
    fpr, tpr, _ = roc_curve(y, y_pred)
    y_pred_proba = search.predict_proba(X)[::,1]
    fpr, tpr, _ = roc_curve(y,  y_pred_proba)
    fig = plt.figure(figsize=(6, 4));
    plt.plot(fpr,tpr);
    mlflow.log_figure(figure=fig, artifact_file='roc_curve.png');
    print(f"- roc_curve.png logged")

    search.best_estimator_.fit(X_test, y_test)
    y_test_pred = search.best_estimator_.predict(X_test)
    score_fbeta_test = fbeta_score(y_test, y_test_pred, beta=2)
    mlflow.log_metric('score_fbeta_test', score_fbeta_test)
    print(f"- score_fbeta_test logged")

    print(f"Finished logging")
    mlflow.end_run()