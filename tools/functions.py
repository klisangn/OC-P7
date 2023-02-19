import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, fbeta_score


def missing_values_summary(X, ascending=True):
    '''
    Missing values summary for a given dataframe
    '''
    null = X.isnull().sum()
    not_null = X.notnull().sum()
    percent = X.isnull().sum()/X.isnull().count() * 100
    missing_data = pd.concat([null, not_null, percent], axis=1, keys=['Null_counts', 'Non_Null_counts', 'Percentage_Null'])
    missing_data.reset_index(inplace=True)

    if not ascending:
        missing_data = missing_data.sort_values(by='Percentage_Null', ascending=False)

    return missing_data


def unique_value_counts(df, variable, percent_cumul=True):
    counts = df[variable].value_counts().sort_values(ascending=False).rename_axis('label').reset_index(name='counts')
    counts.loc[:, 'percent'] = counts['counts']/counts['counts'].sum()*100
    if percent_cumul:
        counts.loc[:, 'percent_cumul'] = counts['percent'].cumsum()

    return counts

def pct_null_buckets(x):
    if x <= 30:
        return "<= 30"
    elif x > 30 and x <= 40:
        return "x > 30 and x <= 40"
    elif x > 40 and x <= 50:
        return "x > 40 and x <= 50"
    elif x > 50 and x <= 60:
        return "x > 50 and x <= 60"
    elif x > 60 and x <= 70:
        return "x > 60 and x <= 70"
    elif x > 70 and x <= 80:
        return "x > 70 and x <= 80"
    elif x > 80 and x <= 90:
        return "x > 80 and x <= 90"
    else:
        return "> 90"

def test_scores(search, X_test, y_test):
    search.best_estimator_.fit(X_test, y_test)
    y_test_pred = search.best_estimator_.predict(X_test)

    a = accuracy_score(y_test, y_test_pred)
    f = fbeta_score(y_test, y_test_pred, beta=2)

    print("accuracy_score:", a)
    print("fbeta_score:", f)

    scores_labels = ["accuracy_score", "fbeta_score"]
    scores_values = [a, f]

    return pd.DataFrame({"score": scores_labels, "value": scores_values})
