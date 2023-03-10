import os
import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import requests

parent_dir = os.path.abspath('..')

# Data
file_path = os.path.join(parent_dir, "data", "sample.csv")
data = pd.read_csv(file_path)

col1, col2 = st.columns([5,15])
col1.image("img\logo.png", width=170)
col2.text("")
col2.text("")
col2.title('Credit scoring')

# Overview
st.markdown('# I. Customers Overview')

st.markdown('## 1. Key metrics')
metric1 = data.shape[0]
metric2 = int(data.describe()['AMT_CREDIT']['mean'])
metric3 = int(data.describe()['AMT_ANNUITY']['mean'])
metric4 = int(data.describe()['AMT_ANNUITY']['mean']/12)

col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Nb of clients", value=metric1)
col2.metric(label="Avg CREDIT", value=metric2)
col3.metric(label="Avg ANNUITY", value=metric3)
col4.metric(label="Avg Monthly Instalment", value=metric4)
# col1.metric(label="Avg INCOME_CREDIT_PERC", value=data.describe()['INCOME_CREDIT_PERC']['mean'])
# col2.metric(label="Avg ANNUITY_INCOME_PERC", value=data.describe()['ANNUITY_INCOME_PERC']['mean'])
# col3.metric(label="Avg PAYMENT_RATE", value=data.describe()['PAYMENT_RATE']['mean'])
# col4.metric(label="Avg Monthly Instalment", value=data.describe()['AMT_ANNUITY']['mean']/12)

st.markdown('## 2. Dataset')
st.dataframe(data)

# Focus
st.markdown('# II. Focus on customer')

client_ids = list(data['SK_ID_CURR'])
client_id_option = st.selectbox('Select the client ID:', client_ids)

## Client's info
st.markdown("## 1. Customer's information")
x = data[data['SK_ID_CURR'] == client_id_option]
st.dataframe(x)

## Credit status
st.markdown("## 2. Credit status")
st.markdown("### 2.1. Theorical status")

url = 'http://localhost:5000/predict'
params = x.to_json()
r = requests.post(url, json=params)
pred = r.json()

opt_threshold = 0.6
opt_pred = 1 if pred >= opt_threshold else 0

col1, col2, col3 = st.columns(3)
col1.write("")
col1.markdown("**Credit granted**:")
if opt_pred == 1:
    col3.error('No')
else:
    col3.success('Yes')

fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = pred,
    mode = "gauge+number+delta",
    title = {'text': "Credit Score"},
    # delta = {'reference': opt_threshold},
    gauge = {'axis': {'range': [None, 1]},
            'bar': {'color': "dimgray"},
             'steps' : [
                 {'range': [0, opt_threshold], 'color': "darkseagreen"},
                 {'range': [opt_threshold, 1], 'color': "tomato"}], #darksalmon
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': pred}}))
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 2.2. Credit Simulator")
col1, col2 = st.columns(2)

chosen_threshold = col1.slider(label="**Status according to selected threshold**:",  min_value=0.0, max_value=1.0, value=0.6, step=0.1)
threshold_pred = 1 if pred >= chosen_threshold else 0

if threshold_pred == 1:
    col2.error('No credit granted')
else:
    col2.success('Credit granted')

fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = pred,
    mode = "gauge+number+delta",
    title = {'text': "Credit Score"},
    # delta = {'reference': opt_threshold},
    gauge = {'axis': {'range': [None, 1]},
            'bar': {'color': "dimgray"},
             'steps' : [
                 {'range': [0, chosen_threshold], 'color': "darkseagreen"},
                 {'range': [chosen_threshold, 1], 'color': "tomato"}], #darksalmon
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': pred}}))
st.plotly_chart(fig, use_container_width=True)

## Feature importance
st.markdown("## 3. Deep-dive")
st.markdown("### 3.1. Feature importance")

model_path = os.path.join(parent_dir, 'notebooks/mlruns/966063637948665005/a2cd557f418b4b81a6f694c7dbc4d55e/artifacts/model/model.pkl')
model = pickle.load(open(model_path, "rb"))
model = model[0]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)

shap.initjs()
fig = shap.force_plot(explainer.expected_value[0], shap_values[0], x, matplotlib=True, show=False)
plt.savefig('img\shap.png')
st.image("img\shap.png")

y_pred =  model.predict(data)
df = data.copy()
df['class'] = y_pred

st.markdown("### 3.2. Focus on feature")
features = list(df.columns)
features.remove('SK_ID_CURR')
features.remove('class')
features.sort()
features_option = st.selectbox('Select the feature:', features)
client_feature = df[df['SK_ID_CURR'] == client_id_option][features_option].values[0]

st.markdown(f"**Client feature's value** _(in red in the charts below_): **:blue[{client_feature}]**")

col1, col2 = st.columns(2)

fig = plt.figure(figsize=(10, 4))
sns.histplot(df[df['class']==0][features_option], kde=True)
plt.axvline(client_feature, color='red')
col1.pyplot(fig)
col1.caption('Distribution of customers with no payments difficulties :heavy_check_mark:')

fig = plt.figure(figsize=(10, 4))
sns.histplot(df[df['class']==1][features_option], kde=True)
plt.axvline(client_feature, color='red')
col2.pyplot(fig)
col2.caption('Distribution of customers with payments difficulties :o:')

st.markdown("### 3.3. Features comparison")
y_proba =  model.predict_proba(data)
df['score'] = y_proba[:,1]

col1, col2 = st.columns(2)
feature1_option = col1.selectbox('Select the 1st feature:', features)
client_feature1 = df[df['SK_ID_CURR'] == client_id_option][feature1_option].values[0]
col1.markdown(f"**Client 1st feature's value**: **:blue[{client_feature1}]**")

feature2_option = col2.selectbox('Select the 2nd feature:', features)
client_feature2 = df[df['SK_ID_CURR'] == client_id_option][feature2_option].values[0]
col2.markdown(f"**Client 2nd feature's value**: **:blue[{client_feature2}]**")

fig = plt.figure(figsize=(10, 4))
plt.scatter(x=feature1_option, y=feature2_option, data=df, c='score', cmap='RdYlGn_r')
plt.axvline(client_feature1, color='red')
plt.axhline(client_feature2, color='red')
plt.xlabel(feature1_option)
plt.ylabel(feature2_option)
plt.colorbar()
st.pyplot(fig)

st.markdown("### 3.1. Feature importance global")
st.image("img\shap_summary.png", width=650)