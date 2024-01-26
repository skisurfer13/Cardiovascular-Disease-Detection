
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# about
st.markdown('''
# Heart Disease Detector 
- This app detects if you have a cardiovascular disease based on Machine Learning!
- Datset: Cleveland and Hungarian heart disease dataset
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.

  Dataset creators:
- Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
''')
st.write('---')

# obtain dataset
df = pd.read_csv(r'heart.csv')

# titles
st.title('Heart Disease Detector')
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(df.describe())

# training
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=0)


# user report
def user_report():
    age = st.sidebar.slider('Age', 0, 90, 55, key='age')
    trestbps = st.sidebar.slider('Resting Blood Pressure', 60, 200, 126, key='trestbps')
    chol = st.sidebar.slider('Cholestrol', 100, 600, 330, key='chol')
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 250, 146, key='thalach')
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 7.5, 2.50, key='oldpeak')
    ca = st.sidebar.slider('Number of major vessels coloured by Flouroscopy', 0, 5, 2, key='ca')

    user_report_data = {
        'Age': age,
        'Resting Blood Pressure': trestbps,
        'Cholestrol': chol,
        'Maximum Heart Rate Achieved': thalach,
        'ST Depression Induced by Exercise': oldpeak,
        'Number of major vessels coloured by Flouroscopy': ca,

    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

st.title('Graphical Patient Report')

if user_result[0] == 0:
    color = 'blue'
else:
    color = 'red'


st.header('Resting Blood Pressure Value Graph (Yours vs Others)')
fig_trestbps = plt.figure()
ax3 = sns.scatterplot(x='Age', y='Resting Blood Pressure', data=df, hue='Outcome', palette='Purples')
ax4 = sns.scatterplot(x=user_data['Age'], y=user_data['Resting Blood Pressure'], s=150, color=color)
plt.xticks(np.arange(0, 100, 5))
plt.yticks(np.arange(60, 220, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_trestbps)


st.header('Cholestrol Value Graph (Yours vs Others)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x='Age', y='Cholestrol', data=df, hue='Outcome', palette='rainbow')
ax10 = sns.scatterplot(x=user_data['Age'], y=user_data['Cholestrol'], s=150, color=color)
plt.xticks(np.arange(0, 100, 5))
plt.yticks(np.arange(100, 750, 50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


st.header('Maximum Heart Rate Achieved Value Graph (Yours vs Others)')
fig_thalach = plt.figure()
ax5 = sns.scatterplot(x='Age', y='Maximum Heart Rate Achieved', data=df, hue='Outcome', palette='Blues')
ax6 = sns.scatterplot(x=user_data['Age'], y=user_data['Maximum Heart Rate Achieved'], s=150, color=color)
plt.xticks(np.arange(0, 100, 5))
plt.yticks(np.arange(40, 250, 25))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_thalach)


st.header('ST Depression Induced by Exercise Value Graph (Yours vs Others)')
fig_oldpeak = plt.figure()
ax11 = sns.scatterplot(x='Age', y='ST Depression Induced by Exercise', data=df, hue='Outcome', palette='Greens')
ax12 = sns.scatterplot(x=user_data['Age'], y=user_data['ST Depression Induced by Exercise'], s=150, color=color)
plt.xticks(np.arange(0, 100, 5))
plt.yticks(np.arange(0, 8, 0.5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_oldpeak)


st.header('Number of major vessels coloured by Flouroscopy Value Graph (Yours vs Others)')
fig_ca = plt.figure()
ax13 = sns.scatterplot(x='Age', y='Number of major vessels coloured by Flouroscopy', data=df, hue='Outcome',
                       palette='rocket')
ax14 = sns.scatterplot(x=user_data['Age'], y=user_data['Number of major vessels coloured by Flouroscopy'], s=150,
                       color=color)
plt.xticks(np.arange(0, 100, 5))
plt.yticks(np.arange(0, 5, 1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_ca)

# Final Report
st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
    output = 'Congratulations, you do not have any heart diseases.'
else:
    output = 'Unfortunately, it is likely that you may be having a heart disease.'
st.title(output)

# Most important for users
st.subheader('Lets raise awareness for cardiovascular health and increase awareness about cardiovascular diseases.')
st.write("World Heart Day: 29 September")


st.write("Dataset License: Creative Commons Attribution 4.0 International (CC BY 4.0)")
st.write("This dataset is also available on the UC Irvine Machine Learning Repository")

st.write(
    "Disclaimer: This is just a learning project based on one particular dataset so please do not depend on it to actually know if you have any cardiovascular diseases or not. It might still be a false positive or false negative. A doctor is still the best fit for the determination of such diseases.")


