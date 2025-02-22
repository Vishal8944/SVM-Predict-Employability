import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import numpy as np

data=pd.read_csv('Student Emp.csv')

le = LabelEncoder()
data['CLASS'] = le.fit_transform(data['CLASS'])
X = data.drop(columns=['Name of Student', 'CLASS'])  # Dropping non-numeric columns
Y = data['CLASS']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=13)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
model=SVC(kernel='linear', random_state=13)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))


st.title('Student Employability Prediction')
st.subheader('Enter Student Details')

G_A=st.slider('General Appearance',1,5,0)
M_S=st.slider('Manner of Speaking',1,5,0)
P_C=st.slider('Physical Condition',1,5,0)
M_A=st.slider('Mental Alertness',1,5,0)
S_C=st.slider('Self confidence',1,5,0)
A_I=st.slider('Abilty to present ideas',1,5,0)
C_S=st.slider('Communication Skills',1,5,0)
S_P=st.slider('Student performance rating',1,5,0)


if st.button('Predict Employability'):
    new_data = np.array([[G_A,M_S,P_C,M_A,S_C,A_I,C_S,S_P]])
    new_data_scaled = sc.transform(new_data)
    prediction = model.predict(new_data_scaled)
    predicted_class = le.inverse_transform(prediction)

    st.subheader('Prediction Result:')
    st.success(f'The student is predicted to be: {predicted_class[0]}')
