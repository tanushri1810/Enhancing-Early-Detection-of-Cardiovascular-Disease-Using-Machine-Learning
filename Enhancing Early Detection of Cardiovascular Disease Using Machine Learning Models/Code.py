import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
st.title('Cardiovascular Disease Risk Predictor')

df = pd.read_csv("HeartDiseaseTrain-Test.csv")

numerical_cols = ['resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']
categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia']

label_encoder = LabelEncoder()

for column in categorical_cols:
    df[column + '_encoded'] = label_encoder.fit_transform(df[column])

df = df.drop(categorical_cols, axis=1)

min_max_scaler = MinMaxScaler()

df[numerical_cols] = min_max_scaler.fit_transform(df[numerical_cols])

def convert_to_multiclass(thalassemia,binary_target):
    if binary_target == 0:
        return 0  # No risk
    elif thalassemia == 0 :
        return 1  # Low risk
    elif thalassemia == 1:
        return 2  # Medium risk
    elif thalassemia ==2:
        return 3  # High risk
    else:
        return 4  

df['multiclass_target'] = df.apply(lambda row: convert_to_multiclass(row['thalassemia_encoded'], row['target']), axis=1)

X = df.drop(['target', 'multiclass_target'], axis=1)

y_multiclass = df['multiclass_target']
y_multiclass = df['multiclass_target']

X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

in1 = st.text_input("Enter Age")
in2 = st.selectbox("Gender",("male", "female"))
in3 = st.selectbox("Chest Pain Type",("asymptomatic", "non-anginal pain","atypical angina","typical angina"))
in4 = st.text_input("Enter Resting Blood Pressure")
in5 = st.text_input("Enter Cholestoral Level")
in6 = st.selectbox("Fasting Blood Sugar",("Lower than 120 mg/ml", "Greater than 120 mg/ml"))
in7 = st.selectbox("Rest ecg",("normal","having ST-T wave abnormality","left ventricular hypertrophy"))
in8 = st.text_input("Enter Max Heart rate")
in9 = st.selectbox("Exercise Induced Angina",("yes","no"))
in10 = st.text_input("Enter oldpeak")
in11 = st.selectbox("Slope",("up","down","flat"))
in12 = st.text_input("Enter vessels colored by flourosopy encoded")
in13 = st.selectbox("Thalassemia",("normal", "reversible defect","fixed defect","no"))

if st.button("Predict"):
    user_input = {
    'age': in1,                          
    'sex_encoded': in2,                      
    'chest_pain_type_encoded': in3,  
    'resting_blood_pressure': in4,      
    'cholestoral': in5,                 
    'fasting_blood_sugar_encoded': in6,        
    'rest_ecg_encoded': in7,               
    'Max_heart_rate': in8,              
    'exercise_induced_angina_encoded': in9,    
    'oldpeak': in10,                     
    'slope_encoded': in11,                           
    'vessels_colored_by_flourosopy_encoded': in12, 
    'thalassemia_encoded': in13             
}
        
    user_input['sex_encoded'] = 1 if user_input['sex_encoded'] == 'male' else 0

    chest_pain_mapping = {
    'asymptomatic': 0,
    'non-anginal pain': 1,
    'atypical angina': 2,
    'typical angina': 3
}
    user_input['chest_pain_type_encoded'] = chest_pain_mapping.get(user_input['chest_pain_type_encoded'], -1)

    user_input['fasting_blood_sugar_encoded'] = 0 if user_input['fasting_blood_sugar_encoded'] == 'Lower than 120 mg/ml' else 1

    rest_ecg_mapping = {
    'normal': 0,
    'having ST-T wave abnormality': 1,
    'left ventricular hypertrophy': 2
}
    user_input['rest_ecg_encoded'] = rest_ecg_mapping.get(user_input['rest_ecg_encoded'], -1)

    user_input['exercise_induced_angina_encoded'] = 1 if user_input['exercise_induced_angina_encoded'] == 'yes' else 0

    slope_mapping = {
    'up': 0,
    'flat': 1,
    'down': 2
}
    user_input['slope_encoded'] = slope_mapping.get(user_input['slope_encoded'], -1)

    thalassemia_mapping = {
    'normal': 0,
    'fixed defect': 1,
    'reversible defect': 2,
    'no':0
}
    user_input['thalassemia_encoded'] = thalassemia_mapping.get(user_input['thalassemia_encoded'], -1)
    user_df = pd.DataFrame(user_input, index=[0])

    user_df = user_df[X_train.columns]
    prediction = decision_tree.predict(user_df)

    risk_mapping = {
    0: "No risk",
    1: "Low risk",
    2: "Medium risk",
    3: "High risk",
    4: "Very high risk"
}
    predicted_risk = risk_mapping.get(prediction[0], "Unknown risk")

    st.text("The predicted risk category is:")
    st.text(predicted_risk)