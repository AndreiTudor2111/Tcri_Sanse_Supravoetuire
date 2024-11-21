# app.py
import streamlit as st
import pandas as pd
import joblib

# Funcția de încărcare a modelului și scaler-ului
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    try:
        svm_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return svm_model, scaler
    except Exception as e:
        st.error(f"Eroare la încărcarea modelului sau scaler-ului: {e}")
        return None, None

# Definirea căilor către model și scaler
model_path = r'C:\Users\ostac\Desktop\TCRI\Baza de date\SVM_sanse.pkl'
scaler_path = r'C:\Users\ostac\Desktop\TCRI\Baza de date\SVM_scaler_sanse.pkl'

# Încărcarea modelului și scaler-ului
svm_model, scaler = load_model_and_scaler(model_path, scaler_path)

# Verificare dacă modelul și scaler-ul au fost încărcate cu succes
if svm_model is not None and scaler is not None:
    st.success("Modelul și scaler-ul au fost încărcate cu succes!")
    
    # Funcția de predicție pentru probabilitate
    def predict_survival_probability(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
        # Creează un DataFrame din intrare
        input_data = pd.DataFrame({
            'Pclass': [Pclass],
            'Sex': [Sex],
            'Age': [Age],
            'SibSp': [SibSp],
            'Parch': [Parch],
            'Fare': [Fare],
            'Embarked': [Embarked]
        })
        
        # Scalează datele de intrare folosind scaler-ul încărcat
        try:
            input_data_scaled = scaler.transform(input_data)
        except Exception as e:
            st.error(f"Eroare la scalarea datelor: {e}")
            return None
        
        # Calculează probabilitatea de supraviețuire
        try:
            probabilities = svm_model.predict_proba(input_data_scaled)
            survival_probability = probabilities[0][1] * 100  # Probabilitatea clasei 1
        except Exception as e:
            st.error(f"Eroare la realizarea predicției: {e}")
            return None
        
        # Returnează probabilitatea de supraviețuire
        return survival_probability
    
    # Streamlit app layout
    st.title("Predicție Supraviețuire Titanic")
    
    st.header("Introduceți Detaliile Călătorului")
    
    # Input pentru "Nume"
    name = st.text_input("Introdu numele călătorului")
    
    # Input pentru "Sex"
    sex = st.selectbox("Sex", ("Barbat", "Femeie"))
    sex_encoded = 0 if sex == "Barbat" else 1
    
    # Input pentru "Vârstă"
    age = st.number_input("Vârsta", value=25, min_value=0)
    
    # Input pentru "Numărul de rude la bord" (SibSp)
    sibsp = st.number_input("Numărul de rude la bord", min_value=0, value=0)
    
    # Input pentru "Statut familial" (Parch)
    parch = st.number_input("Numărul de copii/părinți la bord", min_value=0, value=0)
    
    # Input pentru "Clasa biletului" (Pclass)
    clasa = st.selectbox("Clasa biletului", ("Clasa 1", "Clasa 2", "Clasa 3"))
    pclass_encoded = {"Clasa 1": 1, "Clasa 2": 2, "Clasa 3": 3}[clasa]
    
    # Input pentru "Locul de îmbarcare" (Embarked)
    imbarcare = st.selectbox("Locul de îmbarcare", ("Southampton", "Cherbourg", "Queenstown"))
    imbarcare_encoded = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[imbarcare]
    
    # Input pentru "Tariful biletului" (Fare)
    fare = st.number_input("Tariful biletului", value=32.0, min_value=0.0, step=0.1)
    
    # Buton pentru realizarea predicției
    if st.button("Calculează Probabilitatea de Supraviețuire"):
        if name.strip() == "":
            st.warning("Te rog să introduci numele călătorului.")
        else:
            probability = predict_survival_probability(
                Pclass=pclass_encoded,
                Sex=sex_encoded,
                Age=age,
                SibSp=sibsp,
                Parch=parch,
                Fare=fare,
                Embarked=imbarcare_encoded
            )
            if probability is not None:
                st.success(f"{name}, probabilitatea ca să supraviețuiești este de {probability:.2f}%.")
else:
    st.error("Nu s-a putut încărca modelul și scaler-ul. Verifică căile și fișierele.")
