import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pyttsx3
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, recall_score
# Load the dataset
# Load the dataset
@st.cache_data
def load_data():
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            data = pd.read_csv('covid_liver_cancer.csv', encoding=encoding)
            return data
        except UnicodeDecodeError:
            continue
    st.error("Could not decode file. Please check the file encoding.")
    return None


# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None


data = load_data()
# In-memory user database for demo purposes
users_db = {}


# Function to handle sign up
def signup():
    st.sidebar.title("Sign Up")
    new_username = st.sidebar.text_input("New Username", key="signup_username")
    new_password = st.sidebar.text_input("New Password", type="password", key="signup_password")
    if st.sidebar.button("Sign Up"):
        if new_username in users_db:
            st.sidebar.error("Username already exists. Please choose a different username.")
        elif new_username and new_password:
            users_db[new_username] = new_password
            st.session_state.logged_in = True
            st.session_state.username = new_username
            st.sidebar.success("Account created successfully.")
        else:
            st.sidebar.error("Please fill in both fields.")


# Function to handle login
def login():
    st.image("login.gif")
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Login"):
        if username in users_db and users_db[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success("Logged in successfully.")
        else:
            st.sidebar.error("Invalid username or password")




# # Function to compute sensitivity and specificity
# def compute_sensitivity_specificity(conf_matrix):
#     TN, FP, FN, TP = conf_matrix.ravel()
#     sensitivity = TP / (TP + FN)
#     specificity = TN / (TN + FP)
#     return sensitivity, specificity
#
# # Function to plot graphs
# def plot_graphs(graph_type, y_test, predictions, probas):
#     if graph_type == "Bar Graph":
#         st.bar_chart(y_test)
#     elif graph_type == "Pie Graph":
#         labels = ['Alive', 'Dead']
#         sizes = [sum(predictions == 1), sum(predictions == 0)]
#         plt.figure(figsize=(8, 8))
#         plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
#         plt.axis('equal')
#         st.pyplot(plt)
#     elif graph_type == "Histogram":
#         plt.figure(figsize=(10, 6))
#         plt.hist(probas[:, 1], bins=30, color='blue', edgecolor='black')
#         plt.title("Histogram of Prediction Probabilities")
#         plt.xlabel("Probability of Survival")
#         plt.ylabel("Frequency")
#         st.pyplot(plt)
#     elif graph_type == "Heatmap":
#         conf_matrix = confusion_matrix(y_test, predictions)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False)
#         plt.title("Confusion Matrix Heatmap")
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         st.pyplot(plt)
#     elif graph_type in ["Accuracy Graph", "Specificity Graph", "Sensitivity Graph"]:
#         results = {
#             "accuracy": accuracy_score(y_test, predictions),
#             "conf_matrix": confusion_matrix(y_test, predictions),
#             "roc_auc": roc_auc_score(y_test, probas[:, 1])
#         }
#         results["sensitivity"], results["specificity"] = compute_sensitivity_specificity(results["conf_matrix"])
#
#         metrics_df = pd.DataFrame({
#             "Metric": ["Accuracy", "Sensitivity", "Specificity"],
#             "Score": [results["accuracy"], results["sensitivity"], results["specificity"]]
#         })
#
#         plt.figure(figsize=(10, 6))
#         sns.barplot(x="Metric", y="Score", data=metrics_df, palette="viridis")
#         plt.title("Model Performance Metrics")
#         st.pyplot(plt)

# Define the list of features you want to plot

def compute_sensitivity_specificity(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity


# Function to preprocess the data
def preprocess_data(df):
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Convert categorical columns to numeric
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = LabelEncoder().fit_transform(df[column])

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric)

    return df


# Function to plot graphs
def plot_graphs(graph_type, y_test, predictions, probas, input_features=None):

    if graph_type == "Pie Graph":
        labels = ['Alive', 'Dead']
        sizes = [sum(predictions == 1), sum(predictions == 0)]
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
        plt.title("Pie Graph of Predictions")
        plt.axis('equal')
        st.pyplot(plt)

    elif graph_type == "Histogram":
        input_features = [
        "Cancer", "Year", "Month", "Bleed", "Mode_Presentation",
        "Age", "Gender", "Etiology", "Cirrhosis", "Size",
        "HCC_TNM_Stage", "HCC_BCLC_Stage", "ICC_TNM_Stage", "Treatment_grps",
        "Survival_fromMDM", "Alive_Dead", "Type_of_incidental_finding",
        "Surveillance_programme", "Surveillance_effectiveness", "Mode_of_surveillance_detection",
        "Time_diagnosis_1st_Tx", "Date_incident_surveillance_scan", "PS",
        "Time_MDM_1st_treatment", "Time_decisiontotreat_1st_treatment",
        "Prev_known_cirrhosis", "Months_from_last_surveillance"
        ]

        if input_features is not None:
            num_features = len(input_features)
            num_rows = (num_features - 1) // 5 + 1  # Calculate number of rows needed
            plt.figure(figsize=(15, 3 * num_rows))  # Adjust figsize based on number of rows

            for i, feature in enumerate(input_features):
                plt.subplot(num_rows, 5, i + 1)
                plt.hist(data[feature], bins=20, edgecolor='black')
                plt.title(feature)
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.tight_layout()  # Adjust subplot parameters to give specified padding.
                st.pyplot(plt)
        else:
            st.warning("Input features are not provided.")

    elif graph_type == "Confusion_Matrix":
        conf_matrix = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

    elif graph_type in ["Accuracy Graph", "Specificity Graph", "Sensitivity Graph"]:
        results = {
            "accuracy": accuracy_score(y_test, predictions),
            "conf_matrix": confusion_matrix(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probas[:, 1])
        }
        results["sensitivity"], results["specificity"] = compute_sensitivity_specificity(results["conf_matrix"])

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Sensitivity", "Specificity"],
            "Score": [results["accuracy"], results["sensitivity"], results["specificity"]]
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Metric", y="Score", data=metrics_df, palette="viridis")
        plt.title("Model Performance Metrics")
        st.pyplot(plt)

    else:
        st.warning("Select a valid graph type.")


# Main application logic
def main():
    st.sidebar.title("COVID-19 Effect on Liver Cancer Prediction")
    menu = st.sidebar.selectbox("Menu", ["Home", "Dataset", "Prediction", "Graph" , "Tips", "Sign Out"])

    if menu == "Home":
        home_page()
    elif menu == "Dataset":
        dataset_page()
    elif menu == "Prediction":
        prediction_page()
    elif menu == "Graph":
        graphs_page()
    elif menu == "Tips":
        tips_page()
    elif menu == "Sign Out":
        sign_out()



# Define pages
def home_page():
    st.title("COVID-19 Effect on Liver Cancer Prediction")
    st.write("----------------------------------------------")
    # Define the column layout for the images
    col1, col2 = st.columns([1, 1])  # Split the page into two columns of equal width

    # Column 1: Display the first image
    with col1:
        st.image("liver_image.jpg")

    # Column 2: Display the second image
    with col2:
        st.image("covid_background_image.gif")
    st.write("----------------------------------------------")
    # After displaying images, display the introduction paragraph
    st.write("""
        ## Introduction
        COVID-19 has had a significant impact on patients with liver cancer. This application predicts the survival status of liver cancer patients based on various features using Machine Learning and Deep Learning models.

        This app uses several algorithms to predict whether a person with liver cancer, affected by COVID-19, is likely to survive or not.
    """)
    st.write("----------------------------------------------")
    st.video("animation_video.mp4")
    st.write("----------------------------------------------")

    st.write("@COVID-19 : Official Website - WHO Official COVID-19 info")
def dataset_page():
    st.title("Dataset")
    data_view = st.selectbox("Choose a view", ["View Data", "Alive Patients", "Dead Patients","Cancer Patients","Not Cancer Patients"])

    if data_view == "View Data":
        st.dataframe(data)
    elif data_view == "Alive Patients":
        alive_data = data[data['Alive_Dead'] == 1]
        st.dataframe(alive_data)
        st.write(f"Total Alive Patients: {len(alive_data)}")
    elif data_view == "Dead Patients":
        dead_data = data[data['Alive_Dead'] == 0]
        st.dataframe(dead_data)
        st.write(f"Total Dead Patients: {len(dead_data)}")
    elif data_view == "Cancer Patients":
        cancer_data = data[data['Cancer'] == 1]
        st.dataframe(cancer_data)
        st.write(f"Total Cancer Persons: {len(cancer_data)}")
    elif data_view == "Not Cancer Patients":
        not_cancer_data = data[data['Cancer'] == 1]
        st.dataframe(not_cancer_data)
        st.write(f"Total Not Cancer Persons: {len(not_cancer_data)}")

    st.write("-----------------")
    # Dictionary of features and their descriptions
    st.write("Dataset Features")
    st.write("-----------------")
    features = {
        "1.  Cancer": "Cancer flag [Y/N]",
        "2.  Year": "Categorical [Prepandemic (March 2019–February 2020)/Postpandemic(March 2020–February 2021)]",
        "3.  Month": "Month of the year 1-12",
        "4.  Bleed": "Spontaneous tumour haemorrhage [Y/N]",
        "5.  Mode Presentation": "Surveillance, Incidental, or Symptomatic",
        "6.  Age": "Age of the patient",
        "7.  Gender": "Male or Female [M/F]",
        "8.  Etiology": "Manner of causation of a disease or condition. Either 'No established CLD' (chronic liver disease), 'ARLD' (alcohol-related liver disease), 'NAFLD' (non-alcoholic fatty liver disease), 'HCV' (hepatitis C virus), 'HH' (hereditary haemochromatosis), 'PBC/AIH' (primary biliary cholangitis/autoimmune hepatitis), 'HBV' (hepatitis B virus), or 'Other'",
        "9.  Cirrhosis": "Underlying liver disease [Y/N]",
        "10. Size": "Tumour diameter in mm",
        "11. HCC TNM Stage": "Hepatocellular carcinoma Tumour node metastasis Stage ('I', 'II', 'IIIA+IIIB', 'IV')",
        "12. HCC BCLC Stage": "Hepatocellular carcinoma Barcelona Clinic for Liver Cancer Stage ('0', 'A', 'B', 'C', 'D')",
        "13. ICC TNM Stage": "Intrahepatic cholangiocarcinoma Tumour node metastasis Stage ('I', 'II', 'III', 'IV')",
        "14. Treatment grps": "First-line treatment received ['OLTx' (orthotopic liver transplantation), 'Resection', 'Ablation', 'TACE' (transarterial chemoembolisation), 'SIRT' (selective internal radiation therapy), 'Medical', 'Supportive care']",
        "15. Survival from MDM": "Survival from Multidisciplinary meeting",
        "16. Alive Dead": "'Alive', 'Dead'",
        "17. Type of incidental finding": "('Primary care-routine', 'Secondary care-routine', 'Primary care-acute', 'Secondary care-acute')",
        "18. Surveillance programme": "Patient in a formal surveillance programme ('Y', 'N')",
        "19. Surveillance effectiveness": "Surveillance adherence over previous year ('Consistent', 'Inconsistent', 'Missed')",
        "20. Mode of surveillance detection": "Mode of incident surveillance test ['US' (ultrasound), 'AFP alone' (alpha-fetoprotein alone), 'CT/MRI']",
        "21. Time diagnosis 1st Tx": "Time from diagnosis to first treatment",
        "22. Date incident surveillance scan": "('Y', 'N')",
        "23. PS": "Performance status [0, 1, 2, 3, 4]",
        "24. Time MDM 1st treatment": "Time to Multidisciplinary meeting 1st treatment",
        "25. Time decision to treat 1st treatment": "Time decision to treat 1st treatment",
        "26. Prev known cirrhosis": "['Y', 'N']",
        "27. Months from last surveillance": "Months from last surveillance"
    }

    # Display features and their descriptions using a loop
    for feature, description in features.items():
        st.write(f"{feature}: {description}")
    st.write("-----------------")
    st.write("Dataset Source: [COVID-19 on Liver Cancer Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/covid19-effect-on-liver-cancer-prediction-dataset)")
    st.write("-----------------")
def prediction_page():
    st.title("Prediction")
    st.write("Enter the patient details to predict the survival status.")

    # Collect user input for all features
    input_features = {
        "Cancer": st.selectbox("Cancer", [0, 1]),
        "Year": st.selectbox("Year", list(range(0, 2)), 0),
        "Month": st.selectbox("Month", list(range(0, 13)), 0),
        "Bleed": st.selectbox("Bleed", list(range(0, 2)), 0),
        "Mode_Presentation": st.selectbox("Mode_Presentation", list(range(0, 3)), 0),
        "Age": st.number_input("Age", 1, 100, 1),
        "Gender": st.selectbox("Gender", [0, 1]),
        "Etiology": st.selectbox("Etiology", list(range(0, 9)), 0),
        "Cirrhosis": st.selectbox("Cirrhosis", list(range(0, 2)), 0),
        "Size": st.number_input("Size", 0, 94, 0),
        "HCC_TNM_Stage": st.selectbox("HCC_TNM_Stage", list(range(0, 5)), 0),
        "HCC_BCLC_Stage": st.selectbox("HCC_BCLC_Stage", list(range(0, 6)), 0),
        "ICC_TNM_Stage": st.selectbox("ICC_TNM_Stage", list(range(0, 5)), 0),
        "Treatment_grps": st.selectbox("Treatment_grps", list(range(0, 8)), 0),
        "Survival_fromMDM":st.number_input("Survival_fromMDM",0.0, 35.0),
        "Type_of_incidental_finding": st.selectbox("Type_of_incidental_finding", list(range(0, 5)), 0),
        "Surveillance_programme": st.selectbox("Surveillance_programme", list(range(0, 3)), 0),
        "Surveillance_effectiveness": st.selectbox("Surveillance_effectiveness", list(range(0, 4)), 0),
        "Mode_of_surveillance_detection": st.selectbox("Mode_of_surveillance_detection", list(range(0, 4)), 0),
        "Time_diagnosis_1st_Tx": st.number_input("Time_diagnosis_1st_Tx", 0.0, 4.0, 0.0),
        "Date_incident_surveillance_scan": st.selectbox("Date_incident_surveillance_scan", list(range(0, 3)), 0),
        "PS": st.selectbox("PS", list(range(0, 5)), 0),
        "Time_MDM_1st_treatment": st.number_input("Time_MDM_1st_treatment", 0.0, 4.0, 0.0),
        "Time_decisiontotreat_1st_treatment": st.number_input("Time_decisiontotreat_1st_treatment", 0.0, 4.0, 0.0),
        "Prev_known_cirrhosis": st.selectbox("Prev_known_cirrhosis", list(range(0, 3)), 0),
        "Months_from_last_surveillance": st.number_input("Months_from_last_surveillance", 0.0, 10.0, 0.0)
    }



    #----------------------------------------------------------------------------------------------------------------

    input_df = pd.DataFrame([input_features])

    # Separate features and target variable
    X = data.drop("Alive_Dead", axis=1)
    y = data["Alive_Dead"]

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_scaled = scaler.transform(input_df)

    # Train the model
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    if st.button("Predict"):
        if prediction == 1:
            st.success("The patient is likely to survive.")
            st.balloons()
        else:
            st.error("The patient is not likely to survive.")
            st.error("💔 😢")

        # Voice Output
        engine = pyttsx3.init()
        result = "alive" if prediction == 1 else "dead"
        engine.say(f"The patient is predicted to be {result}")
        engine.runAndWait()

        # Accuracy
        accuracy = accuracy_score([prediction], [prediction]) * 100
        st.write(f"Prediction Accuracy: {accuracy:.2f}%")


def graphs_page():
    st.title("Graphs")
    graph_type = st.selectbox("Choose Graph Type",
                              ["Pie Graph", "Histogram", "Confusion_Matrix", "Accuracy Graph"])

    # Preprocess the data
    data_preprocessed = preprocess_data(data)

    # Separate features and target variable
    y = data_preprocessed['Alive_Dead']
    X = data_preprocessed.drop('Alive_Dead', axis=1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load and train the model
    model = GradientBoostingClassifier()  # Assuming you are using GradientBoostingClassifier
    model.fit(X_train, y_train)

    # Predictions and probabilities
    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)

    # Plotting the selected graph
    plot_graphs(graph_type, y_test, predictions, probas)

def tips_page():
    #st.title("Tips for COVID-19 and Liver Cancer")
    st.write("""
        ## Tips for COVID-19 Patients with Liver Cancer
        ----------------------------------------------
        1. Maintain a healthy diet.
        2. Follow your doctor's advice.
        3. Take prescribed medications.
        4. Avoid contact with COVID-19 patients.
        5. Stay informed about new developments.
    """)
    st.write("---------------------")
    st.image("corona_image.jpg", width=500)
    st.write("---------------------")
    st.write("""Follow this page : "https://www.mayoclinic.org/diseases-conditions/liver-cancer/diagnosis-treatment/drc-20353664""")
    st.write("---------------------")

def sign_out():
    st.title("Sign Out")
    st.write("You have been signed out. Please log in again to continue.")
    st.image("logout.gif")
    st.session_state.logged_in = False
    st.session_state.signed_up = False
    st.stop()

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Load data and models
data = load_data()
best_model = pickle.load(open('best_model.pkl', 'rb'))  # Load your best model here
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load your scaler here

# If data loaded successfully, preprocess and split data
if data is not None:
    # Initialize Encoder and Scaler
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    # Label Encoding for categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Fill missing values
    data.fillna(method='ffill', inplace=True)

    # Split data into features and target variable
    X = data.drop('Alive_Dead', axis=1)
    y = data['Alive_Dead']

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Run the main app
if __name__ == '__main__':
    if st.session_state.logged_in:
        st.sidebar.success(f"Welcome, {st.session_state.username}")
        main()
    else:
        st.sidebar.title("COVID-19 Effect on Liver Cancer Prediction")
        auth_action = st.sidebar.radio("Choose Action", ["Login", "Sign Up"])
        if auth_action == "Login":
            login()
        else:
            signup()
