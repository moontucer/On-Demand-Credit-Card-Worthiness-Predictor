import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#####################

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from keras.models import Sequential
#from keras.layers import Dense

#####################

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from streamlit_option_menu import option_menu

#####################

path = "/Users/mountasser/Desktop/Data Mining Project/Dataset/Clean_Dataset.csv"

#####################


def main():

    # $ 2. horizontal menu
    selected = option_menu(
    menu_title= "Welcome to The Predictor!", #required
    options = ["Home", "Algorithms", "Predict"], #required
    icons = ["house", "book", "boxes"], #optional
    menu_icon="cast", #optional
    default_index=0, #optional
    orientation= "horizontal")

    if selected == "Algorithms":
        st.title("Credit Card Approval Prediction App")
        st.sidebar.title("Set Parameters for our Binary Classification")
        st.markdown("💳 Will your credit card be approved or rejected?")
        st.markdown("💳 Do the math and find out yourself!")
        st.sidebar.markdown("Will your credit card💳 be approved or rejected?")

        @st.cache(persist=True)
        def load_data():
            data = pd.read_csv(path)
            #Encoding for the categorical values
            #Make it easier for the model to work
            categorical_columns = ['Income_type', 'Education_type', 'Family_status', 'Occupation_type', 'Housing_type']
            label = LabelEncoder()
            for col in categorical_columns:
                data[col] = label.fit_transform(data[col])
            return data

        
        @st.cache(persist=True)
        def split(df):
            y = df.Target
            x = df.drop(columns=['Target'])
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
            return x_train, x_test, y_train, y_test

        #fig, ax = plb.pyplot.subplots()

        def plot_metrics(metrics_list):
            if 'Confusion Matrix' in metrics_list:
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
                st.pyplot()

            if 'ROC Curve' in metrics_list:
                st.subheader("ROC Curve")
                plot_roc_curve(model, x_test, y_test)
                st.pyplot()

            if 'Precision-Recall Curve' in metrics_list:
                st.subheader("Precision-Recall Curve")
                plot_precision_recall_curve(model, x_test, y_test)
                st.pyplot()

        #Disable an annoying warning that kept fucking with me
        st.set_option('deprecation.showPyplotGlobalUse', False)


        df = load_data()
        x_train, x_test, y_train, y_test = split(df)
        class_names = ['Accepted', 'Rejected']
        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "K-Means Clustering", "K-Nearest Neighbors (KNN)", "Naive Bayes", "Decision Tree", "Artificial Neural Network (ANN)"))

        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
            gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Support Vector Machine (SVM) Results")
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)


        if classifier == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')


            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)


        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.number_input("The maximum depth of a tree", 1, 20, step=1, key='max_depth')
            bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')


            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)


        if classifier == 'K-Means Clustering':
            st.sidebar.subheader("Model Hyperparameters")
            n_clusters = st.sidebar.number_input("The number of clusters", 0, 5, step=1, key='n_clusters')


            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("K-Means Clustering Results")
                model = RandomForestClassifier(n_estimators=n_clusters)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)

        
        if classifier == 'K-Nearest Neighbors (KNN)':
            st.sidebar.subheader("Model Hyperparameters")
            n_neighbors = st.sidebar.number_input("The number of neighbors", 2, 10, step=1, key='n_neighbors')

            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("KNN Results")
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)


        if classifier == 'Naive Bayes':
            st.sidebar.subheader("Model Hyperparameters")
            alpha = st.sidebar.number_input("Adjust the Alplha parameter", 0, 10, step=1, key='alpha')

            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Naive Bayes Results")
                model = BernoulliNB(alpha=alpha)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)


        if classifier == 'Decision Tree':
            st.sidebar.subheader("Model Hyperparameters")
            min_samples_leaf = st.sidebar.number_input("The maximum depth of the decision tree", 0, 100, step=10, key='min_samples_leaf')
            max_depth = st.sidebar.number_input("The minimum samples leaf of the decision tree", 0, 10, step=1, key='max_depth')

            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Naive Bayes Results")
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)

        
        if classifier == 'Artificial Neural Network (ANN)':
            st.sidebar.subheader("Model Hyperparameters")
            epochs = st.sidebar.number_input("Number of epochs while training", 10, 50, step=10, key='epochs')

            metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Neural Networks Results")
                # Create a neural network with 2 hidden layers, each with 10 neurons
                model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=42) 
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)


######################################################################


    if selected == "Home":
        raw_data = pd.read_csv(path)
        st.sidebar.title("Checkbox this to show the raw data!")
        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Credit Card Dataset Classification")
            st.write(raw_data)
        st.title("Credit Card Approval Predictor Solution")
        st.markdown("💳 Will your credit card be approved or rejected?")
        st.markdown("💳 This is a project made at The National School for Computer Science and Systems Analysis - ENSIAS, Mohamed V University - UM5")
        image1 = Image.open('/Users/mountasser/Desktop/Projects/NN Project/Images/ENSIAS.png')
        st.image(image1, caption='ENSIAS')


    if selected == "Predict":
        with st.form("my_form"):
            new_row = [] #The ID's already there since it don't matter. Nvm, managed to drop it.
            st.write("Kindly, fill your case data below:")

            gender = st.radio("Gender?", ('Male', 'Female'))
            if gender == 'Male':
                gender = 1
            else:
                gender = 0
            new_row.append(gender)

            def askForBinary(text):
                x = st.radio(text, ('Yes', 'No'))
                if x == 'Yes':
                    x = 1
                else:
                    x = 0
                new_row.append(x)
                return 0

            askForBinary('Own a car?')
            askForBinary('Own a property?')
            askForBinary('Have a work phone?')
            askForBinary('Have a phone?')
            askForBinary('Do you have an Email?')
            askForBinary('Unemployed?')

            children = st.number_input("How many children do you have?", 0, 10, step=1)
            new_row.append(children)
            family = st.number_input("How many family members?", 0, 10, step=1)
            new_row.append(family)
            length = st.slider("How many months since you created an account with your bank?")
            new_row.append(length)
            income = st.number_input("What's your total income?", 0, 100000000, step=10000)
            new_row.append(income)
            age = st.slider("What's your age?")
            new_row.append(age)
            years = st.slider("How many years of employment?")
            new_row.append(years)
            rand = st.selectbox("Income type", ("Working", "Commercial associate", "Pensioner", "State servant", "Student"))
            new_row.append(0)
            rand = st.selectbox("Education type", ("Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"))
            new_row.append(1)
            rand = st.selectbox("Family status", ("Civil marriage", "Married", "Single / not married", "Separated", "Widow"))
            new_row.append(3)
            rand = st.selectbox("Housing type", ("Rented apartment", "House / apartment", "Municipal apartment", "With parents", "Co-op apartment", "Office apartment"))
            new_row.append(1)
            rand = st.selectbox("Occupation type", ('Other', 'Security staff', 'Sales staff', 'Accountants', 'Laborers', 'Managers', 'Drivers', 'Core staff', 'High skill tech staff', 'Cleaning staff', 'Private service staff', 'Cooking staff', 'Low-skill Laborers', 'Medicine staff', 'Secretaries', 'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff'))
            new_row.append(10)

            submitted = st.form_submit_button("Check Eligibility")

            if submitted:


                data = pd.read_csv(path)
                data = data.drop(columns=['ID'])
                #Encoding for the categorical values
                #Make it easier for the model to work
                categorical_columns = ['Income_type', 'Education_type', 'Family_status', 'Occupation_type', 'Housing_type']
                label = LabelEncoder()
                for col in categorical_columns:
                    data[col] = label.fit_transform(data[col])
                y = data.Target
                x = data.drop(columns=['Target'])
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


                model = RandomForestClassifier(n_estimators=50, max_depth=20, bootstrap=True)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                new_predict = model.predict([new_row])
                if new_predict.tolist()[0] == 0:
                    st.title("Ops, this client is not elligible for a credit card!")
                else:
                    st.title("Yay, this client is elligible for a credit card!")
            #st.write("Outside the form")

if __name__ == '__main__':
    main()


