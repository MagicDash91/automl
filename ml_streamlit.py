import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from nltk import word_tokenize
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredCSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

st.title("AutoML Insight: Streamlined Predictive Modeling")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV file:")

def clean_data(df):
    # Remove currency symbols and convert to float
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['value', 'price', 'cost', 'amount']):
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '', regex=False)
                df[col] = df[col].str.replace('£', '', regex=False)
                df[col] = df[col].str.replace('€', '', regex=False)
                df[col] = df[col].replace('[^\d.-]', '', regex=True).astype(float)
    
    # Identify columns to drop based on null percentage
    null_percentage = df.isnull().sum() / len(df)
    columns_to_drop = null_percentage[null_percentage > 0.25].index

    # Also drop columns containing 'id', 'address', 'phone', 'longitude', 'latitude'
    columns_to_drop = columns_to_drop.union(df.columns[df.columns.str.contains('id|address|phone|longitude|latitude', case=False)])

    # Drop identified columns
    df.drop(columns=columns_to_drop, inplace=True)

    # Drop object columns with more than 15 unique values
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > 15:
            df.drop(columns=[col], inplace=True)
    
    # Fill remaining null values with median for numeric columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if null_percentage[col] <= 0.25:
                if df[col].dtype in ['float64', 'int64']:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
    
    # Convert remaining object columns to lowercase
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
    
    return df



# Check if the file is uploaded
if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Show the DataFrame
    st.dataframe(df)
    df = clean_data(df)

    target = st.selectbox("Select Target Variable for Analysis", df.columns)
    #method = st.selectbox("Select Prediction Method", ("Regression", "Classification"))

    algorithm = st.selectbox("Select Algorithm for Prediction", ("Decision Tree", "Random Forest"))
    question = st.text_input("Additional Question you want to ask about the dataset...")

    if st.button("Process"):

        if df[target].dtype in ['float64', 'int64']:
            unique_values = df[target].nunique()

            # If unique values > 20, treat it as regression, else classification
            if unique_values > 20:
                method = "Regression"
            else:
                method = "Classification"
        else:
            # If the target is not numeric, treat it as classification
            method = "Classification"

        # Display the selected method
        st.write(f"Automatically detected method: {method}")

        if method == "Classification":
            st.markdown("<h2 style='text-align: center; color: black;'>Pairplot</h2>", unsafe_allow_html=True)
            pairplot_fig = sns.pairplot(df, hue=target)
            st.pyplot(pairplot_fig)
            pairplot_fig.savefig("pair1.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            import PIL.Image

            img = PIL.Image.open("pair1.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(img)

            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image. explain it by points", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)


        if method == "Regression":
            st.markdown("<h2 style='text-align: center; color: black;'>Pairplot</h2>", unsafe_allow_html=True)
            pairplot_fig = sns.pairplot(df)
            st.pyplot(pairplot_fig)
            pairplot_fig.savefig("pair1.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            import PIL.Image

            img = PIL.Image.open("pair1.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(img)

            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image. explain it by points", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)


        if method == "Classification":
            st.markdown("<h2 style='text-align: center; color: black;'>Countplot Barchart</h2>", unsafe_allow_html=True)

            # Get the names of all columns with data type 'object' (categorical columns) excluding 'Country'
            cat_vars = [col for col in df.select_dtypes(include='object').columns if col != target and df[col].nunique() > 1 and df[col].nunique() <= 10]

            # Create a figure with subplots
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a countplot for the top 10 values of each categorical variable using Seaborn, with hue as 'method'
            for i, var in enumerate(cat_vars):
                top_values = df[var].value_counts().head(10).index
                filtered_df = df.copy()
                filtered_df[var] = df[var].apply(lambda x: x if x in top_values else 'Other')
                
                # Assuming 'method' is a column in df or some categorical value
                sns.countplot(x=var, data=filtered_df, ax=axs[i], hue=target)  # Added hue parameter
                axs[i].set_title(var)
                axs[i].tick_params(axis='x', rotation=90)

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plots using Streamlit
            st.pyplot(fig)
            fig.savefig("count.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            import PIL.Image

            img = PIL.Image.open("count.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(img)

            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)



        if method == "Regression":
            st.markdown("<h2 style='text-align: center; color: black;'>Countplot Barchart</h2>", unsafe_allow_html=True)

            # Get the names of all columns with data type 'object' (categorical columns) excluding 'Country'
            cat_vars = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() > 1 and df[col].nunique() <= 10]

            # Create a figure with subplots
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a countplot for the top 10 values of each categorical variable using Seaborn, with hue as 'method'
            for i, var in enumerate(cat_vars):
                top_values = df[var].value_counts().head(10).index
                filtered_df = df.copy()
                filtered_df[var] = df[var].apply(lambda x: x if x in top_values else 'Other')
                
                # Assuming 'method' is a column in df or some categorical value
                sns.countplot(x=var, data=filtered_df, ax=axs[i])  
                axs[i].set_title(var)
                axs[i].tick_params(axis='x', rotation=90)

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plots using Streamlit
            st.pyplot(fig)
            fig.savefig("count.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            import PIL.Image

            img = PIL.Image.open("count.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(img)

            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)


        
        if method == "Regression":
            st.markdown("<h2 style='text-align: center; color: black;'>Histplot</h2>", unsafe_allow_html=True)
            # Get the names of all columns with data type 'int' or 'float'
            num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns]

            # Create a figure with subplots
            num_cols = len(num_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a histplot for each numeric variable using Seaborn
            for i, var in enumerate(num_vars):
                sns.histplot(df[var], ax=axs[i], kde=True)
                axs[i].set_title(var)
                axs[i].set_xlabel('')

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plots using Streamlit
            st.pyplot(fig)
            fig.savefig("hist.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            img = PIL.Image.open("hist.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)

        if method == "Classification":
            st.markdown("<h2 style='text-align: center; color: black;'>Histplot</h2>", unsafe_allow_html=True)
            # Get the names of all columns with data type 'int' or 'float'
            num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns]

            # Create a figure with subplots
            num_cols = len(num_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a histplot for each numeric variable using Seaborn
            for i, var in enumerate(num_vars):
                sns.histplot(data=df, x=var, hue=target, kde=True, ax=axs[i])
                axs[i].set_title(var)
                axs[i].set_xlabel('')

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plots using Streamlit
            st.pyplot(fig)
            fig.savefig("hist.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            img = PIL.Image.open("hist.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)

        from sklearn import preprocessing
        for col in df.select_dtypes(include=['object']).columns:
    
            # Initialize a LabelEncoder object
            label_encoder = preprocessing.LabelEncoder()
    
            # Fit the encoder to the unique values in the column
            label_encoder.fit(df[col].unique())
    
            # Transform the column using the encoder
            df[col] = label_encoder.transform(df[col])


        # Display Correlation Heatmap
        st.markdown("<h2 style='text-align: center; color: black;'>Correlation Matrix</h2>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(30, 24))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        fig.savefig("correlation_matrix.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

        img = PIL.Image.open("correlation_matrix.png")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
        response.resolve()
        st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
        st.write(response.text)


        X = df.drop(target, axis=1)
        y = df[target]
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

        from scipy import stats
        threshold = 3

        for col in X_train.columns:
            if X_train[col].nunique() > 20:
                # Calculate Z-scores for the column
                z_scores = np.abs(stats.zscore(X_train[col]))

                # Find and remove outliers based on the threshold
                outlier_indices = np.where(z_scores > threshold)[0]
                X_train = X_train.drop(X_train.index[outlier_indices])
                y_train = y_train.drop(y_train.index[outlier_indices])

        
        st.title("Machine Learning Modelling")

        if algorithm == "Decision Tree" and method == "Regression":

            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import GridSearchCV

            # Create a DecisionTreeRegressor object
            dtree = DecisionTreeRegressor()

            # Define the hyperparameters to tune and their values
            param_grid = {
                'max_depth': [4, 6, 8],
                'min_samples_split': [4, 6, 8],
                'min_samples_leaf': [1, 2, 3, 4],
                'random_state' :  [0, 42],
                'max_features': ['auto', 'sqrt', 'log2']
            }

            # Create a GridSearchCV object
            grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='neg_mean_squared_error')

            # Fit the GridSearchCV object to the data
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            dtree = DecisionTreeRegressor(**best_params)
            dtree.fit(X_train, y_train)

            from sklearn import metrics
            from sklearn.metrics import mean_absolute_percentage_error
            import math
            y_pred = dtree.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = math.sqrt(mse)

            st.markdown("<h2 style='text-align: center; color: black;'>Decision Tree Regressor</h2>", unsafe_allow_html=True)

            st.write('MAE is {}'.format(mae)) 
            st.write('MAPE is {}'.format(mape))
            st.write('MSE is {}'.format(mse))
            st.write('R2 score is {}'.format(r2))
            st.write('RMSE score is {}'.format(rmse))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": dtree.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Decision Tree Regressor)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)
            fig.savefig("dtree.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            img = PIL.Image.open("dtree.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)

        if algorithm == "Decision Tree" and method == "Classification":

            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import GridSearchCV

            # Create a DecisionTreeRegressor object
            dtree = DecisionTreeClassifier()

            # Define the hyperparameters to tune and their values
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 3, 4],
                'min_samples_leaf': [1, 2, 3],
                'random_state': [0, 42]
            }

            # Create a GridSearchCV object
            grid_search = GridSearchCV(dtree, param_grid, cv=5)

            # Fit the GridSearchCV object to the data
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            dtree = DecisionTreeClassifier(**best_params)
            dtree.fit(X_train, y_train)

            from sklearn import metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
            import math
            y_pred = dtree.predict(X_test)
            acc = round(accuracy_score(y_test, y_pred)*100 ,2)
            f1 = f1_score(y_test, y_pred, average='micro')
            prec = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')

            st.markdown("<h2 style='text-align: center; color: black;'>Decision Tree Classifier</h2>", unsafe_allow_html=True)

            st.write('Accuracy is {}'.format(acc), "%") 
            st.write('F1 Score is {}'.format(f1)) 
            st.write('Precision is {}'.format(prec)) 
            st.write('Recall is {}'.format(recall)) 
            

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": dtree.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Decision Tree Classifier)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)
            fig.savefig("dtree.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            img = PIL.Image.open("dtree.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)

        if algorithm == "Random Forest" and method == "Regression":

            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import GridSearchCV

            # Create a Random Forest Regressor object
            rf = RandomForestRegressor()

            # Define the hyperparameter grid
            param_grid = {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state' :  [0, 42],
                'max_features': ['auto', 'sqrt']
            }

            # Create a GridSearchCV object
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')

            # Fit the GridSearchCV object to the training data
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            rf = RandomForestRegressor(**best_params)
            rf.fit(X_train, y_train)

            from sklearn import metrics
            from sklearn.metrics import mean_absolute_percentage_error
            import math
            y_pred = rf.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = math.sqrt(mse)

            st.markdown("<h2 style='text-align: center; color: black;'>Random Forest Regressor</h2>", unsafe_allow_html=True)

            st.write('MAE is {}'.format(mae))
            st.write('MAPE is {}'.format(mape))
            st.write('MSE is {}'.format(mse))
            st.write('R2 score is {}'.format(r2))
            st.write('RMSE score is {}'.format(rmse))


            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": rf.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Random Forest Regressor)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)
            fig.savefig("rf.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            img = PIL.Image.open("rf.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)
        
        if algorithm == "Random Forest" and method == "Classification":

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import GridSearchCV

            # Create a DecisionTreeRegressor object
            rfc = RandomForestClassifier(class_weight='balanced')

            # Define the hyperparameters to tune and their values
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 5, 10],
                'max_features': ['sqrt', 'log2', None],
                'random_state': [0, 42]
            }

            # Create a GridSearchCV object
            grid_search = GridSearchCV(rfc, param_grid, cv=5)

            # Fit the GridSearchCV object to the data
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            rfc = RandomForestClassifier(**best_params)
            rfc.fit(X_train, y_train)

            from sklearn import metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
            import math
            y_pred = rfc.predict(X_test)
            acc = round(accuracy_score(y_test, y_pred)*100 ,2)
            f1 = f1_score(y_test, y_pred, average='micro')
            prec = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')

            st.markdown("<h2 style='text-align: center; color: black;'>Random Forest Classifier</h2>", unsafe_allow_html=True)

            st.write('Accuracy is {}'.format(acc), "%") 
            st.write('F1 Score is {}'.format(f1)) 
            st.write('Precision is {}'.format(prec)) 
            st.write('Recall is {}'.format(recall)) 
            

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": rfc.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Random Forest Classifier)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)
            fig.savefig("dtree.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

            img = PIL.Image.open("dtree.png")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.markdown("<h2 style='text-align: center; color: black;'>Google Gemini Response About Data</h2>", unsafe_allow_html=True)
            st.write(response.text)



        os.environ["GOOGLE_API_KEY"] = "AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA"
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.environ["GOOGLE_API_KEY"])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Use .name instead of .filename
        uploaded_file_path = "uploaded_file" + os.path.splitext(uploaded_file.name)[1]

        # Saving the uploaded file locally
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements", encoding="utf8", errors="ignore")
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        document_search = FAISS.from_texts(chunks, embeddings)
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)
        retrieved_texts = " ".join([result.page_content for result in results])
        rag_template = """
        Based on the following retrieved context:
        "{retrieved_texts}"
        
        Answer the question: {question}
        
        Answer:"""
        rag_prompt = PromptTemplate(input_variables=["retrieved_texts", "question"], template=rag_template)
        rag_llm_chain = LLMChain(llm=llm, prompt=rag_prompt)

        rag_response = rag_llm_chain.run(retrieved_texts=retrieved_texts, question=question)

        # Display the LLM response from RAG
        st.markdown("### Answer based on the question")
        st.write(rag_response)



        
        







