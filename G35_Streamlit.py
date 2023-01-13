import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTEN
from scipy.linalg.decomp import inf
import math

data_ori = pd.read_csv("G35_Streamlit.csv")

# Create a page dropdown
page = st.sidebar.selectbox("Hello there! I'll guide you! Please select model",
                            ["Method",
                             "Chi 2: Gaussian Naive Bayes",
                             "Chi 2: Logistic Regression",
                             "Chi 2: Decision Tree Classifier",
                             "ANOVA F-test: Decision Tree Regressor",
                             "ANOVA F-test: Bayesian Regression",
                             "ANOVA F-test: Linear Regression"])

######################################### Function #########################################


def report_metric(accuracy_score, nX_test, cy_test, model_name):
    # Creates report with mae, rmse and r2 metric and returns as df
    pre_score = str(precision_score(cy_test, nX_test, average='macro'))
    rec_score = str(recall_score(cy_test, nX_test, average='macro'))
    f1_s = str(f1_score(cy_test, nX_test, average='macro'))
    metric_data = {'Metric': ['Accuracy Score', 'Precision Score', 'Recall score', 'F1 score'],
                   model_name: [accuracy_score, pre_score, rec_score, f1_s]}
    metric_df = pd.DataFrame(metric_data)
    return metric_df


def report_metric1(prediction, ry_test, model_name):
    # Creates report with mae, rmse and r2 metric and returns as df
    mae = metrics.mean_absolute_error(ry_test, prediction)
    mse = metrics.mean_squared_error(ry_test, prediction)
    rmse = np.sqrt(metrics.mean_squared_error(ry_test, prediction))
    metric_data = {'Metric': ['MAE', 'MSE', 'RMSE'],
                   model_name: [mae, mse, rmse]}
    metric_df = pd.DataFrame(metric_data)
    return metric_df

######################################### Classification #########################################


df_classification = data_ori.copy()
cX = df_classification.drop(["Date", "Time", "Race", "Gender", "Body_Size", "Basket_Size", "Wash_Item", "weather",
                             "With_Kids", "Kids_Category", "Basket_Size_oe"], axis=1)
cy = df_classification['Basket_Size_oe']

# split the X and y into train and test sets
cX_train, cX_test, cy_train, cy_test = train_test_split(
    cX, cy, test_size=0.3, random_state=1)

# feature selection using chi2


def fselect_chi2(cX_train, cy_train, cX_test):
    # k = 5 to extract the top 5 best performing features
    cfselect = SelectKBest(score_func=chi2, k=6)
    cfselect.fit(cX_train, cy_train)
    # transform the new X_train and X_test to the new version
    cnewX_train = cfselect.transform(cX_train)
    cnewX_test = cfselect.transform(cX_test)
    return cnewX_train, cnewX_test, cfselect


cnewX_train, cnewX_test, cfselect = fselect_chi2(cX_train, cy_train, cX_test)
cvector_names = list(cX.columns)

######################################### Regression #########################################

df_regression = data_ori.copy()
rX = df_regression.drop(["Date", "Time", "Race", "Gender", "Body_Size", "Age_Range", "weather",
                         "Basket_Size", "Wash_Item", "With_Kids", "Kids_Category"], axis=1)
ry = df_regression['Age_Range']

# split the X and y to train and test sets
rX_train, rX_test, ry_train, ry_test = train_test_split(
    rX, ry, test_size=0.3, random_state=1)

# feature selection


def fselect_anovaF(rX_train, ry_train, rX_test):
    # k = 5 to extract the top 5 features
    rfselect = SelectKBest(score_func=f_classif, k=5)
    rfselect.fit(rX_train, ry_train)
    # transform the new X_train and X_test sets to the filtered new version
    rnewX_train = rfselect.transform(rX_train)
    rnewX_test = rfselect.transform(rX_test)
    return rnewX_train, rnewX_test, rfselect


rnewX_train, rnewX_test, rfselect = fselect_anovaF(rX_train, ry_train, rX_test)
rvector_names = list(rX.columns)

######################################### Model #########################################

# Prepare for model 1 Gaussian Naive Bayes
# before SMOTEN
nbc = GaussianNB()
nbc.fit(cnewX_train, cy_train)
gau_accuracy_score = str(nbc.score(cnewX_test, cy_test))
gaussian_test = nbc.predict(cnewX_test)
gaussian_metric1 = report_metric(gau_accuracy_score, gaussian_test,
                                 cy_test, "Gaussian Naive Bayes")
# perform SMOTEN
gau_smt = SMOTEN(random_state=42)
gau_smtX_train1, gau_smty_train1 = gau_smt.fit_resample(cnewX_train, cy_train)
# after SMOTEN
nbc = GaussianNB()
nbc.fit(gau_smtX_train1, gau_smty_train1)
gau_acc_score = str(nbc.score(cnewX_test, cy_test))
gaussian_test1 = nbc.predict(cnewX_test)
gaussian_metric2 = report_metric(gau_acc_score, gaussian_test1,
                                 cy_test, "Gaussian Naive Bayes")

# Prepare for model 2 Logistic Regression
# before SMOTEN
lrc = LogisticRegression()
lrc.fit(cnewX_train, cy_train)
reg_accuracy_score = str(lrc.score(cnewX_test, cy_test))
reg_test = lrc.predict(cnewX_test)
reg_metric1 = report_metric(reg_accuracy_score, reg_test,
                            cy_test, "Logistic Regression")
# perform SMOTEN
reg_smt = SMOTEN(random_state=42)
reg_smtX_train1, reg_smty_train1 = reg_smt.fit_resample(cnewX_train, cy_train)
# after SMOTEN
lrc = LogisticRegression()
lrc.fit(reg_smtX_train1, reg_smty_train1)
reg_acc_score = str(lrc.score(cnewX_test, cy_test))
reg_test1 = lrc.predict(cnewX_test)
reg_metric2 = report_metric(reg_acc_score, reg_test1,
                            cy_test, "Logistic Regression")

# Prepare for model 3 Decision Tree Classifier model
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=3)
dtc.fit(cnewX_train, cy_train)
dtc_accuracy_score = str(dtc.score(cnewX_test, cy_test))
dtc_test = dtc.predict(cnewX_test)
dtc_metric1 = report_metric(dtc_accuracy_score, dtc_test,
                            cy_test, "Decision Tree Classifier")
# perform SMOTEN
dtc_smt = SMOTEN(random_state=42)
dtc_smtX_train1, dtc_smty_train1 = dtc_smt.fit_resample(cnewX_train, cy_train)
# after SMOTEN
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=3)
dtc.fit(dtc_smtX_train1, dtc_smty_train1)
dtc_acc_score = str(dtc.score(cnewX_test, cy_test))
dtc_test1 = dtc.predict(cnewX_test)
dtc_metric2 = report_metric(dtc_acc_score, dtc_test1,
                            cy_test, "Decision Tree Classifier")


# # Prepare for model 4 Decision Tree Regressor model
reg_decision_model = DecisionTreeRegressor()
reg_decision_model.fit(rnewX_train, ry_train)
prediction = reg_decision_model.predict(rnewX_test)
reg_decision_metric1 = report_metric1(
    prediction, ry_test, "Decision Tree Regressor")
# after tuning
tuned_hyper_model = DecisionTreeRegressor(max_depth=1,
                                          max_features='log2',
                                          max_leaf_nodes=20,
                                          min_samples_leaf=6,
                                          min_weight_fraction_leaf=0.4,
                                          splitter='best')
tuned_hyper_model.fit(rnewX_train, ry_train)
tuned_pred = tuned_hyper_model.predict(rnewX_test)
reg_decision_metric2 = report_metric1(
    tuned_pred, ry_test, "Decision Tree Regressor")


# # Prepare for model 5 Bayesian Regression model
brr = BayesianRidge()
brr.fit(rnewX_train, ry_train)
brr_m5pred = brr.predict(rnewX_test)
brr_metric1 = report_metric1(brr_m5pred, ry_test, "Bayesian Regression")
# after tuning
param_grid = {'alpha_1': [0.1, 1.0, 10.0],
              'alpha_2': [0.1, 1.0, 10.0],
              'lambda_1': [0.1, 1.0, 10.0],
              'lambda_2': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(brr, param_grid, cv=5)
grid_search.fit(rnewX_train, ry_train)
tuned_pred_brr = grid_search.predict(rnewX_test)
brr_metric2 = report_metric1(tuned_pred_brr, ry_test, "Bayesian Regression")

# # Prepare for model 6 Linear Regression model
lnr = LinearRegression()
lnr.fit(rnewX_train, ry_train)
lnr_m6pred = lnr.predict(rnewX_test)
lnr_metric6 = report_metric1(lnr_m6pred, ry_test, "Linear Regression")

######################################### Display #########################################

if page == "Method":
    # INFO
    st.write("# Feature Selection - Chi2")
    st.write('### What basket size would each customer choose?')
    st.write("We perform Chi2 feature selection to extract the relevant features.")
    # plot the feature scores
    plt.bar([cvector_names[i]
            for i in range(len(cvector_names))], cfselect.scores_)
    y_pos = range(len(cvector_names))
    plt.bar(y_pos, cfselect.scores_)
    plt.xticks(y_pos, cvector_names, rotation=90)
    plt.title('Feature Relevancy Scores')
    plt.xlabel('Features')
    plt.ylabel('Scores')
    st.pyplot(plt)

    # print out all the feature scores
    names = cX.columns.values[cfselect.get_support()]
    scores = cfselect.scores_[cfselect.get_support()]
    names_scores = list(zip(names, scores))
    ns_df = pd.DataFrame(data=names_scores, columns=['Feat_names', 'F_Scores'])
    ns_df_sorted = ns_df.sort_values(
        ['F_Scores', 'Feat_names'], ascending=[False, True])
    st.write(ns_df_sorted)
    st.write("Based from the figure above, we take the top 6 features which are Kids_Category_oe, Race_fac, Wash_Item_fac, With_Kids_oe, Body_Size_oe, and Age_Range and use them for the classification models.")

    plt.clf()
    st.write("# Feature Selection - ANOVA F-test")
    st.write(
        '### What would be the predicted Age_Range of the customers who go to the laundry shop?')
    st.write(
        "We perform ANOVA F-test feature selection to extract the relevant features.")

    # plot the feature scores
    df_q5 = data_ori.copy()

    mapy = df_q5['Age_Range']
    mapX = df_q5.drop(["Date", "Time", "Race", "Gender", "Body_Size", "weather",
                       "Basket_Size", "Wash_Item", "With_Kids", "Kids_Category"], axis=1)
    anova_scores = []

    # split the X and y to train and test sets
    mapX_train, mapX_test, mapy_train, mapy_test = train_test_split(
        mapX, mapy, test_size=0.3, random_state=1)
    for x in mapX:
        eX_train, eX_test, fselect = fselect_anovaF(mapX, mapX[x], mapX_test)
        scores = fselect.scores_
        for i, x in enumerate(scores):
            if math.isinf(x):
                scores[i] = 1
        anova_scores.append(scores)

    anova_array = np.array(anova_scores)
    plt.figure(figsize=(20, 10))
    ax = plt.axes()
    sns.heatmap(anova_array, ax=ax, annot=True, vmax=5, vmin=0)
    st.pyplot(plt)
    plt.clf()

    plt.bar([rvector_names[i]
            for i in range(len(rvector_names))], rfselect.scores_)
    y_pos = range(len(rvector_names))
    plt.bar(y_pos, rfselect.scores_)
    plt.xticks(y_pos, rvector_names, rotation=90)
    plt.title('Feature Relevancy Scores')
    plt.xlabel('Features')
    plt.ylabel('Scores')
    st.pyplot(plt)
    # print out all the feature scores
    names = rX.columns.values[rfselect.get_support()]
    scores = rfselect.scores_[rfselect.get_support()]
    names_scores = list(zip(names, scores))
    ns_df = pd.DataFrame(data=names_scores, columns=['Feat_names', 'F_Scores'])
    ns_df_sorted = ns_df.sort_values(
        ['F_Scores', 'Feat_names'], ascending=[False, True])
    st.write(ns_df_sorted)
    st.write("Based on the figure above, we take the top 5 features which are Basket_Size_oe, With_Kids_oe, Kids_Category_oe, Race_fac, and Wash_Item_fac and use them for the regression models.")
elif page == "Chi 2: Gaussian Naive Bayes":
   # Base model, it uses Gaussian Naive Bayes.
    st.write("## Classif. MOD 1: Gaussian Naive Bayes")
    st.write("Model 1 works with Gaussian Naive Bayes as base model.")
    st.write("The columns it used are: Kids_Category_oe, Race_fac, Wash_Item_fac, With_Kids_oe, Body_Size_oe, and Age_Range")
    st.write("#### Performing Prediction before SMOTEN:")
    st.write(gaussian_metric1)
    st.write('The predicted basket size is '+str(gaussian_test[0]))
    st.write("#### Performing Prediction after SMOTEN:")
    st.write(gaussian_metric2)
    st.write('The predicted basket size is '+str(gaussian_test1[0]))
elif page == "Chi 2: Logistic Regression":
    # Base model, it uses Logistic Regression.
    st.write("## Classif. MOD 2: Logistic Regression")
    st.write("Model 2 works with Logistic Regression as base model.")
    st.write("The columns it used are: Kids_Category_oe, Race_fac, Wash_Item_fac, With_Kids_oe, Body_Size_oe, and Age_Range")
    st.write("#### Performing Prediction before SMOTEN:")
    st.write(reg_metric1)
    st.write('The predicted basket size is '+str(reg_test[0]))
    st.write("#### Performing Prediction after SMOTEN:")
    st.write(reg_metric2)
    st.write('The predicted basket size is '+str(reg_test1[0]))
elif page == "Chi 2: Decision Tree Classifier":
    # Base model, it uses Decision Tree Classifier.
    st.write("## Classif. MOD 3: Decision Tree Classifier")
    st.write("Model 3 works with Decision Tree Classifier as base model.")
    st.write("The columns it used are: Kids_Category_oe, Race_fac, Wash_Item_fac, With_Kids_oe, Body_Size_oe, and Age_Range")
    st.write("#### Performing Prediction before SMOTEN:")
    st.write(dtc_metric1)
    st.write('The predicted basket size is '+str(dtc_test[0]))
    st.write("#### Performing Prediction after SMOTEN:")
    st.write(dtc_metric2)
    st.write('The predicted basket size is '+str(dtc_test1[0]))
elif page == "ANOVA F-test: Decision Tree Regressor":
    # Base model, it uses Decision Tree Regressor.
    st.write("## Regress. MOD 4: Decision Tree Regressor")
    st.write("Model 4 works with Decision Tree Regressor as base model.")
    st.write("The columns it used are: Basket_Size_oe, With_Kids_oe, Kids_Category_oe, Race_fac, Wash_Item_fac, and weather_fac")
    st.write("#### Without hyperparameter tuning:")
    st.write(reg_decision_metric1)
    st.write('Prediction: Age range is ' + str(int(prediction[0])))
    plt.scatter(ry_test, prediction)
    st.pyplot(plt)
    st.write("#### With hyperparameter tuning:")
    st.write(reg_decision_metric2)
    st.write('Prediction: Age range is ' + str(int(tuned_pred[0])))
    plt.scatter(ry_test, tuned_pred)
    st.pyplot(plt)
elif page == "ANOVA F-test: Bayesian Regression":
    # Base model, it uses Bayesian Regression.
    st.write("## Regress. MOD 5: Bayesian Regression")
    st.write("Model 5 works with Bayesian Regression as base model.")
    st.write("The columns it used are: Basket_Size_oe, With_Kids_oe, Kids_Category_oe, Race_fac, and Wash_Item_fac")
    st.write("#### Without hyperparameter tuning:")
    st.write(brr_metric1)
    st.write('Prediction: Age range is ' + str(int(brr_m5pred[0])))
    st.write("#### With hyperparameter tuning:")
    st.write(brr_metric2)
    st.write('Prediction: Age range is ' + str(int(tuned_pred_brr[0])))
elif page == "ANOVA F-test: Linear Regression":
    # Base model, it uses Linear Regression.
    st.write("## Regress. MOD 6: Linear Regression")
    st.write("Model 5 works with Linear Regression as base model.")
    st.write("The columns it used are: Basket_Size_oe, With_Kids_oe, Kids_Category_oe, Race_fac, and Wash_Item_fac")
    st.write(lnr_metric6)
    st.write('Prediction: Age range is ' + str(int(lnr_m6pred[0])))
