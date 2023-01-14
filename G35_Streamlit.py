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
import random
from scipy.stats import chi2_contingency
from kmodes.kmodes import KModes
from apyori import apriori

df = pd.read_csv("dataset.csv")
external_weather_dataset = pd.read_csv("external_weather_dataset.csv")
data_ori = pd.read_csv("G35_Streamlit.csv")

# Create a page dropdown
page = st.sidebar.selectbox("-",
                            ["Exploratory Data Analysis",
                             "Feature Selection",
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
if page == "Exploratory Data Analysis":
    # EDA
    st.write("# Project")
    st.write("Group members:")
    st.write('(1) 1191302247 Tan Yi Chuan')
    st.write("(2) 1191302309 Tan Yee Chuan")
    st.write("(3) 1191302166 Cheok Yi Xuan")
    st.write("(4) 1201302035 Chuah JieYi")
    external_weather_dataset['weather'].value_counts(
        normalize=True, dropna=False)
    df = pd.concat([df, external_weather_dataset], axis=1)
    st.write("# Exploratory Data Analysis and Data Pre-Processing")
    st.write("### Remove Uninterested Attributes")
    # 1. Make a copy of the original dataset
    df_temp = df.copy()
    # 2. Drop all the columns that not in our interests
    df_temp.drop(["Basket_colour", "Attire", "Shirt_Colour", "shirt_type", "Pants_Colour", "pants_type",
                 "Spectacles", "buyDrinks", "latitude", "longitude", "Num_of_Baskets"], axis=1, inplace=True)
    st.write("Check Duplicate Elements in Each Column")
    columns = ["Race", "Gender", "Body_Size", "Basket_Size",
               "Wash_Item", "Washer_No", "Dryer_No", "With_Kids", "Kids_Category"]
    # for column in columns:
    #     st.write(column, df_temp[column].unique())
    st.write(df_temp["Kids_Category"].unique())
    df_temp2 = df_temp.copy()
    df_temp.drop(['Kids_Category'], axis=1, inplace=True)
    df_temp2 = df_temp2['Kids_Category'].replace(['toddler '], 'toddler')
    df_temp = pd.concat([df_temp, df_temp2], axis=1)
    df_temp['Kids_Category'].unique()
    st.write(
        'We noticed that there are TWO ways of categorizing Kids_Category as toddler:')
    st.write('* "toddler "')
    st.write('* "toddler"')
    st.write(
        'Hence, we replaced "toddler " with "toddler" to maintain the data consistency.')
    st.write("### Detecting and Handling Missing values")
    st.write('Check missing data')
    st.write("Total number of missing data :",
             len(df_temp[df.isna().any(axis=1)]))
    st.write(df_temp.shape)
    dfCopy = df_temp.copy()
    # 1. Replace missing values with mode for all the categorical columns
    columns_categorical = ["Race", "Gender", "Body_Size",
                           "Basket_Size", "Wash_Item", "Washer_No", "Dryer_No", "weather"]
    for column in columns_categorical:
        dfCopy[column] = dfCopy[column].fillna(dfCopy[column].mode()[0])

    # 2. Replace missing values with mean for all the numerical columns
    columns_numerical = ["Age_Range", "TimeSpent_minutes", "TotalSpent_RM"]
    for column in columns_numerical:
        dfCopy[column] = dfCopy[column].fillna(round(dfCopy[column].mean()))

    # 3. Handle missing values in ["With_Kids"] and ["Kids_Category"] columns
    df_filter1 = dfCopy[["With_Kids", "Kids_Category"]].copy()
    for index, row in df_filter1.iterrows():
        if row['Kids_Category'] is np.nan and row['With_Kids'] is np.nan:
            df_filter1.loc[index, 'With_Kids'] = dfCopy['With_Kids'].mode()[0]
    dfCopy.drop(['Kids_Category', 'With_Kids'], axis=1, inplace=True)
    dfCopy = pd.concat([dfCopy, df_filter1], axis=1)

    def randKidCategory(x):
        if x == 1:
            return "baby"
        elif x == 2:
            return "toddler"
        elif x == 3:
            return "young"

    df_filter2 = dfCopy[["With_Kids", "Kids_Category"]].copy()
    for index, row in df_filter2.iterrows():
        if row['With_Kids'] == "no":
            df_filter2.loc[index, "Kids_Category"] = "no_kids"
        elif row['With_Kids'] == "yes" and row['Kids_Category'] is np.nan:
            df_filter2.loc[index, "Kids_Category"] = randKidCategory(
                random.randint(1, 3))
        elif row['Kids_Category'] == "no_kids":
            df_filter2.loc[index, "With_Kids"] = "no"
        elif row['Kids_Category'] != "no_kids":
            df_filter2.loc[index, "With_Kids"] = "yes"

    dfCopy.drop(['Kids_Category', 'With_Kids'], axis=1, inplace=True)
    dfCopy = pd.concat([dfCopy, df_filter2], axis=1)
    st.write('We imputed the missing values in "Race", "Gender", "Body_Size", "Basket_Size", "Wash_Item", "Washer_No", and "Dryer_No" columns with the mode of each respective column.')
    st.write('We imputed the missing values in "Age_Range", "TimeSpent_minutes", and "TotalSpent_RM" columns with the mean of each respective column.')
    st.write('We imputed the missing values in "Kids_Category" column as "no_kids" with the condition where "With_Kids" column == "no". If the "With_Kids" column == "yes" but the "Kids_Category" column has no value, we will random assign the value with either "baby", "toddler", or "young".')
    st.write('Recheck missing values')
    st.write("Total number of missing data :",
             len(dfCopy[dfCopy.isna().any(axis=1)]))
    st.write("### Encoding the Categorical Variables")
    st.write("*Label encoding using pandas.factorize() method*")
    st.write("* Race      => 0: malay, 1: indian, 2: chinese, 3: foreigner")
    st.write("* Gender    => 0: male, 1: female")
    st.write("* Wash_Item => 0: clothes, 1: blankets")
    st.write("* weather => 0: rainy, 1: sunny")
    st.write("*Ordinal encoding (assign a number to the ordered categories)*")
    st.write("* Body_Size     => 0: thin, 1: moderate, 2: fat")
    st.write("* Basket_Size   => 0: small, 1: big")
    st.write("* Kids_Category => 0: no_kids, 1: baby, 2: toddler, 3: young")
    st.write("* With_Kids     => 0: no, 1: yes")
    df_fac = dfCopy.copy()
    df_fac['Race_fac'] = pd.factorize(df_fac['Race'])[0].reshape(-1, 1)
    df_fac['Gender_fac'] = pd.factorize(df_fac['Gender'])[0].reshape(-1, 1)
    df_fac['Wash_Item_fac'] = pd.factorize(
        df_fac['Wash_Item'])[0].reshape(-1, 1)
    df_fac['weather_fac'] = pd.factorize(df_fac['weather'])[0].reshape(-1, 1)
    df_oe = df_fac.copy()
    Body_Size_dict = {'thin': 0, 'moderate': 1, 'fat': 2}
    Basket_Size_dict = {'small': 0, 'big': 1}
    Kids_Category_dict = {'no_kids': 0, 'baby': 1, 'toddler': 2, 'young': 3}
    With_Kids_dict = {'no': 0, 'yes': 1}
    df_oe['Body_Size_oe'] = df_oe.Body_Size.map(Body_Size_dict)
    df_oe['Basket_Size_oe'] = df_oe.Basket_Size.map(Basket_Size_dict)
    df_oe['Kids_Category_oe'] = df_oe.Kids_Category.map(Kids_Category_dict)
    df_oe['With_Kids_oe'] = df_oe.With_Kids.map(With_Kids_dict)
    dfCopy = df_oe
    st.write(dfCopy)
    st.write("### Detecting and Handling Outliers")
    st.write('Detecting the outliers')
    fig, ax = plt.subplots(figsize=(8, 3))
    ax = sns.boxplot(x=dfCopy["Age_Range"])
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax = sns.boxplot(x=dfCopy["TimeSpent_minutes"])
    st.pyplot(fig)
    st.write(
        'There are no outliers for the numerical attributes "Age_Range" and "TimeSpent_minutes".')
    st.write('### Is there any relationship between basket size and race?')
    myField1 = dfCopy['Basket_Size_oe']
    myField2 = dfCopy['Race_fac']

    gp = pd.crosstab(myField2, myField1)
    barplot = gp.plot.bar(rot=0)
    st.pyplot(barplot.figure)
    chisq = pd.crosstab(dfCopy['Basket_Size_oe'], dfCopy['Race_fac'])
    c, p, dof, expected = chi2_contingency(chisq)
    st.write("P-value:", p)
    st.write("The two hypotheses for the test are as follows:")
    st.write("* 𝐻0 : There is no relationship between basket size and race.")
    st.write("* 𝐻1 : There is a relationship between basket size and race.")
    st.write("Since the p-value for basket size and race is less than .05, we can reject the null hypothesis of the test. We have sufficient evidence to say that there is a relationship between basket size and race.")
    st.write(
        '### What types of customers will likely to choose Washer No. 4 and Dryer No. 10?')
    st.write(
        "We take the washer number and dryer number from the cluster computed which is cluster 0")
    df_q2 = dfCopy[dfCopy['Washer_No'] == 4]
    df_q2 = df_q2[df_q2['Dryer_No'] == 10]
    df_q2 = df_q2[['Race', 'Gender', 'Body_Size', 'With_Kids', 'Age_Range']]
    st.write(df_q2.mode())
    columns = ['Race', 'Gender', 'Body_Size', 'With_Kids']
    allFeature = []
    for i in columns:
        val, cnt = np.unique(df_q2[i], return_counts=True)
        data = {'Value': val, 'Count': cnt}
        temp = pd.DataFrame(data)
        allFeature.append(temp)
    for i in range(len(allFeature)):
        plt.figure()
        plt.bar(allFeature[i]['Value'], allFeature[i]['Count'])
        plt.title(columns[i])
        st.pyplot(plt)
    plt.figure()
    ax = plt.axes()
    sns.distplot(dfCopy['Age_Range'], kde=False)
    st.pyplot(plt)
    st.write('In conclusion, based on the figures above, it would seem that the majority of the customers are: Malay females of around 39 years old, whose body size are fat and do not have kids.')
    st.write('### Did weather information impact the sales?')
    df_q3 = dfCopy.copy()

    weather_sales = df_q3.groupby(df_q3['weather'])[
        'TotalSpent_RM'].sum().reset_index(name='Total_Sales')
    st.write(weather_sales)
    df_q3.groupby(df_q3['weather'])['TotalSpent_RM'].sum().reset_index(
        name='Total_Sales').plot(kind='bar', y='Total_Sales')
    plt.title('Weather with Total Sales')
    plt.xlabel('Weather')
    plt.ylabel('Total Sales')
    st.pyplot(plt)
    st.write("Based on the graph above, we can see that the total sales on sunny days are significantly higher than on rainy days. Therefore, we can say that weather does impact sales.")
    st.write('### Extra Findings to Support Conclusion')
    st.write('We transform the data of time into a new column, "Parts_Of_Day":')
    st.write("* Morning : 00:00:00 - 11:59:59")
    st.write("* Afternoon : 12:00:00 - 17:59:59")
    st.write("* Evening : 18:00:00 - 23:59:59")
    # Get the hour from Time attribute
    df_q3 = dfCopy.copy()
    df_q3['Date'] = pd.to_datetime(df_q3['Date'])

    # Change time data from '15:47;02' to '15:47:02'
    replace_time = df_q3.copy()
    df_q3.drop(['Time'], axis=1, inplace=True)
    replace_time = replace_time['Time'].replace(['15:47;02'], '15:47:02')
    df_q3 = pd.concat([df_q3, replace_time], axis=1)

    # # change time data from '15:52;08' to '15:52:08'
    replace_time = df_q3.copy()
    df_q3.drop(['Time'], axis=1, inplace=True)
    replace_time = replace_time['Time'].replace(['15:52;08'], '15:52:08')
    df_q3 = pd.concat([df_q3, replace_time], axis=1)

    # create a new variable "Parts_Of_Day" to define "Morning", "Afternoon", an "Evening"
    df_q3['Time'] = pd.to_datetime(df_q3['Time'], format='%H:%M:%S')
    df_q3['Hour_Of_Day'] = df_q3['Time'].dt.hour
    df_q3["Parts_Of_Day"] = pd.cut(
        df_q3["Hour_Of_Day"], bins=[-1, 12, 18, 24], labels=["Morning", "Afternoon", "Evening"])

    df_hour = pd.Series([val.time() for val in df_q3['Time']])
    df_mae = pd.DataFrame({"Date": df_q3.Date, "Time": df_hour,
                          "Hour_Of_Day": df_q3.Hour_Of_Day, "Parts_Of_Day": df_q3.Parts_Of_Day})
    st.write("see when do most ppl go to do laundry. Do for each race/gender.")
    st.write(df_q3.groupby(["Parts_Of_Day", "Race"]).agg({"Race": "count"}))
    st.write(df_q3.groupby(["Parts_Of_Day", "Gender"]).agg(
        {"Gender": "count"}))
    # transform into dataframe
    grouper1 = df_q3.groupby(["Parts_Of_Day", "Race"]).agg({"Race": "count"})
    test_race = grouper1['Race'].to_frame(name='Frequency').reset_index()

    st.write('Plotting grouped bar plot by Race')
    test_race.pivot("Parts_Of_Day", "Race", "Frequency").plot(
        kind='bar', figsize=(8, 6))
    plt.title('Frequency by Parts of Day and Race')
    plt.xlabel('Parts of Day')
    plt.ylabel('Frequency')
    plt.xticks(rotation=360)
    st.pyplot(plt)
    # transform into dataframe
    grouper2 = df_q3.groupby(["Parts_Of_Day", "Gender"]
                             ).agg({"Gender": "count"})
    test_gender = grouper2['Gender'].to_frame(name='Frequency').reset_index()

    st.write('Plotting grouped bar plot by Gender')
    test_gender.pivot("Parts_Of_Day", "Gender", "Frequency").plot(
        kind='bar', figsize=(8, 6))
    plt.title('Frequency by Parts of Day and Gender')
    plt.xlabel('Parts of Day')
    plt.ylabel('Frequency')
    plt.xticks(rotation=360)
    st.pyplot(plt)
    st.write('There is a trend that regardless of race and gender, customers tend to do their laundry in the morning, and then there is quite a similar number of customers doing laundry in the evening and the afternoon.')
    st.write('We can assume that customers seldom go to the laundromat in the afternoon since this is the hottest time of day and the sun is highest in the sky, and assuming that customers seldom go to the laundromat in the evening, probably because most people need to rest at that time.')
    st.write('The weather is not so hot in the morning, so the customers choose to wash their clothes in the laundry shop.')
    st.write('Therefore, it can be concluded that the weather will impact the sales.')
    plt.clf()
    st.write(
        '### Which Washer_No and Dryer_No will most likely be used to wash Wash_Item?')
    st.write('**K-Modes Clustering Technique**')
    st.write('Identify the optimal K value using Elbow curve method')
    df_q6 = dfCopy.copy()
    df_cluster = df_q6[['Washer_No', 'Dryer_No', 'Wash_Item']]

    # Elbow curve to find optimal K
    cost = []
    K = range(1, 10)
    for num_clusters in list(K):
        kmode = KModes(n_clusters=num_clusters,
                       init="random", n_init=5, verbose=1)
        kmode.fit_predict(df_cluster)
        cost.append(kmode.cost_)

    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal K')
    st.pyplot(plt)
    st.write('Based on the graph above, we can see a bend at K = 3, indicating 3 is the optimal number of clusters.')
    st.write('Building the model with 3 number of clusters')
    kmode = KModes(n_clusters=3, init="random", n_init=5, verbose=1)
    clusters = kmode.fit_predict(df_cluster)
    plt.clf()
    st.write('Finally, insert the predicted cluster values in our original dataset.')
    df_cluster.insert(0, "Cluster", clusters, True)
    df_cluster = df_cluster.loc[:, ~df_cluster.columns.duplicated()]
    st.write(df_cluster)
    st.write('Get the cluster centroids (the middle of a cluster)')
    df_clusterCentroids = pd.DataFrame(kmode.cluster_centroids_)
    df_clusterCentroids = df_clusterCentroids.reset_index()
    df_clusterCentroids = df_clusterCentroids.rename(
        columns={"index": "Cluster", 0: "Washer_No", 1: "Dryer_No", 2: "Wash_Item"})
    st.write(df_clusterCentroids)
    st.write(
        "We used K-Modes as our clustering techniques to cluster categorical variables.")
    st.write("Dataframe above showed the cluster centroids for each cluster.")
    st.write("It shows the combination / the cluster of specific Washer_No and Dryer_No that are most likely to be picked by the customers to wash each different types of Wash_Item.")
    st.write('Blankets')
    blankets = df_cluster.loc[(df_cluster['Wash_Item'] == "blankets")]
    blankets = blankets[["Wash_Item", "Cluster", "Washer_No"]]
    group1 = blankets.groupby(["Wash_Item", "Cluster"]).count()
    df_blankets = group1['Washer_No'].to_frame(name='Frequency').reset_index()
    st.write(df_blankets)
    st.write('Clothes')
    clothes = df_cluster.loc[(df_cluster['Wash_Item'] == "clothes")]
    clothes = clothes[["Wash_Item", "Cluster", "Washer_No"]]
    group2 = clothes.groupby(["Wash_Item", "Cluster"]).count()
    df_clothes = group2['Washer_No'].to_frame(name='Frequency').reset_index()
    st.write(df_clothes)
    # Combine both dataframe
    df_wash_items = pd.concat([df_blankets, df_clothes], axis=0)
    st.write('Plotting grouped bar chart by Wash_Item')
    df_wash_items.pivot("Cluster", "Wash_Item", "Frequency").plot(
        kind='bar', figsize=(6, 6))
    plt.title('Frequency by Cluster No. and Wash Item')
    plt.xlabel('Cluster No.')
    plt.ylabel('Frequency')
    plt.xticks(rotation=360)
    st.pyplot(plt)
    st.write('Since there is no attribute in the dataset given explaining about the weight that a Washer_No or a Dryer_No can hold, we assumed that in order to wash different types of Wash_Item, different sizes of Washer_No and Dryer_No were used, both holding a specific weight ranging from light to heavy.')
    st.write('Based on the dataframe above, Cluster 0 has the highest count for washing blankets, while Cluster 1 has the highest count for washing clothes.')
    st.write('Some possible solutions to owner would be:')
    st.write('* Increase the number of big size washers and dryers in Cluster 0 because customers tend to use these Washer_No and Dryer_No to wash blankets.')
    st.write('* Increase the number of small size washers and dryers in Cluster 1 because customers tend to use these Washer_No and Dryer_No to wash clothes.')
    st.write('### Extra Findings to Support Conclusion')
    st.write('Since that customers tend to use washers and dryers in Cluster 0 to wash big size Wash_Item like blankets, we assume that customers are washing heavy blankets and large amounts of clothes if they are bringing a bigger basket size.')
    washer = df.groupby(['Basket_Size', "Washer_No"]
                        ).agg({"Wash_Item": "count"})
    dryer = df.groupby(['Basket_Size', "Dryer_No"]).agg({"Wash_Item": "count"})
    st.write(washer)
    st.write(dryer)
    st.write("It is shown that most customers who used Washer_No and Dryer_No from Cluster 0 tend to bring bigger basket.")
    st.write("Hence, we can confirm that the Washer_No and Dryer_No from Cluster 0 can support heavy weight wash like blankets.")
    st.write(
        '### Which combination of Washer_No and Dryer_No is frequently used by customers?')
    st.write('**Apriori Association Rule Mining**')
    df_q7 = dfCopy.copy()
    df_arm = df_q7[["Washer_No", "Dryer_No"]]

    records = []
    for i in range(0, 4000):
        records.append([str(df_arm.values[i, j]) for j in range(0, 2)])
    st.write(df_arm)
    association_rules = apriori(
        records, min_support=0.0025, min_confidence=0.2, min_lift=0.7, min_length=2)
    association_results = list(association_rules)

    # # filter array so that they have 2 elements
    association_results = association_results[8:]
    cnt = 0

    for item in association_results:
        cnt += 1
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        st.write("(Rule " + str(cnt) + ") Washer_No " +
                 items[0] + " -> Dryer_No " + items[1])

        # second index of the inner list
        st.write("Support: " + str(round(item[1], 3)))

        # third index of the list located at 0th
        # of the third index of the inner list
        st.write("Confidence: " + str(round(item[2][0][2], 4)))
        st.write("Lift: " + str(round(item[2][0][3], 4)))
        st.write("=====================================")
    st.write('Based on the results, we can see that Rule 5 has the highest value of Support, Confidence and Lift. Therefore, we can conclude that the most used combination of washer and dryer is Washer_No 3 and Dryer_No 7.')
elif page == "Feature Selection":
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
