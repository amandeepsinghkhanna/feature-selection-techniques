import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from statsmodels.regression.linear_model import OLS
from collections import Counter


class Filter:
    """
    class: Filter
    description: Implementation of the filter methods for feature-selection.

    instance-variables:
    None

    methods:
        1. quasi_constant_filter - Removes Quasi Constant columns (i.e., columns
        that have one value with frequency => threshold) from the input DataFrame.
        2. generate_missing_report - Computes missing frequency, missing % and
        cumulative missing % for all columns of the input DataFrame.
        3. missing_frequency_filter - Generates the missing report for an input
        DataFrame and returns a list of columns to be removed on the basis of the
        threshold for missing %.
        4. compute_vif - Computes and tabulates the Variance Inflation Factor(VIF)
        for all the columns in the input DataFrame.
        5. correlation_filter
        6. filter_chi2
    """

    def __init__(self):
        pass

    @staticmethod
    def quasi_constant_filter(df, threshold=0.99):
        """
        method: quasi_constant_filter
        descripption: Removes Quasi Constant columns from the input DataFrame.

        args:
            1. df - pandas.core.DataFrame
            2. threshold - float - [0.0-1.0]
        """
        columns_to_remove = []
        for col in df.columns:
            modal_frequency = df[col].value_counts(normalize=True).iloc[0]
            if modal_frequency >= threshold:
                columns_to_remove.append(col)
        if len(columns_to_remove) == 0:
            print("No Quasi Constant Columns Found")
        else:
            return columns_to_remove

    @staticmethod  # Add the method to a base class if there is a base class
    def generate_missing_report(df):
        """
        method: generate_missing_report
        description: Generates the missing frequency and cumulative missing frequency of missing values in a dataframe.

        args:
            1. df - pandas.core.DataFrame
        """
        missing_report = df.isnull().sum().reset_index()
        missing_report.columns = ["FEATURE_NAME", "MISSING_FREQUENCY"]
        missing_report["MISSING_PERCENTAGE"] = (
            missing_report["MISSING_FREQUENCY"] / df.shape[0]
        ) * 100
        missing_report = missing_report.sort_values(
            by="MISSING_FREQUENCY", ascending=False
        )
        missing_report["CUMULATIVE_MISSING_PERCENTAGE"] = missing_report[
            "MISSING_PERCENTAGE"
        ].cumsum()
        return missing_report

    def missing_frequency_filter(self, df, threshold=0.10):
        """
        method: missing_frequency_filter
        description: Generates the percent of missing values according to feature name in a dataframe.

        args:
            1. df - pandas.core.DataFrame
            2. threshold - float - [0.0-1.0]
        """
        missing_report = self.generate_missing_report(df)
        columns_to_remove = missing_report.loc[
            missing_report["MISSING_PERCENTAGE"] >= threshold, "FEATURE_NAME"
        ].to_list()
        return columns_to_remove

    @staticmethod
    def compute_vif(df):
        """
        method: compute_vif
        description: Variance Inflation Factor (VIF) is used to detect the severity of multicollinearity in regression analysis.
        """
        vif_results = {}
        for col in df.columns:
            x_features = [c for c in df.columns if c != col]
            r_squared_val = OLS(df[col], df[x_features]).fit().rsquared
            vif_results.append(
                {
                    "FEATURE_NAME": col,
                    "VARIANCE_INFLATION_FACTOR": 1.0 / (1.0 - r_squared_val),
                }
            )
        vif_results = pd.DataFrame(vif_results).sort_values(
            by=["VARIANCE_INFLATION_FACTOR"], ascending=False
        )
        return vif_results

    @staticmethod
    def correlation_filter(df, target=None, threshold=None, type="pearson"):
        """
        method: pearsons_corr_filter
        description: Correlation is a statistical measurement of relationship between two variables.
        """

        def mapper(x):
            if x == 0:
                return "NO_CORRELATION"
            elif (x > 0 and x < 0.3) or (x < 0 and x > -0.3):
                return "NEGLIGIBLE_CORRELATION"
            elif (x >= 0.3 and x < 0.5) or (x <= -0.3 and x > -0.5):
                return "LOW_CORRELATION"
            elif (x >= 0.5 and x < 0.7) or (x <= -0.5 and x > -0.7):
                return "MODERATE_CORRELATION"
            elif (x >= 0.7 and x < 0.9) or (x <= -0.7 and x > -0.9):
                return "HIGH_CORRELATION"
            elif (x >= 0.9 and x < 1) or (x <= -0.9 and x > -1):
                return "VERY_HIGH_CORRELATION"
            elif x == 1:
                return "PERFECT_CORRELATION"

        coorelation_matrix = df.corr(method=type, numeric_only=True)

        if target == None:
            return coorelation_matrix
        else:
            correlations = coorelation_matrix[target].reset_index()
            correlations.columns = ["VARIABLE_NAME", "PEARSONS_CORRELATION_COEFFICIENT"]
            correlations["CORRELATION_INTERPRETATION"] = correlations[
                "PEARSONS_CORRELATION_COEFFICIENT"
            ].apply(lambda x: mapper(x))

            if threshold == None:
                columns_to_remove = correlations.loc[
                    ~(
                        correlations["CORRELATION_INTERPRETATION"].isin(
                            ["NO_CORRELATION", "NEGLIGIBLE_CORRELATION"]
                        )
                    ),
                    "VARIABLE_NAME",
                ].to_list()
            else:
                columns_to_remove = correlations.loc[
                    (correlations["PEARSONS_CORRELATION_COEFFICIENT"] >= threshold)
                    | (
                        correlations["PEARSONS_CORRELATION_COEFFICIENT"]
                        < 0 & correlations["PEARSONS_CORRELATION_COEFFICIENT"]
                        <= -threshold
                    ),
                    "VARIABLE_NAME",
                ].to_list()
            return columns_to_remove

    @staticmethod
    def filter_chi2(df, target):
        """
        method: filter_chi2
        description: Chi-square test is used to measure the difference between the observed and expected frequency of the outcomes of variables. It is mainly used for analysing such differences in categorical variables.
        """
        x_features = [c for c in df.columns if c != target]
        X = df[x_features]
        y = df[target]
        chi2_results = chi2(X, y)
        chi2_results = pd.DataFrame(
            {
                "FEATURE_NAMES": x_features,
                "CHI2_STATISTIC": chi2_results[0],
                "P_VALUE": chi2_results[0],
            }
        )
        chi2_results["INTERPRETATION"] = chi2_results["P_VALUE"].apply(
            lambda x: "DEPENDENT_ON_TARGET" if x < 0.05 else "INDEPENDENT_FROM_TARGET"
        )
        return dict(
            reuslts_table=chi2_results,
            columns_to_remove=chi2_results.loc[
                chi2_results["INTERPRETATION"] == "INDEPENDENT_FROM_TARGET",
                "INTERPRETATION",
            ].to_list(),
        )
    
    @staticmethod
    def fisher_score(x, y):
        """
        method: fisher_score
        description: The main idea behind the Fisher score is to find a subset of features such that the distances between data points in different classes are as large as possible in the data space spanned by the selected features, while the distances between data sets in the same class are as small as possible.
        """
        class_labels, class_sizes = np.unique(y, return_counts = True)
        mu = np.mean(x)
        var = np.var(x)
        inter_class = 0
        intra_class = 0
        for class_label_idx, class_label in enumerate(class_labels):
            class_mu = np.mean(x[(y == class_label)])
            class_var = np.var(x[(y == class_label)])
            inter_class += class_sizes[class_label_idx]*((class_mu - mu)**2)
            intra_class += (class_sizes[class_label_idx] - 1)*class_var
        fisher_score = inter_class/intra_class
        return fisher_score
    
    def fisher_score_ranking(self, df, target, cont_cols = None):
        """
        method: fisher_score_ranking
        description: This is a feature ranking algorithm which ranks features according to fisher score. The larger the fisher score, the more important the feature is.
        """
        fisher_scores = {}
        
        if isinstance(target, str):
            y = df[target].copy()
        elif isinstance(target, pd.core.Series):
            y = target.values.copy()
        elif isinstance(target, np.ndarray):
            y = target.copy()
        else:
            raise TypeError("")
        
        if cont_cols == None:
            cont_cols = list(df.select_dtypes(include="number").columns)
        
        for cont_col in cont_cols:
            fisher_scores[cont_col] = self.fisher_score(
                x = df[cont_col].values,
                y = y
            )

        fisher_scores = (
            pd.Series(fisher_scores, name = "FISHER_SCORE")
            .sort_values(ascending = False)
            .reset_index().rename(columns = {"index": "VARIABLE_NAME"})
            .reset_index().rename(columns = {"index": "FISHER_SCORE_RANKING"})
        )

        fisher_scores["FISHER_SCORE_RANKING"] += 1

        return fisher_scores

     def gini_impurity (classes):
         """
         contributor: Alexei Beuno H
         method: gini_score_filter
         description: Gini score or gini index represents the amount of inequality present in a population. It's value ranges from 0 to 1. 0 represents perfect equality while 1 represents perfect inequality.
         """
         c = Counter(classes)
         p_zero = c[0]/len(classes)
         p_one = c[1]/len(classes)
    
         psum = p_zero ** 2 + p_one ** 2
         gini = 1-psum
         return gini
    
     def gini_score(attribute):
         attribute_values = df[attribute].value_counts()
         gini_A = 0 
         for key in attribute_values.keys():
             df_k = df[attribute][df[attribute] == key].value_counts()
             n_k = attribute_values[key]
             n = df.shape[0]
             gini_A = gini_A + (( n_k / n) * gini_impurity(df_k))
             return gini_A

     gini_attiribute ={}
     for key in df.columns:
         gini_attiribute[key] = gini_score(key)
         print(f'Gini for {key} is {gini_attiribute[key]:.3f}')
