import pandas as pd
from statsmodels.regression.linear_model import OLS


class Filter:
    """
    class: Filter
    description: Implementation of the filter methods for feature-selection.
    """

    def __init__(self):
        pass

    @staticmethod
    def quasi_constant_filter(df, threshold=0.99):
        """
        method: quasi_constant_filter
        descripption:
        """
        columns_to_remove = []
        for col in df.columns:
            modal_frequency = df[col].value_counts().iloc[0]
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
        description:
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
        description:
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
        description:
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
    def pearsons_corr_filter(df, target=None, threshold=None):
        """
        method: pearsons_corr_filter
        description:
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

        coorelation_matrix = df.corr()

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
                        >= -threshold
                    ),
                    "VARIABLE_NAME",
                ].to_list()

            return columns_to_remove
