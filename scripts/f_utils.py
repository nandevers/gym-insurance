import pandas as pd


class CumulativeCountEncoder:
    def __init__(self, group_by_cols, sort_col):
        self.group_by_cols = group_by_cols
        self.sort_col = sort_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X[self.sort_col] = pd.to_datetime(X[self.sort_col], dayfirst=True)
        X[f"nr_previous_{self.sort_col}"] = (
            X.sort_values(self.sort_col)
            .groupby(self.group_by_cols)[self.sort_col]
            .cumcount()
            .transform(lambda x: x)
        )
        return X


class CumulativeEncoder:
    def __init__(self, operation, col, group_by_cols, sort_col):
        self.operation = operation
        self.group_by_cols = group_by_cols
        self.sort_col = sort_col
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.sort_values(self.sort_col)
        fn = {"max": self._cum_max, "min": self._cum_min, "sum": self._cum_sum}
        if filter(["max", "min", "sum"], self.operation) is None:
            raise ValueError(
                "Invalid operation. Only 'max', 'min', 'sum' are supported."
            )
        if isinstance(self.operation, list):
            return pd.concat([fn[o](X) for o in self.operation], axis=1)
        return fn[self.operation](X)

    def _cum_max(self, data):
        data.loc[:, f"max_{self.col}"] = (
            data.sort_values(self.sort_col)
            .groupby(self.group_by_cols)[self.col]
            .cummax()
            .transform(lambda x: x)
        )
        return data

    def _cum_min(self, data):
        data.loc[:, f"min_{self.col}"] = (
            data.sort_values(self.sort_col)
            .groupby(self.group_by_cols)[self.col]
            .cummin()
            .transform(lambda x: x)
        )
        return data

    def _cum_sum(self, data):
        data.loc[:, f"sum_{self.col}"] = (
            data.sort_values(self.sort_col)
            .groupby(self.group_by_cols)[self.col]
            .cumsum()
            .transform(lambda x: x)
        )
        return data
