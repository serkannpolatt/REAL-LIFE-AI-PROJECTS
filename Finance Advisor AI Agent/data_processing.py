import os


class DataProcessor:
    def __init__(self, df_alloc, df_financial):
        self.df_alloc = df_alloc
        self.df_financial = df_financial
        self.save_path = "processed/"
        self.alloc_filename = "allocations.csv"
        self.financial_filename = "financial.csv"

    def _handle_missing_values_alloc(self):
        self.df_alloc = self.df_alloc[~self.df_alloc["Client"].isna()]
        self.df_alloc = self.df_alloc[~self.df_alloc["Target Allocation (%)"].isna()]
        self.df_alloc[["Target Portfolio", "Asset Class"]] = self.df_alloc[
            ["Target Portfolio", "Asset Class"]
        ].fillna("Unknown")

    def _handle_missing_values_financial(self):
        self.df_financial = self.df_financial[~self.df_financial["Client"].isna()]
        cols = [
            "Symbol",
            "Name",
            "Sector",
            "Purchase Date",
            "Analyst Rating",
            "Risk Level",
        ]
        self.df_financial[cols] = self.df_financial[cols].fillna("Unknown")
        self.df_financial.dropna(inplace=True)

    def _handle_client_column_financial(self):
        self.df_financial = self.df_financial[
            self.df_financial["Client"].isin(self.df_alloc["Client"].unique())
        ]

    def _get_client_id(self):
        self._handle_client_column_financial()
        self.df_alloc["Client"] = self.df_alloc["Client"].apply(
            lambda x: x.split("_")[1]
        )
        self.df_financial["Client"] = self.df_financial["Client"].apply(
            lambda x: x.split("_")[1]
        )

    def save_processed_data(self):
        self._handle_missing_values_alloc()
        self._handle_client_column_financial()
        self._get_client_id()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.df_alloc.to_csv(
            os.path.join(self.save_path, self.alloc_filename), index=False
        )
        self.df_financial.to_csv(
            os.path.join(self.save_path, self.financial_filename), index=False
        )
