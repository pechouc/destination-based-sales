"""
This module is used to split the revenue variables of the IRS country-by-country data into three categories (sales to
the affiliate country, sales to the US and sales to any third country), based on BEA data. It therefore relies on other
Python files, essentially irs.py and bea.py. The methodology is detailed either in the PDF report or in the docstrings
and comments below.
"""


########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.bea import BEADataPreprocessor

from destination_based_sales.utils import eliminate_irrelevant_percentages, impute_missing_values


########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))


########################################################################################################################
# --- Content

class RevenueSplitter:

    def __init__(
        self,
        year,
        include_US=True,
        path_to_dir=path_to_dir
    ):
        """
        The logic allowing to split revenue variables is encapsulated in a Python class, RevenueSplitter.

        This is the instantiation function for this class, which requires the following arguments:

        - the year to consider;
        - a boolean, indicating whether or not to include US-US sales in the split;
        - and the path to the directory where the Python file is located, to retrieve the necessary data.

        """
        self.year = year

        self.irs_preprocessor = IRSDataPreprocessor(year=year)
        self.bea_preprocessor = BEADataPreprocessor(year=year)

        self.include_US = include_US

        # We reconstruct the path to the Excel file that contains the BEA data we use for the split of US-US sales
        self.path_to_BEA_KR_tables = os.path.join(
            path_to_dir,
            'data',
            str(year),
            'Part-I-K1-R2.xls'
        )

    def merge_dataframes(self, include_US=True):
        """
        This class method is used to combine the IRS and BEA dataset. If US-US sales are excluded, it consists in a sim-
        ple merge, with a few duplicate columns to filter out. On the other hand, if US-US sales are included, since
        their split is based on a secondary file provided by the BEA, we have to operate the split separately and re-
        introduce it in the merged dataset.
        """
        irs = self.irs_preprocessor.load_final_data()
        bea = self.bea_preprocessor.load_final_data()

        # We merge the two datasets (IRS and BEA)
        merged_df = irs.merge(
            bea,
            how='left',
            on='CODE'
        )

        merged_df.drop(
            columns=['AFFILIATE_COUNTRY_NAME_y', 'CONTINENT_NAME_y', 'CONTINENT_CODE_y', 'NAME'],
            inplace=True
        )

        merged_df.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME_x': 'AFFILIATE_COUNTRY_NAME',
                'CONTINENT_NAME_x': 'CONTINENT_NAME',
                'CONTINENT_CODE_x': 'CONTINENT_CODE'
            },
            inplace=True
        )

        if include_US:
            # We load the secondary file from the BEA, with the split of US-US sales
            df = pd.read_excel(self.path_to_BEA_KR_tables, sheet_name='Table I.O 1')

            column_names = df.loc[4].to_dict().copy()
            column_names[list(column_names.keys())[0]] = 'Industry'
            column_names['Unnamed: 1'] = 'Total'

            df.rename(columns=column_names, inplace=True)

            us_sales = df.loc[6].to_dict()

            us_imputation = {}

            for column in merged_df.columns[-11:]:

                if column == 'TOTAL':
                    us_imputation[column] = us_sales['Total']

                elif 'US' in column:
                    us_imputation[column] = us_sales['To U.S. persons'] * 1

                elif 'AFFILIATE_COUNTRY' in column:
                    us_imputation[column] = us_sales['To U.S. persons'] * 0

                else:
                    us_imputation[column] = us_sales['To foreign affiliates'] + us_sales['To other foreign persons']

            for column in merged_df.columns[-11:]:
                merged_df[column] = merged_df.apply(
                    lambda row: us_imputation[column] if row['CODE'] == 'USA' else row[column],
                    axis=1
                )

            return merged_df.copy()

        else:

            merged_df = merged_df[merged_df['CODE'] != 'USA'].copy()

            return merged_df.copy()

    def add_indicator_variables(self):

        merged_df = self.merge_dataframes(include_US=self.include_US)

        mask_US = (merged_df['CODE'] == 'USA')

        related = ['TOTAL_US_RELATED', 'TOTAL_AFFILIATE_COUNTRY_RELATED', 'TOTAL_OTHER_COUNTRY_RELATED']

        mask_0 = ~merged_df[related[0]].isnull()
        mask_1 = ~merged_df[related[1]].isnull()
        mask_2 = ~merged_df[related[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        merged_df['IS_RELATED_COMPLETE'] = mask * 1 + mask_US * 1

        self.related = related.copy()

        unrelated = ['TOTAL_US_UNRELATED', 'TOTAL_AFFILIATE_COUNTRY_UNRELATED', 'TOTAL_OTHER_COUNTRY_UNRELATED']

        mask_0 = ~merged_df[unrelated[0]].isnull()
        mask_1 = ~merged_df[unrelated[1]].isnull()
        mask_2 = ~merged_df[unrelated[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        merged_df['IS_UNRELATED_COMPLETE'] = mask * 1 + mask_US * 1

        self.unrelated = unrelated.copy()

        total = ['TOTAL_US', 'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_OTHER_COUNTRY']

        mask_0 = ~merged_df[total[0]].isnull()
        mask_1 = ~merged_df[total[1]].isnull()
        mask_2 = ~merged_df[total[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        merged_df['IS_TOTAL_COMPLETE'] = mask * 1 + mask_US * 1

        self.total = total.copy()

        return merged_df.copy()

    def compute_revenue_percentages(self):

        merged_df = self.add_indicator_variables()

        bases = ['TOTAL_US', 'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_OTHER_COUNTRY']

        percentage_columns = []

        for sales_type in ['RELATED', 'UNRELATED', 'TOTAL']:
            if sales_type in ['RELATED', 'UNRELATED']:
                existing_columns = [column + '_' + sales_type for column in bases]

                total_column = 'TOTAL_' + sales_type

            else:
                existing_columns = bases.copy()

                total_column = 'TOTAL_COMPUTED'

            merged_df[total_column] = merged_df[existing_columns].sum(axis=1)

            for i, destination in enumerate(['US', 'AFFILIATE_COUNTRY', 'OTHER_COUNTRY']):
                new_column = '_'.join(['PERC', sales_type, destination])

                percentage_columns.append(new_column)

                merged_df[new_column] = merged_df[existing_columns[i]] / merged_df[total_column]

        self.percentage_columns = percentage_columns.copy()

        for column in percentage_columns:
            merged_df[column] = merged_df.apply(
                lambda row: eliminate_irrelevant_percentages(row, column),
                axis=1
            )

        return merged_df.copy()

    def build_imputations_dict(self):

        merged_df = self.compute_revenue_percentages()

        imputations = {}

        for continent_code in merged_df['CONTINENT_CODE'].unique():

            restricted_df = merged_df[merged_df['CONTINENT_CODE'] == continent_code].copy()

            imputations[continent_code] = {}

            for sales_type in ['UNRELATED', 'RELATED', 'TOTAL']:

                indicator_column = '_'.join(['IS', sales_type, 'COMPLETE'])

                restricted_df = restricted_df[restricted_df[indicator_column] == 1].copy()

                sums = restricted_df.sum()

                if sales_type in ['UNRELATED', 'RELATED']:
                    suffix = '_' + sales_type
                    total_column = 'TOTAL' + suffix

                else:
                    suffix = ''
                    total_column = 'TOTAL_COMPUTED'

                for destination in ['US', 'AFFILIATE_COUNTRY', 'OTHER_COUNTRY']:
                    column = 'TOTAL_' + destination + suffix

                    key = '_'.join(['PERC', sales_type, destination])

                    imputations[continent_code][key] = sums.loc[column] / sums.loc[total_column]

        return imputations.copy()

    def impute_missing_percentages(self):

        merged_df = self.compute_revenue_percentages()

        imputations = self.build_imputations_dict()

        for column in self.percentage_columns:
            merged_df[column] = merged_df.apply(
                lambda row: impute_missing_values(row, column, imputations),
                axis=1
            )

        merged_df.drop(
            columns=self.related + self.unrelated + self.total + ['TOTAL_FOREIGN'],
            inplace=True
        )

        return merged_df.copy()

    def deduce_absolute_amounts(self):

        merged_df = self.impute_missing_percentages()

        absolute_amount_columns = []

        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']:

            sales_type = column.split('_')[0]

            for destination in ['US', 'OTHER_COUNTRY', 'AFFILIATE_COUNTRY']:

                new_column = column + '_TO_' + destination

                merged_df[new_column] = (
                    merged_df[column] * merged_df['PERC_' + sales_type + '_' + destination]
                )

                absolute_amount_columns.append(new_column)

        self.absolute_amount_columns = absolute_amount_columns.copy()

        return merged_df.copy()

    def get_splitted_revenues(self):

        merged_df = self.deduce_absolute_amounts()

        merged_df = merged_df[['AFFILIATE_COUNTRY_NAME', 'CODE'] + self.absolute_amount_columns].copy()

        return merged_df.copy()

    def get_sales_percentages(self):

        merged_df = self.deduce_absolute_amounts()

        merged_df = merged_df[['AFFILIATE_COUNTRY_NAME', 'CODE'] + self.percentage_columns].copy()

        return merged_df.copy()
