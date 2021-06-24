import numpy as np
import pandas as pd

from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.bea import BEADataPreprocessor

from destination_based_sales.utils import eliminate_irrelevant_percentages, impute_missing_values


class RevenueSplitter:

    def __init__(self):
        self.IRSDataPreprocessor = IRSDataPreprocessor()
        self.BEADataPreprocessor = BEADataPreprocessor()

    def merge_dataframes(self):
        irs = self.IRSDataPreprocessor.load_final_data()
        bea = self.BEADataPreprocessor.load_final_data()

        merged_df = irs.merge(
            bea,
            how='left',
            on='CODE'
        )

        merged_df = merged_df[merged_df['CODE'] != 'USA'].copy()

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

        return merged_df.copy()

    def add_indicator_variables(self):

        merged_df = self.merge_dataframes()

        related = ['TOTAL_US_RELATED', 'TOTAL_AFFILIATE_COUNTRY_RELATED', 'TOTAL_OTHER_COUNTRY_RELATED']

        mask_0 = ~merged_df[related[0]].isnull()
        mask_1 = ~merged_df[related[1]].isnull()
        mask_2 = ~merged_df[related[2]].isnull()

        mask = np.logical_and(mask_0, np.logical_and(mask_1, mask_2))

        merged_df['IS_RELATED_COMPLETE'] = mask * 1

        self.related = related.copy()

        unrelated = ['TOTAL_US_UNRELATED', 'TOTAL_AFFILIATE_COUNTRY_UNRELATED', 'TOTAL_OTHER_COUNTRY_UNRELATED']

        mask_0 = ~merged_df[unrelated[0]].isnull()
        mask_1 = ~merged_df[unrelated[1]].isnull()
        mask_2 = ~merged_df[unrelated[2]].isnull()

        mask = np.logical_and(mask_0, np.logical_and(mask_1, mask_2))

        merged_df['IS_UNRELATED_COMPLETE'] = mask * 1

        self.unrelated = unrelated.copy()

        total = ['TOTAL_US', 'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_OTHER_COUNTRY']

        mask_0 = ~merged_df[total[0]].isnull()
        mask_1 = ~merged_df[total[1]].isnull()
        mask_2 = ~merged_df[total[2]].isnull()

        mask = np.logical_and(mask_0, np.logical_and(mask_1, mask_2))

        merged_df['IS_TOTAL_COMPLETE'] = mask * 1

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
