import os

import numpy as np
import pandas as pd

from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.utils import UK_CARIBBEAN_ISLANDS, ensure_country_overlap_with_IRS


path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_US_bop_data = os.path.join(path_to_dir, 'data', 'us_bop.xls')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class USBalanceOfPaymentsProcessor():

    def __init__(
        self,
        year,
        path_to_US_bop_data=path_to_US_bop_data,
        path_to_geographies=path_to_geographies
    ):

        self.year = year

        self.path_to_US_bop_data = path_to_US_bop_data
        self.path_to_geographies = path_to_geographies

        self.UK_CARIBBEAN_ISLANDS = UK_CARIBBEAN_ISLANDS.copy()

        preprocessor = IRSDataPreprocessor(year=year)
        self.unique_IRS_country_codes = preprocessor.load_final_data()['CODE'].unique()

    def load_raw_data(self):

        dfs = []

        dfs = []

        for sheet_name in ['Sheet0', 'Sheet1']:

            sheet = pd.read_excel(
                self.path_to_US_bop_data,
                sheet_name=sheet_name
            )

            sheet = sheet.iloc[4:13, 1:].copy()

            sheet = sheet.transpose().reset_index(drop=True)

            sheet = sheet.set_axis(
                axis=1,
                labels=sheet.iloc[0]
            ).iloc[1:].reset_index(drop=True)

            sheet.columns = [
                'OTHER_COUNTRY_NAME', 'YEAR', 'TOTAL_EXPORTS', 'GOODS_EXPORTS', 'MERCHANDISE_EXPORTS',
                'SERVICES_EXPORTS', 'FINANCIAL_SERVICES', 'INTELLECTUAL_PROPERTY', 'TELECOM'
            ]

            dfs.append(sheet)

        df = pd.concat(dfs, axis=0)

        df['OTHER_COUNTRY_NAME'] = df['OTHER_COUNTRY_NAME'].map(
            lambda name: name.strip('\xa0')
        )

        df['YEAR'] = df['YEAR'].astype(int)
        df = df[df['YEAR'] == self.year].copy()

        # for column in ['FINANCIAL_SERVICES', 'INTELLECTUAL_PROPERTY', 'TELECOM']:
        #     df[column] = df[column].map(lambda x: 0 if isinstance(x, str) and x in ['(D)', '(*)'] else x)

        # df['SERVICES_EXPORTS'] -= df['INTELLECTUAL_PROPERTY']

        df['ALL_EXPORTS'] = df['MERCHANDISE_EXPORTS'] + df['SERVICES_EXPORTS']
        df = df[['OTHER_COUNTRY_NAME', 'ALL_EXPORTS', 'MERCHANDISE_EXPORTS', 'SERVICES_EXPORTS']].copy()

        return df.copy()

    def load_data_with_geographies(self):
        df = self.load_raw_data()

        geographies = pd.read_csv(self.path_to_geographies)

        merged_df = df.merge(
            geographies,
            how='left',
            left_on='OTHER_COUNTRY_NAME', right_on='NAME'
        )

        imputation = {
            'CODE': 'UKI',
            'CONTINENT_NAME': 'America',
            'CONTINENT_CODE': 'AMR'
        }

        for column in ['CODE', 'CONTINENT_NAME', 'CONTINENT_CODE']:
            merged_df[column] = merged_df.apply(
                (
                    lambda row: imputation[column]
                    if row['OTHER_COUNTRY_NAME'] == 'United Kingdom Islands, Caribbean'
                    else row[column],
                ),
                axis=1
            )

        merged_df = merged_df.drop(columns=['NAME']).dropna()

        merged_df = merged_df[~merged_df['CODE'].isin(['EUR', 'AFR', 'AMR', 'ASIA'])].copy()

        return merged_df.copy()

    def load_data_with_IRS_overlap(self):
        df = self.load_data_with_geographies()

        df = df.rename(columns={'CODE': 'OTHER_COUNTRY_CODE'})

        df['OTHER_COUNTRY_CODE'] = df.apply(
            lambda row: ensure_country_overlap_with_IRS(
                row,
                self.unique_IRS_country_codes,
                self.UK_CARIBBEAN_ISLANDS
            ),
            axis=1
        )

        df = df.drop(
            columns=['OTHER_COUNTRY_NAME', 'CONTINENT_CODE', 'CONTINENT_NAME']
        )

        df = df.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        return df.copy()

    def load_data_for_adjustment(self):
        df = self.load_data_with_IRS_overlap()

        df['AFFILIATE_COUNTRY_CODE'] = 'USA'

        df['EXPORT_PERC'] = df['ALL_EXPORTS'] / df['ALL_EXPORTS'].sum()
        df['ALL_EXPORTS'] *= 10**6

        return df.drop(columns=['MERCHANDISE_EXPORTS', 'SERVICES_EXPORTS'])
