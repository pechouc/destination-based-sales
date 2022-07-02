import os

import numpy as np
import pandas as pd

from destination_based_sales.utils import UK_CARIBBEAN_ISLANDS, ensure_country_overlap_with_IRS, \
    online_path_to_geo_file, url_to_data


path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_US_bop_data = os.path.join(path_to_dir, 'data', 'us_bop.xls')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')

GEOGRAPHIES_TO_IMPUTE = {
    'United Kingdom Islands, Caribbean': {
        'CODE': 'UKI',
        'CONTINENT_NAME': 'America',
        'CONTINENT_CODE': 'AMR'
    },
    'South Korea': {
        'CODE': 'KOR',
        'CONTINENT_NAME': 'Asia',
        'CONTINENT_CODE': 'ASIA'
    },
    'Vietnam': {
        'CODE': 'VNM',
        'CONTINENT_NAME': 'Asia',
        'CONTINENT_CODE': 'ASIA'
    }
}


class USBalanceOfPaymentsProcessor():

    def __init__(
        self,
        year,
        load_data_online=False
    ):

        self.year = year

        if not load_data_online:
            self.path_to_US_bop_data = path_to_US_bop_data
            self.path_to_geographies = path_to_geographies

        else:
            self.path_to_US_bop_data = url_to_data + 'us_bop.xls'
            self.path_to_geographies = online_path_to_geo_file

        geographies = pd.read_csv(self.path_to_geographies)
        self.geographies = geographies.groupby('CODE').first().reset_index()

        self.GEOGRAPHIES_TO_IMPUTE = GEOGRAPHIES_TO_IMPUTE.copy()

    def load_raw_data(self):

        dfs = []

        for sheet_name in ['Sheet0', 'Sheet1']:

            sheet = pd.read_excel(
                self.path_to_US_bop_data,
                sheet_name=sheet_name
            )

            sheet = sheet.iloc[4:14, 1:].copy()

            sheet = sheet.transpose().reset_index(drop=True)

            sheet = sheet.set_axis(
                axis=1,
                labels=sheet.iloc[0]
            ).iloc[1:].reset_index(drop=True)

            sheet.columns = [
                'OTHER_COUNTRY_NAME', 'YEAR', 'TOTAL_EXPORTS', 'GOODS_EXPORTS', 'MERCHANDISE_EXPORTS',
                'SERVICES_EXPORTS', 'FINANCIAL_SERVICES', 'INTELLECTUAL_PROPERTY', 'TELECOM', 'GOVERNMENT_SERVICES'
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

        # We net out government services:
        # - Observations with "(D)" correspond to non-disclosures for confidentiality purposes (one entity involved)
        # - Observations with "(*)" correspond to amounts comprised between 0 and USD 500,000
        # We impute 0 in both of these cases as an approximation for very small transactions
        df['GOVERNMENT_SERVICES'] = df['GOVERNMENT_SERVICES'].map(
            lambda x: 0 if isinstance(x, str) and x in ['(D)', '(*)'] else x
        )
        df['SERVICES_EXPORTS'] -= df['GOVERNMENT_SERVICES']

        df['ALL_EXPORTS'] = df['MERCHANDISE_EXPORTS'] + df['SERVICES_EXPORTS']
        df = df[['OTHER_COUNTRY_NAME', 'ALL_EXPORTS', 'MERCHANDISE_EXPORTS', 'SERVICES_EXPORTS']].copy()

        return df.copy()

    def load_data_with_geographies(self):
        df = self.load_raw_data()

        # We add the other country code and the other country continent code
        merged_df = df.merge(
            self.geographies,
            how='left',
            left_on='OTHER_COUNTRY_NAME', right_on='NAME'
        )

        # We impute these codes for each country whose name is not identified in geographies.csv
        for country_name, imputation in self.GEOGRAPHIES_TO_IMPUTE.items():
            for column in ['CODE', 'CONTINENT_NAME', 'CONTINENT_CODE']:
                merged_df[column] = merged_df.apply(
                    (
                        lambda row: imputation[column]
                        if row['OTHER_COUNTRY_NAME'] == country_name
                        else row[column],
                    ),
                    axis=1
                )

        # We do not need the full names anymore
        merged_df = merged_df.drop(columns=['NAME', 'OTHER_COUNTRY_NAME', 'CONTINENT_NAME'])

        # We eliminate the continental aggregates for which a code is found in geographies.csv
        merged_df = merged_df[~merged_df['CODE'].isin(['EUR', 'AFR', 'AMR', 'ASIA'])].copy()

        # We eliminate all the rows for which codes were not found in geographies.csv as they correspond to geographic
        # aggregates that either induce double-counting (continental total) or cannot be matched with the aggregates in
        # OECD CbCR or in IRS data ("Other ..." aggregates)
        merged_df = merged_df.dropna()

        # We restrict the continent codes to the 4 that are relevant in the following steps of the adjustment
        merged_df['CONTINENT_CODE'] = merged_df['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR'] else x
        )
        merged_df['CONTINENT_CODE'] = merged_df['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        # We rename columns in the standardized way
        merged_df = merged_df.rename(
            columns={
                'CODE': 'OTHER_COUNTRY_CODE',
                'CONTINENT_CODE': 'OTHER_COUNTRY_CONTINENT_CODE'
            }
        )

        return merged_df.copy()

    def load_final_merchandise_data(self):
        df = self.load_data_with_geographies()

        # We add the two columns that are still missing to match the standardized trade statistics format
        df['AFFILIATE_COUNTRY_CODE'] = 'USA'
        df['AFFILIATE_COUNTRY_CONTINENT_CODE'] = 'AMR'

        # We drop irrelevant columns
        df = df.drop(columns=['ALL_EXPORTS', 'SERVICES_EXPORTS'])

        # We move from USD millions to USD
        df['MERCHANDISE_EXPORTS'] *= 10**6

        return df.copy()

    def load_final_services_data(self):
        df = self.load_data_with_geographies()

        # We add the two columns that are still missing to match the standardized trade statistics format
        df['AFFILIATE_COUNTRY_CODE'] = 'USA'
        df['AFFILIATE_COUNTRY_CONTINENT_CODE'] = 'AMR'

        # We drop irrelevant columns
        df = df.drop(columns=['ALL_EXPORTS', 'MERCHANDISE_EXPORTS'])

        # We move from USD millions to USD
        df['SERVICES_EXPORTS'] *= 10**6

        return df.copy()
