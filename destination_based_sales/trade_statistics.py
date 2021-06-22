import os

import numpy as np
import pandas as pd

from revenue_split import RevenueSplitter

from utils import UK_CARIBBEAN_ISLANDS, CONTINENT_CODES_TO_IMPUTE_TRADE, impute_missing_continent_codes

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_merchandise_data = os.path.join(path_to_dir, 'data', 'merchandise_trade_statistics.csv')
path_to_services_data = os.path.join(path_to_dir, 'data', 'services_trade_statistics.csv')

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class TradeStatisticsProcessor:

    def __init__(
        self,
        path_to_merchandise_data=path_to_merchandise_data, path_to_services_data=path_to_services_data,
        path_to_geographies=path_to_geographies
    ):
        self.path_to_merchandise_data = path_to_merchandise_data
        self.path_to_services_data = path_to_services_data

        self.path_to_geographies = path_to_geographies

        self.UK_CARIBBEAN_ISLANDS = UK_CARIBBEAN_ISLANDS.copy()
        self.CONTINENT_CODES_TO_IMPUTE_TRADE = CONTINENT_CODES_TO_IMPUTE_TRADE.copy()

    def load_clean_merchandise_data(self):

        merchandise = pd.read_csv(self.path_to_merchandise_data)

        merchandise.drop(
            columns=[
                'Reporter Country', 'Partner Country', 'COMMODITY',
                'Commodity HS2017', 'MEASURE', 'Measure', 'TIME',
                'Time', 'Flag Codes', 'Flags'
            ],
            inplace=True
        )

        merchandise.rename(
            columns={
                'REPORTER': 'AFFILIATE_COUNTRY_CODE',
                'PARTNER': 'OTHER_COUNTRY_CODE',
                'Value': 'MERCHANDISE_EXPORTS'
            },
            inplace=True
        )

        return merchandise.copy()

    def load_clean_services_data(self):

        services = pd.read_csv(self.path_to_services_data)

        services = services[services['Measure'] == 'Final balanced value'].copy()

        services.drop(
            columns=[
                'Reporter Country', 'Partner Country', 'EXPRESSION', 'Expression',
                'SERVICE', 'Service', 'MEASURE', 'Measure', 'TIME', 'Year',
                'Unit Code', 'Unit', 'PowerCode', 'Reference Period Code',
                'Reference Period', 'Flag Codes', 'Flags'
            ],
            inplace=True
        )

        services['Value'] = services['Value'] * services['PowerCode Code'].map(lambda x: 10**x)

        services.drop(columns=['PowerCode Code'], inplace=True)

        services.rename(
            columns={
                'LOCATION': 'AFFILIATE_COUNTRY_CODE',
                'PARTNER': 'OTHER_COUNTRY_CODE',
                'Value': 'SERVICES_EXPORTS'
            },
            inplace=True
        )

        services.reset_index(drop=True, inplace=True)

        return services.copy()

    def load_merged_dataframe(self):

        merged_df = merchandise.merge(
            services,
            how='inner',
            on=['AFFILIATE_COUNTRY_CODE', 'OTHER_COUNTRY_CODE']
        )

        mask_domestic = (merged_df['AFFILIATE_COUNTRY_CODE'] != merged_df['OTHER_COUNTRY_CODE'])
        mask_US = (merged_df['OTHER_COUNTRY_CODE'] != 'USA')

        mask = np.logical_and(mask_domestic, mask_US)

        merged_df = merged_df[mask].copy()

        merged_df['ALL_EXPORTS'] = merged_df['MERCHANDISE_EXPORTS'] + merged_df['SERVICES_EXPORTS']

        totals = {}

        for country in merged_df['AFFILIATE_COUNTRY_CODE'].unique():

            restricted_df = merged_df[merged_df['AFFILIATE_COUNTRY_CODE'] == country].copy()

            totals[country] = restricted_df['ALL_EXPORTS'].sum()

        merged_df['TOTAL_EXPORTS'] = merged_df['AFFILIATE_COUNTRY_CODE'].map(totals)

        merged_df['EXPORT_PERC'] = merged_df['ALL_EXPORTS'] / merged_df['TOTAL_EXPORTS']

    def manage_UK_caribbean_islands(self):

        merged_df = self.load_merged_dataframe()

        extract = merged_df[merged_df['AFFILIATE_COUNTRY_CODE'].isin(uk_caribbean_islands)].copy()

        if extract.shape[0] == 0:
            return merged_df.copy()

        else:
            raise Exception('We now have to manage the case of UK Caribbean Islands properly.')

    def compute_exports_per_continent(self):

        merged_df = self.manage_UK_caribbean_islands()

        geographies = pd.read_csv(self.path_to_geographies)

        continents = merged_df.merge(
            geographies[['CODE', 'CONTINENT_CODE']].copy(),
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        continents.drop(columns='CODE', inplace=True)

        continents['CONTINENT_CODE'] = continents['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR'] else x
        )

        continents['CONTINENT_CODE'] = continents['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        exports_per_continent = {}

        for continent in continents['CONTINENT_CODE'].unique():

            restricted_df = continents[continents['CONTINENT_CODE'] == continent].copy()

            restricted_df = restricted_df[['OTHER_COUNTRY_CODE', 'ALL_EXPORTS']].copy()

            restricted_df = restricted_df.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

            exports_per_continent[continent] = restricted_df.copy()

        return exports_per_continent.copy()

    def load_data_with_imputations(self):

        merged_df = self.manage_UK_caribbean_islands()

        splitter = RevenueSplitter()

        splitted_revenues = splitter.get_splitted_revenues()

        exports_per_continent = self.compute_exports_per_continent()

        missing_countries = splitted_revenues[
            ~splitted_revenues['CODE'].isin(merged_df['AFFILIATE_COUNTRY_CODE'])
        ][['AFFILIATE_COUNTRY_NAME', 'CODE']]

        missing_countries = missing_countries.merge(
            geographies[['CODE', 'CONTINENT_CODE']],
            how='left',
            on='CODE'
        )

        missing_countries['CONTINENT_CODE'] = missing_countries.apply(
            lambda row: impute_missing_continent_codes(row, self.CONTINENT_CODES_TO_IMPUTE_TRADE),
            axis=1
        )

        missing_countries['CONTINENT_CODE'] = missing_countries['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR'] else x
        )

        missing_countries['CONTINENT_CODE'] = missing_countries['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        missing_countries.drop_duplicates(inplace=True)

        output_df = merged_df[
            ['AFFILIATE_COUNTRY_CODE', 'OTHER_COUNTRY_CODE', 'EXPORT_PERC']
        ].copy()

        for _, row in missing_countries.iterrows():

            df = exports_per_continent[row['CONTINENT_CODE']].copy()

            df = df[df['OTHER_COUNTRY_CODE'] != row['CODE']].copy()

            df['EXPORT_PERC'] = df['ALL_EXPORTS'] / df['ALL_EXPORTS'].sum()

            df['AFFILIATE_COUNTRY_CODE'] = row['CODE']

            df.drop(columns=['ALL_EXPORTS'], inplace=True)

            output_df = pd.concat(
                [output_df, df],
                axis=0
            )

        return output_df.copy()
