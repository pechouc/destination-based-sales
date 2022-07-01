import os

import numpy as np
import pandas as pd

import warnings

from destination_based_sales.utils import UK_CARIBBEAN_ISLANDS, CONTINENT_CODES_TO_IMPUTE_TRADE, \
    impute_missing_continent_codes, online_path_to_geo_file, url_to_data


path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_merchandise_data = os.path.join(path_to_dir, 'data', 'merchandise_trade_statistics.csv')
path_to_services_data = os.path.join(path_to_dir, 'data', 'OECD-WTO_BATIS_BPM6_Jan2021_bulk.csv')

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class BalancedTradeStatsProcessor:

    def __init__(
        self,
        year,
        service_flows_to_exclude=None,
        load_data_online=False
    ):

        self.year = year

        self.service_flows_to_exclude = service_flows_to_exclude

        if not load_data_online:
            self.path_to_merchandise_data = path_to_merchandise_data
            self.path_to_services_data = path_to_services_data

            self.path_to_geographies = path_to_geographies

        else:
            # # If relevant, we construct the URL from which we can load the BIMTS data
            # url_base = 'http://stats.oecd.org/SDMX-JSON/data/'
            # merchandise_dataset_identifier = 'BIMTS_HS2017/'
            # services_dataset_identifier = 'BATIS_EBOPS2010'
            # dimensions = 'ALL/'
            # agency_name = 'OECD'

            # self.path_to_merchandise_data = (
            #     url_base + merchandise_dataset_identifier + dimensions + agency_name + '?contenttype=csv'
            # )
            # self.path_to_services_data = (
            #     url_base + services_dataset_identifier + dimensions + agency_name + '?contenttype=csv'
            # )

            # For now, we rely on the files available on GitHub
            self.path_to_merchandise_data = url_to_data + 'merchandise_trade_statistics.csv'

            temp = 'https://github.com/pechouc/destination-based-sales/blob/main/destination_based_sales/data/'
            temp += 'OECD-WTO_BATIS_BPM6_Jan2021_bulk.csv?raw=true'
            self.path_to_services_data = temp

            self.path_to_geographies = online_path_to_geo_file

        self.geographies = pd.read_csv(self.path_to_geographies)

        self.UK_CARIBBEAN_ISLANDS = UK_CARIBBEAN_ISLANDS.copy()
        self.CONTINENT_CODES_TO_IMPUTE_TRADE = CONTINENT_CODES_TO_IMPUTE_TRADE.copy()

    def load_clean_merchandise_data(self):

        # Reading the BIMTS dataset
        merchandise = pd.read_csv(self.path_to_merchandise_data)

        # Focusing on the year of interest
        if self.year == 2019:
            year_considered = 2018
            warnings.warn('BIMTS data not available for 2019; using the 2018 statistics instead.')

        else:
            year_considered = self.year

        merchandise = merchandise[merchandise['TIME'] == year_considered].copy()

        # Eliminating irrelevant columns
        merchandise.drop(
            columns=[
                'Reporter Country', 'Partner Country', 'COMMODITY',
                'Commodity HS2017', 'MEASURE', 'Measure', 'TIME',
                'Time', 'Flag Codes', 'Flags'
            ],
            inplace=True
        )

        # Renaming variables in the standardized way
        merchandise.rename(
            columns={
                'REPORTER': 'AFFILIATE_COUNTRY_CODE',
                'PARTNER': 'OTHER_COUNTRY_CODE',
                'Value': 'MERCHANDISE_EXPORTS'
            },
            inplace=True
        )

        # Adding the OTHER_COUNTRY_CONTINENT_CODE column
        merchandise = merchandise.merge(
            self.geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='inner',
            left_on='OTHER_COUNTRY_CODE', right_on='CODE'
        )

        merchandise['CONTINENT_CODE'] = merchandise['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR'] else x
        )
        merchandise['CONTINENT_CODE'] = merchandise['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        merchandise = merchandise.rename(columns={'CONTINENT_CODE': 'OTHER_COUNTRY_CONTINENT_CODE'})
        merchandise = merchandise.drop(columns=['CODE'])

        # Adding the AFFILIATE_COUNTRY_CONTINENT_CODE column
        merchandise = merchandise.merge(
            self.geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='inner',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        merchandise['CONTINENT_CODE'] = merchandise['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR'] else x
        )
        merchandise['CONTINENT_CODE'] = merchandise['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        merchandise = merchandise.rename(columns={'CONTINENT_CODE': 'AFFILIATE_COUNTRY_CONTINENT_CODE'})
        merchandise = merchandise.drop(columns=['CODE'])

        # Summing exports over the different sectors (we are interested in the total of all commodities)
        merchandise = merchandise.groupby(
            [
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CONTINENT_CODE',
                'OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE'
            ]
        ).sum().reset_index()

        return merchandise.copy()

    def load_clean_services_data(self):

        if self.service_flows_to_exclude is None:
            raise Exception(
                "To load service exports based on BaTIS data, you need to specify what flows of services to exclude:"
                + "\n - if you don't want to exclude any flow of services, pass an empty list as argument;"
                + "\n - the following flows of services can be excluded for example:"
                + "\n   - 'SF': insurance and pension services;"
                + "\n   - 'SG': financial services;"
                + "\n   - 'SH': charges for the use of intellectual property;"
                + "\n   - 'SJ': other business services."
            )

        # This part can be quite long as the dataset is very heavy
        services = pd.read_csv(self.path_to_services_data)

        # We only keep rows that correspond to the year considered
        try:
            services = services[services['Year'] == self.year].copy()

        except KeyError:
            url_to_file = 'https://github.com/pechouc/destination-based-sales/blob/main/'
            url_to_file += 'destination_based_sales/data/OECD-WTO_BATIS_BPM6_Jan2021_bulk.csv?raw=true'

            services = pd.read_csv(url_to_file)
            services = services[services['Year'] == self.year].copy()

        # We only consider countries and eliminate geographical aggregates from reporters and partners
        services = services[
            np.logical_and(
                services['type_Reporter'] == 'c',
                services['type_Partner'] == 'c'
            )
        ]

        # We eliminate the rows that involve the State Union of Serbia and Montenegro (dissolved in 2006)
        services = services[
            np.logical_and(
                services['Reporter'] != 'SCG',
                services['Partner'] != 'SCG'
            )
        ]

        # We will treat a few countries separately (not based on the balanced value but on extrapolated reported values)
        services_extract = services[
            services['Partner'].isin(['BMU', 'CYM', 'BRB'])
        ].copy()

        # We focus on exports (but since we take balanced values, this does not really matter)
        services = services[services['Flow'] == 'EXP'].copy()

        # We reshape the DataFrame so as to have the different types of service flows as columns
        # We consider the "balanced values"
        services = services.pivot(
            index=['Reporter', 'Partner'],
            columns='Item_code',
            values='Balanced_value'
        ).reset_index()

        # We net out the service flows that we have chosen to exclude
        if len(self.service_flows_to_exclude) > 0:
            services['SOX'] -= services[self.service_flows_to_exclude].sum(axis=1)

        # We drop irrelevant columns - We focus on the flow of commercial services
        services = services[['Reporter', 'Partner', 'SOX']].copy()

        # We prepare similarly the extract for the few countries treated separately
        extract_imports = services_extract[services_extract['Flow'] == 'EXP'].copy()   # Exports to BMU, CYM, BRB
        extract_exports = services_extract[services_extract['Flow'] == 'IMP'].copy()   # Exports of BMU, CYM, BRB

        services_extract = []

        for i, df in enumerate([extract_imports, extract_exports]):
            df = df.pivot(
                index=['Reporter', 'Partner'],
                columns='Item_code',
                values='Final_value'
            ).reset_index()

            if len(self.service_flows_to_exclude) > 0:
                df['SOX'] -= df[self.service_flows_to_exclude].sum(axis=1)

            df = df[['Reporter', 'Partner', 'SOX']].copy()

            if i == 1:
                df = df.rename(
                    columns={'Reporter': 'Partner_', 'Partner': 'Reporter_'}
                ).rename(
                    columns={'Reporter_': 'Reporter', 'Partner_': 'Partner'}
                )

            services_extract.append(df)

        services_extract = pd.concat(services_extract, axis=0)

        # We filter out the few countries treated separately from the central table and concatenate both DataFrames
        services = services[
            ~np.logical_or(
                services['Reporter'].isin(['BMU', 'CYM', 'BRB']),
                services['Partner'].isin(['BMU', 'CYM', 'BRB'])
            )
        ].copy()
        services = pd.concat([services, services_extract], axis=0)

        # Final trade values are expressed in USD millions in the dataset, we convert them to USD
        services['SOX'] *= 10**6

        # We modify the ISO alpha-3 code associated with Kosovo for both the Reporter and Partner columns
        services['Reporter'] = services['Reporter'].map(lambda x: 'XXK' if x == 'XKV' else x)
        services['Partner'] = services['Partner'].map(lambda x: 'XXK' if x == 'XKV' else x)

        # We rename columns in the standardized way
        services = services.rename(
            columns={
                'Reporter': 'AFFILIATE_COUNTRY_CODE',
                'Partner': 'OTHER_COUNTRY_CODE',
                'SOX': 'SERVICES_EXPORTS'
            }
        )

        services = services.reset_index(drop=True)

        # Adding the OTHER_COUNTRY_CONTINENT_CODE column
        services = services.merge(
            self.geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='OTHER_COUNTRY_CODE', right_on='CODE'
        )

        services['CONTINENT_CODE'] = services['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR'] else x
        )
        services['CONTINENT_CODE'] = services['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        services = services.drop(columns=['CODE'])
        services = services.rename(columns={'CONTINENT_CODE': 'OTHER_COUNTRY_CONTINENT_CODE'})

        # Adding the AFFILIATE_COUNTRY_CONTINENT_CODE column
        services = services.merge(
            self.geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        services['CONTINENT_CODE'] = services['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR'] else x
        )
        services['CONTINENT_CODE'] = services['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        services = services.drop(columns=['CODE'])
        services = services.rename(columns={'CONTINENT_CODE': 'AFFILIATE_COUNTRY_CONTINENT_CODE'})

        # Managing the records with "Rest of the world" as partner
        services = services.groupby(
            [
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CONTINENT_CODE',
                'OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE'
            ]
        ).sum().reset_index()

        transformer = ServicesDataTransformer()

        transformer.fit(services)

        services = transformer.transform(services)

        return services.copy()


class ServicesDataTransformer:

    def __init__(self):
        self.amounts_to_distribute = {}
        self.dataframes = []

    def fit(self, data):
        for country in data['AFFILIATE_COUNTRY_CODE'].unique():
            mask_affiliate_country = data['AFFILIATE_COUNTRY_CODE'] == country
            mask_RWD = data['OTHER_COUNTRY_CODE'] == 'RWD'

            mask = np.logical_and(mask_affiliate_country, mask_RWD)

            # We determine the amount of exports to the "Rest of the world" that must be reallocated
            if not data[mask].empty:
                self.amounts_to_distribute[country] = data[mask]['SERVICES_EXPORTS'].iloc[0]

            else:
                self.amounts_to_distribute[country] = 0

            # We determine how this amount will be distributed between the different "Other ..." aggregates
            restricted_df = data[mask_affiliate_country].copy()

            self.affiliate_country_continent_code = restricted_df['AFFILIATE_COUNTRY_CONTINENT_CODE'].iloc[0]

            restricted_df = restricted_df.groupby(
                'OTHER_COUNTRY_CONTINENT_CODE'
            ).sum()['SERVICES_EXPORTS'].reset_index()

            restricted_df['ALLOCABLE_SHARE'] = (
                restricted_df['SERVICES_EXPORTS'] / restricted_df['SERVICES_EXPORTS'].sum()
            )

            # We allocate the ROW amounts to the different "Other ..." aggregates
            restricted_df['OTHER_COUNTRY_CODE'] = restricted_df['OTHER_COUNTRY_CONTINENT_CODE'].map(
                lambda continent_code: 'O' + continent_code
            )

            restricted_df['SERVICES_EXPORTS'] = restricted_df['ALLOCABLE_SHARE'] * self.amounts_to_distribute[country]

            restricted_df = restricted_df.drop(columns=['ALLOCABLE_SHARE'])

            # We add the AFFILIATE_COUNTRY_CODE and AFFILIATE_COUNTRY_CONTINENT_CODE columns
            restricted_df['AFFILIATE_COUNTRY_CODE'] = country
            restricted_df['AFFILIATE_COUNTRY_CONTINENT_CODE'] = self.affiliate_country_continent_code

            # We store the restricted DataFrame
            self.dataframes.append(restricted_df)

    def transform(self, data):
        dataframe_to_append = pd.concat(self.dataframes, axis=0)

        data = data[data['OTHER_COUNTRY_CODE'] != 'RWD'].copy()

        data = pd.concat([data, dataframe_to_append], axis=0)

        # For the Netherlands Antilles, we have 0 exports of services to all destinations, so we replace the NaNs
        # created via the allocable share ratio by 0s
        data['SERVICES_EXPORTS'] = data['SERVICES_EXPORTS'].fillna(0)

        return data.reset_index(drop=True)
