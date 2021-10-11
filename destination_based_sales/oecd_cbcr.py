import os

import numpy as np
import pandas as pd

import requests

from destination_based_sales.utils import impute_missing_continent_codes, CONTINENT_CODES_TO_IMPUTE_OECD_CBCR, \
    UK_CARIBBEAN_ISLANDS


path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')
path_to_GNI_data = os.path.join(path_to_dir, 'data', 'gross_national_income.csv')
path_to_tax_haven_list = os.path.join(path_to_dir, 'data', 'tax_havens.csv')
local_path_to_OECD_CbCR_data = os.path.join(path_to_dir, 'data', 'oecd_cbcr.csv')


class CbCRPreprocessor:

    def __init__(
        self,
        load_raw_data=True,
        path_to_geographies=path_to_geographies,
        continent_code_imputations=CONTINENT_CODES_TO_IMPUTE_OECD_CBCR,
        path_to_GNI_data=path_to_GNI_data,
        path_to_tax_haven_list=path_to_tax_haven_list,
        fetch_data_online=False
    ):
        if fetch_data_online:
            self.url_base = 'http://stats.oecd.org/SDMX-JSON/data/'
            self.dataset_identifier = 'CBCR_TABLEI/'
            self.dimensions = 'ALL/'
            self.agency_name = 'OECD'

            self.path_to_OECD_data = (
                self.url_base + self.dataset_identifier + self.dimensions + self.agency_name + '?contenttype=csv'
            )

        else:
            self.path_to_OECD_CbCR_data = local_path_to_OECD_CbCR_data

        self.path_to_geographies = path_to_geographies
        self.continent_code_imputations = continent_code_imputations

        self.path_to_GNI_data = path_to_GNI_data
        self.path_to_tax_haven_list = path_to_tax_haven_list

        if load_raw_data:
            if fetch_data_online:
                print("Fetching the OECD's aggregated and anonymized CbCR data - This may take up to 30 seconds.")

            self.data = pd.read_csv(self.path_to_OECD_CbCR_data)

            if fetch_data_online:
                print("Loaded the OECD's aggregated and anonymized CbCR data successfully.")

        else:
            self.data = None

    def load_raw_data(self):
        self.data = pd.read_csv(self.path_to_OECD_CbCR_data)

    def get_preprocessed_revenue_data(self):
        if self.data is None:
            raise Exception('You must load the data with the dedicated method before you may run any computation.')

        cbcr = self.data.copy()

        cbcr = cbcr[cbcr['PAN'] == 'PANELA'].copy()
        cbcr = cbcr[cbcr['Year'] == 2016].copy()

        cbcr.drop(
            columns=['PAN', 'Grouping', 'Flag Codes', 'Flags', 'YEA', 'Year'],
            inplace=True
        )

        cbcr_wide = cbcr.pivot(
            index=['COU', 'Ultimate Parent Jurisdiction', 'JUR', 'Partner Jurisdiction'],
            columns='Variable',
            values='Value'
        ).reset_index()

        oecd = cbcr_wide[
            [
                'COU', 'Ultimate Parent Jurisdiction', 'JUR', 'Partner Jurisdiction',
                'Related Party Revenues', 'Total Revenues', 'Unrelated Party Revenues'
            ]
        ].copy()

        oecd.rename(
            columns={
                'COU': 'PARENT_COUNTRY_CODE',
                'Ultimate Parent Jurisdiction': 'PARENT_COUNTRY_NAME',
                'JUR': 'AFFILIATE_COUNTRY_CODE',
                'Partner Jurisdiction': 'AFFILIATE_COUNTRY_NAME',
                'Related Party Revenues': 'RELATED_PARTY_REVENUES',
                'Total Revenues': 'TOTAL_REVENUES',
                'Unrelated Party Revenues': 'UNRELATED_PARTY_REVENUES'
            },
            inplace=True
        )

        oecd.rename_axis(None, axis=1, inplace=True)

        # We eliminate Stateless entities
        oecd = oecd[oecd['AFFILIATE_COUNTRY_NAME'] != 'Stateless'].copy()

        # We eliminate countries with minimum reporting (only a domestic / foreign split)
        mask = oecd['PARENT_COUNTRY_CODE'].map(
            lambda code: (oecd['PARENT_COUNTRY_CODE'] == code).sum()
        ) > 2
        oecd = oecd[mask].copy()

        # And we can now eliminate Foreign Jurisdictions Total rows
        oecd = oecd[oecd['AFFILIATE_COUNTRY_NAME'] != 'Foreign Jurisdictions Total'].copy()

        # We group the rows corresponding to UK Caribbean Islands
        oecd['AFFILIATE_COUNTRY_CODE'] = oecd['AFFILIATE_COUNTRY_CODE'].map(
            lambda country_code: country_code if country_code not in UK_CARIBBEAN_ISLANDS else 'UKI'
        )
        oecd['AFFILIATE_COUNTRY_NAME'] = oecd.apply(
            (
                lambda row: row['AFFILIATE_COUNTRY_NAME']
                if row['AFFILIATE_COUNTRY_CODE'] != 'UKI' else 'UK Caribbean Islands'
            ),
            axis=1
        )
        oecd = oecd.groupby(
            ['PARENT_COUNTRY_CODE', 'PARENT_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME']
        ).sum().reset_index()

        geographies = pd.read_csv(self.path_to_geographies)

        oecd = oecd.merge(
            geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        oecd.drop(columns=['CODE'], inplace=True)

        oecd['CONTINENT_CODE'] = oecd.apply(
            lambda row: impute_missing_continent_codes(row, self.continent_code_imputations),
            axis=1
        )

        oecd['CONTINENT_CODE'] = oecd['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        oecd['CONTINENT_CODE'] = oecd['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        # In 2016, one row (India - Bouvet Island) which is full of 0s
        oecd = oecd[oecd['CONTINENT_CODE'] != 'ATC'].copy()

        return oecd.copy()

    def get_revenue_data_with_tax_havens(self):
        oecd = self.get_preprocessed_revenue_data()

        tax_havens = pd.read_csv(self.path_to_tax_haven_list)

        oecd['IS_TAX_HAVEN'] = oecd['AFFILIATE_COUNTRY_CODE'].isin(tax_havens['CODE'].unique())

        return oecd.copy()

    def get_revenue_data_with_GNI(self):
        oecd = self.get_revenue_data_with_tax_havens()

        gross_national_income = pd.read_csv(self.path_to_GNI_data, delimiter=';')

        for column in gross_national_income.columns[2:]:

            gross_national_income[column] = gross_national_income[column].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            gross_national_income[column] = gross_national_income[column].astype(float)

        oecd = oecd.merge(
            gross_national_income[['COUNTRY_CODE', 'GNI_2016']].copy(),
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

        oecd.drop(columns=['COUNTRY_CODE'], inplace=True)

        return oecd.copy()

    def get_scatterplot_data(self, breakdown_threshold):
        oecd = self.get_revenue_data_with_GNI()

        oecd = oecd[oecd['AFFILIATE_COUNTRY_NAME'] != 'Foreign Jurisdictions Total'].copy()
        oecd = oecd[oecd['PARENT_COUNTRY_CODE'] != oecd['AFFILIATE_COUNTRY_CODE']].copy()

        oecd['NB_AFFILIATE_COUNTRIES'] = oecd['PARENT_COUNTRY_CODE'].map(
            lambda code: (oecd['PARENT_COUNTRY_CODE'] == code).sum()
        )

        oecd_restricted = oecd[oecd['NB_AFFILIATE_COUNTRIES'] >= breakdown_threshold].copy()

        return oecd_restricted.dropna()

    def show_scatterplots(self, breakdown_threshold=60):
        oecd_restricted = self.get_scatterplot_data(breakdown_threshold=breakdown_threshold)

        nb_countries = oecd_restricted['PARENT_COUNTRY_CODE'].nunique()

        ncols = 2
        nrows = int(nb_countries / ncols) + 1

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(17, 50))

        for parent_country, ax in zip(oecd_restricted['PARENT_COUNTRY_CODE'].unique(), axes.flatten()):

            restricted_df = oecd_restricted[oecd_restricted['PARENT_COUNTRY_CODE'] == parent_country].copy()

            restricted_df['SHARE_OF_UNRELATED_PARTY_REVENUES'] = (
                restricted_df['UNRELATED_PARTY_REVENUES'] / restricted_df['UNRELATED_PARTY_REVENUES'].sum()
            )

            restricted_df['SHARE_OF_GNI_2016'] = (
                restricted_df['GNI_2016'] / restricted_df['GNI_2016'].sum()
            )

            sns.scatterplot(
                x='SHARE_OF_GNI_2016',
                y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                hue='IS_TAX_HAVEN',
                data=restricted_df,
                ax=ax
            )

            ax.set_title(parent_country)

        plt.show()
