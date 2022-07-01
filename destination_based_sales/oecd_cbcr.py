"""
This module is used to load and preprocess aggregated and anonymized country-by-country data from the OECD. These pro-
vide the three revenue variables that we aim at distributing based on their approximative ultimate destination for head-
quarter countries other than the US.
"""


########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

import requests

from destination_based_sales.utils import impute_missing_continent_codes, CONTINENT_CODES_TO_IMPUTE_OECD_CBCR, \
    UK_CARIBBEAN_ISLANDS, online_path_to_geo_file, online_path_to_GNI_data, online_path_to_TH_list, url_to_data


########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')
path_to_GNI_data = os.path.join(path_to_dir, 'data', 'gross_national_income.csv')
path_to_tax_haven_list = os.path.join(path_to_dir, 'data', 'tax_havens.csv')
local_path_to_OECD_CbCR_data = os.path.join(path_to_dir, 'data', 'oecd_cbcr.csv')


########################################################################################################################
# --- Content

class CbCRPreprocessor:

    def __init__(
        self,
        year,
        breakdown_threshold,
        load_raw_data=True,
        load_data_online=False
    ):
        """
        TO BE COMPLETED.
        The instructions allowing to load and preprocess OECD data are organised in a Python class, CbCRPreprocessor.

        This is the instantiation function for this class. It requires several arguments:

        - "load_raw_data": a boolean indicating whether or not to directly load the data in a class attribute;

        - "load_data_online": a boolean indicating whether to fetch data online (if set to True) or locally from the
        "data" folder (if set to False).
        """
        self.year = year

        self.breakdown_threshold = breakdown_threshold

        if load_data_online:
            # If relevant, we construct the URL from which we can load the CSV country-by-country dataset
            self.url_base = 'http://stats.oecd.org/SDMX-JSON/data/'
            self.dataset_identifier = 'CBCR_TABLEI/'
            self.dimensions = 'ALL/'
            self.agency_name = 'OECD'

            # self.path_to_OECD_CbCR_data = (
            #     self.url_base + self.dataset_identifier + self.dimensions + self.agency_name + '?contenttype=csv'
            # )

            # For now, we rely on the files available on GitHub
            self.path_to_OECD_CbCR_data = url_to_data + 'oecd_cbcr.csv'

            self.path_to_geographies = online_path_to_geo_file
            self.path_to_GNI_data = online_path_to_GNI_data
            self.path_to_tax_haven_list = online_path_to_TH_list

        else:
            # Or we use the path to the local file
            self.path_to_OECD_CbCR_data = local_path_to_OECD_CbCR_data

            self.path_to_geographies = path_to_geographies
            self.path_to_GNI_data = path_to_GNI_data
            self.path_to_tax_haven_list = path_to_tax_haven_list

        self.continent_code_imputations = CONTINENT_CODES_TO_IMPUTE_OECD_CBCR

        # If relevant, we load the data in a dedicated class attribute
        if load_raw_data:
            self.data = pd.read_csv(self.path_to_OECD_CbCR_data)

        else:
            self.data = None

    def load_raw_data(self):
        """
        This class method allows to load the OECD's anonymized and aggregated data from the pre-constructed string path
        (see the instantiation function above), in case it has not been done in the instantiation function (after having
        set load_raw_data=False). Data are stored in a dedicated class attribute.
        """
        self.data = pd.read_csv(self.path_to_OECD_CbCR_data)

    def get_preprocessed_revenue_data(self):
        """
        This class method, relying on the pre-loaded data, applies a series of preprocessing steps to the raw data. The
        latter has indeed been loaded from the complete CSV file with a lot of non-relevant fields and a specific orga-
        nisation. In particular, we pivot the DataFrame to move from a long to a wide dataset.
        """
        if self.data is None:
            raise Exception('You must load the data with the dedicated method before you may run any computation.')

        cbcr = self.data.copy()

        # We keep the full sample, including loss-making entities
        cbcr = cbcr[cbcr['PAN'] == 'PANELA'].copy()

        # And focus on the year of interest
        cbcr = cbcr[cbcr['Year'] == self.year].copy()

        cbcr.drop(
            columns=['PAN', 'Grouping', 'Flag Codes', 'Flags', 'YEA', 'Year'],
            inplace=True
        )

        # We reshape the DataFrame from long into wide format, columns corresponding to the various financial variables
        cbcr_wide = cbcr.pivot(
            index=['COU', 'Ultimate Parent Jurisdiction', 'JUR', 'Partner Jurisdiction'],
            columns='Variable',
            values='Value'
        ).reset_index()

        # We limit ourselves to the relevant variables
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

        # We rename the "XKV" code used for Kosovo into "XXK", covered in the "geographies.csv" file
        oecd['PARENT_COUNTRY_CODE'] = oecd['PARENT_COUNTRY_CODE'].map(
            lambda country_code: {'XKV': 'XXK'}.get(country_code, country_code)
        )
        oecd['AFFILIATE_COUNTRY_CODE'] = oecd['AFFILIATE_COUNTRY_CODE'].map(
            lambda country_code: {'XKV': 'XXK'}.get(country_code, country_code)
        )

        # The "NB_AFFILIATE_COUNTRIES" column shows the number of partners reported by the corresponding parent country
        oecd['NB_AFFILIATE_COUNTRIES'] = oecd['PARENT_COUNTRY_CODE'].map(
            lambda code: (oecd['PARENT_COUNTRY_CODE'] == code).sum()
        )

        # We only prepare an adjusted sales mapping for parents that report a minimum number of partner jurisdictions
        oecd = oecd[oecd['NB_AFFILIATE_COUNTRIES'] >= self.breakdown_threshold].copy()

        # We add continent codes to the dataset from the "geographies.csv" file
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

        # We limit ourselves to 4 continents (Europe, Africa, America and Asia-Pacific)
        oecd['CONTINENT_CODE'] = oecd['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        oecd['CONTINENT_CODE'] = oecd['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        # In 2016, one row (India - Bouvet Island) which is full of 0s; we eliminate it
        oecd = oecd[oecd['CONTINENT_CODE'] != 'ATC'].copy()

        # Eventually, we rename the alpha-3 codes for continental aggregates of the type "Other Europe"
        oecd['AFFILIATE_COUNTRY_CODE'] = oecd['AFFILIATE_COUNTRY_CODE'].map(
            lambda country_code: {
                'OTE': 'OEUR',   # Other Europe
                'OAM': 'OAMR',   # Other Americas
                'OAF': 'OAFR',   # Other Africa
                'OAS': 'OASIAOCN'   # Other Asia
            }.get(country_code, country_code)
        )

        return oecd.copy()

    def get_revenue_data_with_tax_havens(self):
        """
        Building upon the "get_preprocessed_revenue_data" method, we add an indicator variable - IS_TAX_HAVEN - valued
        to 1 if the country code is identified in the TWZ list or to 0 if not.
        """
        oecd = self.get_preprocessed_revenue_data()

        tax_havens = pd.read_csv(self.path_to_tax_haven_list)

        oecd['IS_TAX_HAVEN'] = oecd['AFFILIATE_COUNTRY_CODE'].isin(tax_havens['CODE'].unique())

        return oecd.copy()

    def get_revenue_data_with_GNI(self):
        """
        Building upon the "get_revenue_data_with_tax_havens" method, we add the Gross National Income (GNI) variables
        and return the resulting dataset.
        """
        oecd = self.get_revenue_data_with_tax_havens()

        # We load the data from the CSV file
        gross_national_income = pd.read_csv(self.path_to_GNI_data, delimiter=';')

        # All numeric columns in the file must be slightly preprocessed to be considered as floats
        for column in gross_national_income.columns[2:]:

            gross_national_income[column] = gross_national_income[column].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            gross_national_income[column] = gross_national_income[column].astype(float)

        # We add GNI variables to the main dataset
        oecd = oecd.merge(
            gross_national_income[['COUNTRY_CODE', 'GNI_2016']].copy(),
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

        oecd.drop(columns=['COUNTRY_CODE'], inplace=True)

        return oecd.copy()

    def get_scatterplot_data(self):
        """
        This method allows to output the relevant data for building the scatterplots showing, for each parent jurisdi-
        ction the relationship between partner jurisdictions’ share of multinational companies’ foreign unrelated-party
        revenues and their share of the Gross National Income (GNI) in the dataset.
        """
        oecd = self.get_revenue_data_with_GNI()

        # We eliminate foreign jurisdictions totals and rows corresponding to domestic activities
        oecd = oecd[oecd['AFFILIATE_COUNTRY_NAME'] != 'Foreign Jurisdictions Total'].copy()
        oecd = oecd[oecd['PARENT_COUNTRY_CODE'] != oecd['AFFILIATE_COUNTRY_CODE']].copy()

        # The "NB_AFFILIATE_COUNTRIES" column shows the number of partners reported by the corresponding parent country
        oecd['NB_AFFILIATE_COUNTRIES'] = oecd['PARENT_COUNTRY_CODE'].map(
            lambda code: (oecd['PARENT_COUNTRY_CODE'] == code).sum()
        )

        # We only show the scatterplot for parent countries that report a minimum number of partner jurisdictions
        oecd_restricted = oecd[oecd['NB_AFFILIATE_COUNTRIES'] >= self.breakdown_threshold].copy()

        return oecd_restricted.dropna()

    def show_scatterplots(self):
        """
        Building upon the "get_scatterplot_data" method, this method allows to output the scatterplots showing, for each
        parent jurisdiction in the OECD's country-by-country statistics, the relationship between partner jurisdictions’
        share of multinational companies’ foreign unrelated-party revenues and their share of the Gross National Income
        (GNI) in the dataset.
        """

        # We fetch the data to be displayed within the scatterplots thanks to the dedicated method
        oecd_restricted = self.get_scatterplot_data()

        # The number of parent countries determines the number of graphs to construct
        nb_countries = oecd_restricted['PARENT_COUNTRY_CODE'].nunique()

        ncols = 2
        nrows = int(nb_countries / ncols) + 1

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(17, 50))

        # We plot the graph for each parent country in the preprocessed country-by-country statistics
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
