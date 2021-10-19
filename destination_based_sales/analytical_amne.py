"""
This module is used to load and preprocess data from the OECD's Analytical AMNE database. The latter allows, for non-US
multinational companies, to separate the revenue variables into sales to the host country and sales directed to another
jurisdiction. These are split between sales to the headquarter country and sales to any other country based on BEA data.
"""


########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

from destination_based_sales.bea import BEADataPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor
from destination_based_sales.utils import compute_foreign_owned_gross_output


########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_analytical_amne = os.path.join(path_to_dir, 'data', 'analytical_amne.xlsx')
path_to_analytical_amne_domestic = os.path.join(path_to_dir, 'data', 'analytical_amne_domesticMNEs.xlsx')

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


########################################################################################################################
# --- Content

class AnalyticalAMNEPreprocessor:

    def __init__(
        self,
        path_to_analytical_amne=path_to_analytical_amne,
        path_to_analytical_amne_domestic=path_to_analytical_amne_domestic,
        path_to_geographies=path_to_geographies,
        load_OECD_data=True
    ):
        """
        The logic used to load and preprocess the OECD's Analytical AMNE data is encapsulated in a Python class, Analy-
        ticalAMNEPreprocessor. This is the instantiation method of this class, which takes as arguments:

        - the string path to the main Excel file of the Analytical AMNE database, downloaded under the name "analytical_
        amne.xlsx";

        - the string path to the complementary Excel file, focused on the activities of domestic firms (multinational
        enterprises or purely local companies) and downloaded under the name "analytical_amne_domesticMNEs.xlsx";

        - the string path to the "geographies.csv", used to add ISO country codes to the OECD's data;

        - a boolean, "load_OECD_data", indicating whether to save the OECD's aggregated and anonymized country-by-
        country data in a dedicated class attribute.
        """
        self.path_to_analytical_amne = path_to_analytical_amne
        self.tab_1 = 'GO bilateral'
        self.tab_2 = 'GVA EXGR IMGR'

        self.path_to_analytical_amne_domestic = path_to_analytical_amne_domestic
        self.domestic_aamne_tab = 'MNE GO GVA EXGR IMGR'

        self.bea_processor = BEADataPreprocessor(year=2016)
        self.bea = self.bea_processor.load_final_data()

        # Depending on the boolean passed as argument, we load the OECD's CbCR data or not
        if load_OECD_data:
            self.cbcr_preprocessor = CbCRPreprocessor()
            self.oecd = self.cbcr_preprocessor.get_preprocessed_revenue_data()

        else:
            self.oecd = None

        self.path_to_geographies = path_to_geographies

    def load_OECD_CbCR_data(self):
        """
        If, when instantiating the AnalyticalAMNEPreprocessor object, the option "load_OECD_data=False" was chosen, this
        method allows to load the OECD's country-by-country data and to save it in an "oecd" attribute.
        """
        self.cbcr_preprocessor = CbCRPreprocessor()
        self.oecd = self.cbcr_preprocessor.get_preprocessed_revenue_data()

    def load_clean_foreign_analytical_amne_data(self):
        """
        This method allows to load and clean the data from the second tab of the "analytical_amne.xlsx" file. This file
        provides information and estimations relative to the activities of foreign-owned and domestically-owned compa-
        nies in a sample of countries, from 2005 to 2016.

        For each country, the dataset indicates the gross value-added, exports and imports of the foreign subsidiaries
        of multinational enterprises (foreign-owned companies) and of locally-owned companies. These variables are fur-
        ther broken down by industry.

        We concentrate on 2016 data to align with country-by-country statistics and only consider the rows corresponding
        to foreign-owned companies. We further sum the three variables over all sectors of activity.
        """

        # We read the second tab of the spreadsheet
        aamne = pd.read_excel(
            self.path_to_analytical_amne,
            sheet_name=self.tab_2,
            engine='openpyxl'
        )

        aamne.drop(
            columns=['flag_gva', 'flag_exgr', 'flag_imgr'],
            inplace=True
        )

        aamne = aamne[aamne['year'] == 2016].copy()   # We focus on 2016 data to align with CbCR data
        aamne = aamne[aamne['own'] == 'F'].copy()     # And on the activities of foreign-owned companies
        aamne = aamne[aamne['cou'] != 'ROW'].copy()

        aamne.drop(
            columns=['year', 'own'],
            inplace=True
        )

        aamne.reset_index(drop=True, inplace=True)

        # We consider all sectors of activity and therefore group by countries
        aamne_grouped = aamne.groupby('cou').sum().reset_index()

        aamne_grouped.rename(
            columns={
                'cou': 'COUNTRY_CODE',
                'gva': 'GROSS_VALUE_ADDED',
                'exgr': 'EXPORTS',
                'imgr': 'IMPORTS'
            },
            inplace=True
        )

        return aamne_grouped.copy()

    def load_clean_bilateral_gross_output_data(self):
        """
        This method allows to load and clean the data from the first tab of the "analytical_amne.xlsx" file. Indeed, the
        dataset also provides a bilateral mapping of gross output between host countries and the jurisdictions where
        companies are ultimately headquartered.

        Focusing on 2016 data, we sum these results over all sectors of activity and all jurisdictions of ultimate
        ownership, so as to complement the previous dataset with gross output. As explained in more details in the PDF
        report, we also compute the total gross output excluding US-headquartered multinational enterprises.

        This method relies on a function, "compute_foreign_owned_gross_output", defined in the "utils.py" file.
        """

        # We read the first tab of the spreadsheet
        gross_output = pd.read_excel(
            self.path_to_analytical_amne,
            sheet_name=self.tab_1,
            engine='openpyxl'
        )

        gross_output = gross_output[gross_output['year'] == 2016].copy()   # We focus on 2016 data to align with CbCR
        gross_output = gross_output[gross_output['cou'] != 'ROW'].copy()

        gross_output = gross_output.drop(columns='year').groupby('cou').sum().reset_index()

        # Relying on a function defined in "utils.py", we compute for each country the gross output registered there by
        # foreign-owned multinationals, including the US-headquartered ones
        gross_output['GROSS_OUTPUT_INCL_US'] = gross_output.apply(
            lambda row: compute_foreign_owned_gross_output(row, include_US=True),
            axis=1
        )

        # We do the same, this time excluding US-owned multinational companies
        gross_output['GROSS_OUTPUT_EXCL_US'] = gross_output.apply(
            lambda row: compute_foreign_owned_gross_output(row, include_US=False),
            axis=1
        )

        # We restrict the dataset to the necessary variables
        gross_output = gross_output[['cou', 'GROSS_OUTPUT_INCL_US', 'GROSS_OUTPUT_EXCL_US']].copy()

        gross_output.rename(
            columns={
                'cou': 'COUNTRY_CODE'
            },
            inplace=True
        )

        return gross_output.copy()

    def get_merged_foreign_analytical_amne_data(self):
        """
        This method allows to construct a DataFrame that combines all the relevant information on foreign-owned compa-
        nies, loaded and preprocessed from the "analytical_amne.xlsx" file.
        """

        foreign_aamne = self.load_clean_foreign_analytical_amne_data()
        gross_output = self.load_clean_bilateral_gross_output_data()

        foreign_aamne = foreign_aamne.merge(
            gross_output,
            how='inner',
            on='COUNTRY_CODE'
        )

        return foreign_aamne.copy()

    def get_unextended_foreign_analytical_amne_data(self):
        """
        With the three methods defined above, we have obtained, for each in-sample country, information on the gross
        output, gross value-added exports and imports that foreign-owned companies register. We want to deduce an ap-
        proximation of the following three aggregates:

        - sales of foreign-owned companies to the host country;
        - sales of foreign-owned companies to their headquarter country;
        - and sales of foreign-owned companies to any other country.

        The first one is approximated as (gross output - exports).

        The second one is approximated as (exports * ratio of exports to the headquarter country). For each country con-
        sidered, the ratio is assumed to be the same for all foreign-owned companies (simplifying assumption made neces-
        sary because of the limited data at hand) and we therefore draw it from BEA data. If the country is absent from
        BEA data, we take the ratio averaged across all countries in the BEA.

        The third one is simply computed as (exports - sales to the heaquarter country).

        Eventually, as we know that this distribution will be used to split the revenue variables of non-US multinatio-
        nal companies, we exclude the US from these computations. We already have a gross output variable that excludes
        the US and we approximate US exports from the BEA data.
        """
        bea = self.bea.copy()

        #-- Average ratios used for countries that do not appear in the BEA data

        # Ratio of exports to total sales, used to deduce non-US exports
        self.imputation_exports_ratio = (bea['TOTAL_OTHER_COUNTRY'] + bea['TOTAL_US']).sum() / bea['TOTAL'].sum()

        # Ratio of sales to the headquarter country to total sales outside the host country
        self.imputation_exports_to_US_ratio = (
            bea['TOTAL_US'].sum() / (bea['TOTAL_OTHER_COUNTRY'] + bea['TOTAL_US']).sum()
        )

        #-- Same computation on a per-country basis, used for countries that appear in both Analytical AMNE and BEA data
        bea['BEA_EXPORTS_RATIO'] = (bea['TOTAL_OTHER_COUNTRY'] + bea['TOTAL_US']) / bea['TOTAL']
        bea['BEA_EXPORTS_TO_US_RATIO'] = bea['TOTAL_US'] / (bea['TOTAL_OTHER_COUNTRY'] + bea['TOTAL_US'])

        # Using the method defined above, we get all the information on the activities of foreign-owned firms
        merged_df = self.get_merged_foreign_analytical_amne_data()

        # Activities in the US and in other countries are treated differently, we separate the two
        us_extract = merged_df[merged_df['COUNTRY_CODE'] == 'USA'].copy()
        merged_df = merged_df[merged_df['COUNTRY_CODE'] != 'USA'].copy()

        # We add the US export and sales-to-headquarter ratios to the main DataFrame
        merged_df = merged_df.merge(
            bea[['CODE', 'BEA_EXPORTS_RATIO', 'BEA_EXPORTS_TO_US_RATIO']].copy(),
            how='left',
            left_on='COUNTRY_CODE', right_on='CODE'
        )

        merged_df.drop(columns=['CODE'], inplace=True)

        # We replace missing values due to the absence of some countries from BEA data, using the average ratios
        merged_df['BEA_EXPORTS_RATIO'] = merged_df['BEA_EXPORTS_RATIO'].fillna(self.imputation_exports_ratio)
        merged_df['BEA_EXPORTS_TO_US_RATIO'] = merged_df['BEA_EXPORTS_TO_US_RATIO'].fillna(
            self.imputation_exports_to_US_ratio
        )

        # We compute a proxy for the exports of non-US foreign-owned firms using the BEA export ratios
        merged_df['EXPORTS_EXCL_US'] = (
            merged_df['EXPORTS'] - (   # Exports of all foreign-owned firms
                merged_df['GROSS_OUTPUT_INCL_US'] - merged_df['GROSS_OUTPUT_EXCL_US']   # Gross output of US-owned firms
            ) * merged_df['BEA_EXPORTS_RATIO']   # Export ratio of US firms
        )

        # We rename columns and filter out the ones that are not relevant anymore
        merged_df.drop(
            columns=['GROSS_VALUE_ADDED', 'EXPORTS', 'IMPORTS', 'GROSS_OUTPUT_INCL_US', 'BEA_EXPORTS_RATIO'],
            inplace=True
        )

        merged_df.rename(
            columns={
                'GROSS_OUTPUT_EXCL_US': 'TURNOVER',
                'EXPORTS_EXCL_US': 'EXPORTS',
            },
            inplace=True
        )

        # We deduce from previous computations the three proxies that we are looking for
        merged_df['SALES_TO_AFFILIATE_COUNTRY'] = merged_df['TURNOVER'] - merged_df['EXPORTS']
        merged_df['SALES_TO_HEADQUARTER_COUNTRY'] = merged_df['EXPORTS'] * merged_df['BEA_EXPORTS_TO_US_RATIO']
        merged_df['SALES_TO_OTHER_COUNTRY'] = merged_df['EXPORTS'] - merged_df['SALES_TO_HEADQUARTER_COUNTRY']

        merged_df.drop(
            columns=['TURNOVER', 'EXPORTS'],
            inplace=True
        )

        # We compute equivalent aggregates for the US
        us_extract['SALES_TO_AFFILIATE_COUNTRY'] = us_extract['GROSS_OUTPUT_INCL_US'] - us_extract['EXPORTS']
        us_extract['SALES_TO_HEADQUARTER_COUNTRY'] = us_extract['EXPORTS'] * self.imputation_exports_to_US_ratio
        us_extract['SALES_TO_OTHER_COUNTRY'] = us_extract['EXPORTS'] - us_extract['SALES_TO_HEADQUARTER_COUNTRY']

        us_extract.drop(
            columns=['GROSS_VALUE_ADDED', 'EXPORTS', 'IMPORTS', 'GROSS_OUTPUT_INCL_US', 'GROSS_OUTPUT_EXCL_US'],
            inplace=True
        )

        # And we concatenate the resulting DataFrames
        merged_df = pd.concat(
            [merged_df, us_extract],
            axis=0
        )

        return merged_df.reset_index(drop=True)

    def get_extended_foreign_analytical_amne_data(self):
        """
        Building upon the previous method, we extend the dataset to all the partner countries that appear in the OECD's
        aggregated and anonymized country-by-country data. Imputations for countries that are absent from the Analytical
        AMNE database relies on continental aggregates (possible since we are looking for sales ratios and not absolute
        amounts to split the revenue variables.)
        """
        if self.oecd is None:
            raise Exception(
                "Before you may use this method, you have to load the OECD's CbCR data with the dedicated method."
            )

        # We get the unextended dataset
        aamne_foreign = self.get_unextended_foreign_analytical_amne_data()

        geographies = pd.read_csv(self.path_to_geographies)

        # We complement it with continent codes
        aamne_foreign = aamne_foreign.merge(
            geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='COUNTRY_CODE', right_on='CODE'
        )

        # We restrict the CONTINENT_CODE columns to the 4 usual codes
        aamne_foreign.drop(columns=['CODE'], inplace=True)

        aamne_foreign['CONTINENT_CODE'] = aamne_foreign['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        aamne_foreign['CONTINENT_CODE'] = aamne_foreign['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        #-- Preparing the continental imputations

        # We build the dictionary allowing us to apply the continental imputations
        continent_imputations = {}

        columns_of_interest = [
            'SALES_TO_AFFILIATE_COUNTRY', 'SALES_TO_HEADQUARTER_COUNTRY', 'SALES_TO_OTHER_COUNTRY'
        ]

        for continent in aamne_foreign['CONTINENT_CODE'].unique():
            # For each of the 4 unique continent codes, the value is a dictionary
            continent_imputations[continent] = {}

            restricted_df = aamne_foreign[aamne_foreign['CONTINENT_CODE'] == continent].copy()

            # Total sales registered by foreign-owned companies in this continent
            denominator = restricted_df[columns_of_interest].sum().sum()

            # The dictionary associated with each continent gives the percentage of sales that are associated with each
            # type of destination (host country, headquarter country, any other country)
            for column in columns_of_interest:
                suffix = column.replace('SALES_', '')
                new_column = 'PERC_' + suffix

                numerator = restricted_df[column].sum()

                continent_imputations[continent][new_column] = numerator / denominator

        # We complement the dictionary used for continental imputations
        continent_imputations['OTHER_GROUPS'] = {
            'PERC_TO_AFFILIATE_COUNTRY': 0,
            'PERC_TO_HEADQUARTER_COUNTRY': self.imputation_exports_to_US_ratio,
            'PERC_TO_OTHER_COUNTRY': 1 - self.imputation_exports_to_US_ratio
        }

        #-- Moving from absolute amounts to sales percentages

        # This will serve as a denominator in the computation of sales percentages
        aamne_foreign['TOTAL_SALES'] = aamne_foreign[columns_of_interest].sum(axis=1)

        new_columns = []

        # We add three columns to the DataFrame that correspond to sales percentages instead of the absolute amounts
        for column in columns_of_interest:
            suffix = column.replace('SALES_', '')
            new_column = 'PERC_' + suffix
            new_columns.append(new_column)

            aamne_foreign[new_column] = aamne_foreign[column] / aamne_foreign['TOTAL_SALES']

        # We drop the absolute amounts that are not necessary anymore
        aamne_foreign.drop(
            columns=columns_of_interest + ['TOTAL_SALES', 'CONTINENT_CODE'],
            inplace=True
        )

        #-- Reconstituting the extended DataFrame

        # We start from the list (a DataFrame with continent codes) of unique partner jurisdictions in CbCR data
        partner_jurisdictions = self.oecd[
            ['AFFILIATE_COUNTRY_CODE', 'CONTINENT_CODE']
        ].drop_duplicates()

        # We add sales percentages from the unextended dataset; missing values are created whenever the affiliate / host
        # country is in CbCR data but not in the Analytical AMNE data
        partner_jurisdictions = partner_jurisdictions.merge(
            aamne_foreign,
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

        # We apply the continental imputation for countries that are absent from the Analytical AMNE database
        for new_column in new_columns:
            partner_jurisdictions[new_column] = partner_jurisdictions.apply(
                lambda row: (
                    continent_imputations[row['CONTINENT_CODE']][new_column]
                    if np.isnan(row[new_column]) else row[new_column]
                ),
                axis=1
            )

        partner_jurisdictions.drop(
            columns=['CONTINENT_CODE', 'COUNTRY_CODE', 'BEA_EXPORTS_TO_US_RATIO'],
            inplace=True
        )

        return partner_jurisdictions.copy()

    def load_clean_domestic_analytical_amne_data(self):
        """
        This method allows to load and clean data from the complementary Excel file of the OECD's Analytical AMNE data-
        base, "analytical_amne_domesticMNEs.xlsx". It provides information on and estimations of the gross output, gross
        value-added, exports and imports of domestically-owned companies.

        It allows to distinguish between the domestic branches of multinational enterprises and purely local firms.
        Here also, variables are further broken down by industry.

        We concentrate on 2016 data and only consider the rows corresponding to the activities of multinational enter-
        prises. We sum the four variables over all sectors of activity.
        """

        # We read the Excel file; paths are defined when instantiating the AnalyticalAMNEPreprocessor object
        aamne_domestic = pd.read_excel(
            self.path_to_analytical_amne_domestic,
            sheet_name=self.domestic_aamne_tab,
            engine='openpyxl'
        )

        aamne_domestic = aamne_domestic[aamne_domestic['year'] == 2016].copy()   # We focus on 2016 data
        aamne_domestic = aamne_domestic[aamne_domestic['own'] == 'MNE'].copy()   # And on multinational companies
        aamne_domestic = aamne_domestic[aamne_domestic['cou'] != 'ROW'].copy()

        aamne_domestic.drop(
            columns=['flag_go', 'flag_gva', 'flag_exgr', 'flag_imgr', 'year', 'own'],
            inplace=True
        )

        # We consider all sectors of activity together and thus group by host country
        aamne_domestic = aamne_domestic.groupby('cou').sum().reset_index()

        aamne_domestic.rename(
            columns={
                'cou': 'COUNTRY_CODE',
                'go': 'GROSS_OUTPUT',
                'gva': 'GROSS_VALUE_ADDED',
                'exgr': 'EXPORTS',
                'imgr': 'IMPORTS'
            },
            inplace=True
        )

        return aamne_domestic.copy()

    def get_unextended_domestic_analytical_amne_data(self):
        """
        Equivalently to the "get_unextended_foreign_analytical_amne_data" method, this method allows to move from infor-
        mation on gross output, gross value-added, etc. to proxies of sales to the host / headquarter country and sales
        to any other country. Domestic sales are simply defined as (gross output - exports).
        """
        aamne_domestic = self.load_clean_domestic_analytical_amne_data()

        aamne_domestic['DOMESTIC_SALES'] = (
            aamne_domestic['GROSS_OUTPUT'] - aamne_domestic['EXPORTS']
        )
        aamne_domestic['SALES_TO_OTHER_COUNTRY'] = aamne_domestic['EXPORTS'].values

        aamne_domestic.drop(
            columns=['GROSS_OUTPUT', 'GROSS_VALUE_ADDED', 'EXPORTS', 'IMPORTS'],
            inplace=True
        )

        return aamne_domestic.copy()

    def get_extended_domestic_analytical_amne_data(self):
        """
        Building upon the previous method, this method is used (i) to move from sales proxies in absolute amounts to
        sales percentages and (ii) to extend the dataset to the parent countries in CbCR data that are absent from the
        Analytical AMNE database, via continental imputation.
        """
        if self.oecd is None:
            raise Exception(
                "Before you may use this method, you have to load the OECD's CbCR data with the dedicated method."
            )

        # We start from the unextended dataset
        aamne_domestic = self.get_unextended_domestic_analytical_amne_data()

        # We merge it with the geographies DataFrame to add continent codes
        geographies = pd.read_csv(self.path_to_geographies)

        aamne_domestic = aamne_domestic.merge(
            geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='COUNTRY_CODE', right_on='CODE'
        )

        aamne_domestic.drop(columns=['CODE'], inplace=True)

        # We restrict continent codes to a set of 4 codes
        aamne_domestic['CONTINENT_CODE'] = aamne_domestic['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        aamne_domestic['CONTINENT_CODE'] = aamne_domestic['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        #-- Preparing the continental imputation

        # We build the dictionary allowing us to apply the continental imputations
        continent_imputations = {}

        columns_of_interest = [
            'DOMESTIC_SALES', 'SALES_TO_OTHER_COUNTRY'
        ]

        for continent in aamne_domestic['CONTINENT_CODE'].unique():
            # For each of the 4 unique continent codes, the value is a dictionary
            continent_imputations[continent] = {}

            restricted_df = aamne_domestic[aamne_domestic['CONTINENT_CODE'] == continent].copy()

            # Total sales registered by domestic multinational companies in this continent
            denominator = restricted_df[columns_of_interest].sum().sum()

            for column in columns_of_interest:
                suffix = column.replace('SALES_', '')
                new_column = 'PERC_' + suffix

                numerator = restricted_df[column].sum()

                # The dictionary associated with each continent gives the percentage of sales that are associated with
                # each type of destination (host / headquarter country, any other country)
                continent_imputations[continent][new_column] = numerator / denominator

        #-- Moving from absolute amounts to sales percentages

        # This will serve as a denominator in the computation of sales percentages
        aamne_domestic['TOTAL_SALES'] = aamne_domestic[columns_of_interest].sum(axis=1)

        new_columns = []

        # We add two columns to the DataFrame that correspond to sales percentages instead of the absolute amounts
        for column in columns_of_interest:
            suffix = column.replace('SALES_', '')
            new_column = 'PERC_' + suffix
            new_columns.append(new_column)

            aamne_domestic[new_column] = aamne_domestic[column] / aamne_domestic['TOTAL_SALES']

        # We drop the absolute amounts that are not necessary anymore
        aamne_domestic.drop(
            columns=columns_of_interest + ['TOTAL_SALES', 'CONTINENT_CODE'],
            inplace=True
        )

        #-- Reconstituting the extended DataFrame

        # We start from the list (a DataFrame with continent codes) of unique parent jurisdictions in CbCR data
        parent_jurisdictions = self.oecd[
            self.oecd['PARENT_COUNTRY_CODE'] == self.oecd['AFFILIATE_COUNTRY_CODE']
        ][['PARENT_COUNTRY_CODE', 'CONTINENT_CODE']].drop_duplicates()

        # We add sales percentages from the unextended dataset; missing values are created whenever the parent country
        # is in CbCR data but not in the Analytical AMNE data
        parent_jurisdictions = parent_jurisdictions.merge(
            aamne_domestic,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

        # We apply the continental imputation for countries that are absent from the Analytical AMNE database
        for new_column in new_columns:
            parent_jurisdictions[new_column] = parent_jurisdictions.apply(
                lambda row: (
                    continent_imputations[row['CONTINENT_CODE']][new_column]
                    if np.isnan(row[new_column]) else row[new_column]
                ),
                axis=1
            )

        parent_jurisdictions.drop(
            columns=['CONTINENT_CODE', 'COUNTRY_CODE'],
            inplace=True
        )

        return parent_jurisdictions.copy()
