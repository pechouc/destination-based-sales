import os

import numpy as np
import pandas as pd

from destination_based_sales.bea import BEADataPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor
from destination_based_sales.utils import compute_foreign_owned_gross_output


path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_analytical_amne = os.path.join(path_to_dir, 'data', 'analytical_amne.xlsx')
path_to_analytical_amne_domestic = os.path.join(path_to_dir, 'data', 'analytical_amne_domesticMNEs.xlsx')

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class AnalyticalAMNEPreprocessor:

    def __init__(
        self,
        path_to_analytical_amne=path_to_analytical_amne,
        path_to_analytical_amne_domestic=path_to_analytical_amne_domestic,
        path_to_geographies=path_to_geographies,
        load_OECD_data=True
    ):
        self.path_to_analytical_amne = path_to_analytical_amne
        self.tab_1 = 'GO bilateral'
        self.tab_2 = 'GVA EXGR IMGR'

        self.path_to_analytical_amne_domestic = path_to_analytical_amne_domestic
        self.domestic_aamne_tab = 'MNE GO GVA EXGR IMGR'

        self.bea_processor = BEADataPreprocessor(year=2016)
        self.bea = self.bea_processor.load_final_data()

        if load_OECD_data:
            self.cbcr_preprocessor = CbCRPreprocessor()
            self.oecd = self.cbcr_preprocessor.get_preprocessed_revenue_data()
        else:
            self.oecd = None

        self.path_to_geographies = path_to_geographies

    def load_OECD_CbCR_data(self):
        self.cbcr_preprocessor = CbCRPreprocessor()
        self.oecd = self.cbcr_preprocessor.get_preprocessed_revenue_data()

    def load_clean_foreign_analytical_amne_data(self):
        aamne = pd.read_excel(
            self.path_to_analytical_amne,
            sheet_name=self.tab_2,
            engine='openpyxl'
        )

        aamne.drop(
            columns=['flag_gva', 'flag_exgr', 'flag_imgr'],
            inplace=True
        )

        aamne = aamne[aamne['year'] == 2016].copy()
        aamne = aamne[aamne['own'] == 'F'].copy()
        aamne = aamne[aamne['cou'] != 'ROW'].copy()

        aamne.drop(
            columns=['year', 'own'],
            inplace=True
        )

        aamne.reset_index(drop=True, inplace=True)

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
        gross_output = pd.read_excel(
            self.path_to_analytical_amne,
            sheet_name=self.tab_1,
            engine='openpyxl'
        )

        gross_output = gross_output[gross_output['year'] == 2016].copy()
        gross_output = gross_output[gross_output['cou'] != 'ROW'].copy()

        gross_output = gross_output.drop(columns='year').groupby('cou').sum().reset_index()

        gross_output['GROSS_OUTPUT_INCL_US'] = gross_output.apply(
            lambda row: compute_foreign_owned_gross_output(row, include_US=True),
            axis=1
        )

        gross_output['GROSS_OUTPUT_EXCL_US'] = gross_output.apply(
            lambda row: compute_foreign_owned_gross_output(row, include_US=False),
            axis=1
        )

        gross_output = gross_output[['cou', 'GROSS_OUTPUT_INCL_US', 'GROSS_OUTPUT_EXCL_US']].copy()

        gross_output.rename(
            columns={
                'cou': 'COUNTRY_CODE'
            },
            inplace=True
        )

        return gross_output.copy()

    def get_merged_foreign_analytical_amne_data(self):
        foreign_aamne = self.load_clean_foreign_analytical_amne_data()
        gross_output = self.load_clean_bilateral_gross_output_data()

        foreign_aamne = foreign_aamne.merge(
            gross_output,
            how='inner',
            on='COUNTRY_CODE'
        )

        return foreign_aamne.copy()

    def get_unextended_foreign_analytical_amne_data(self):
        bea = self.bea.copy()

        self.imputation_exports_ratio = (bea['TOTAL_OTHER_COUNTRY'] + bea['TOTAL_US']).sum() / bea['TOTAL'].sum()
        self.imputation_exports_to_US_ratio = (
            bea['TOTAL_US'].sum() / (bea['TOTAL_OTHER_COUNTRY'] + bea['TOTAL_US']).sum()
        )

        bea['BEA_EXPORTS_RATIO'] = (bea['TOTAL_OTHER_COUNTRY'] + bea['TOTAL_US']) / bea['TOTAL']
        bea['BEA_EXPORTS_TO_US_RATIO'] = bea['TOTAL_US'] / (bea['TOTAL_OTHER_COUNTRY'] + bea['TOTAL_US'])

        merged_df = self.get_merged_foreign_analytical_amne_data()

        us_extract = merged_df[merged_df['COUNTRY_CODE'] == 'USA'].copy()
        merged_df = merged_df[merged_df['COUNTRY_CODE'] != 'USA'].copy()

        merged_df = merged_df.merge(
            bea[['CODE', 'BEA_EXPORTS_RATIO', 'BEA_EXPORTS_TO_US_RATIO']].copy(),
            how='left',
            left_on='COUNTRY_CODE', right_on='CODE'
        )

        merged_df.drop(columns=['CODE'], inplace=True)

        merged_df['BEA_EXPORTS_RATIO'] = merged_df['BEA_EXPORTS_RATIO'].fillna(self.imputation_exports_ratio)
        merged_df['BEA_EXPORTS_TO_US_RATIO'] = merged_df['BEA_EXPORTS_TO_US_RATIO'].fillna(
            self.imputation_exports_to_US_ratio
        )

        merged_df['EXPORTS_EXCL_US'] = (
            merged_df['EXPORTS'] - (
                merged_df['GROSS_OUTPUT_INCL_US'] - merged_df['GROSS_OUTPUT_EXCL_US']
            ) * merged_df['BEA_EXPORTS_RATIO']
        )

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

        merged_df['SALES_TO_AFFILIATE_COUNTRY'] = merged_df['TURNOVER'] - merged_df['EXPORTS']
        merged_df['SALES_TO_HEADQUARTER_COUNTRY'] = merged_df['EXPORTS'] * merged_df['BEA_EXPORTS_TO_US_RATIO']
        merged_df['SALES_TO_OTHER_COUNTRY'] = merged_df['EXPORTS'] - merged_df['SALES_TO_HEADQUARTER_COUNTRY']

        merged_df.drop(
            columns=['TURNOVER', 'EXPORTS'],
            inplace=True
        )

        us_extract['SALES_TO_AFFILIATE_COUNTRY'] = us_extract['GROSS_OUTPUT_INCL_US'] - us_extract['EXPORTS']
        us_extract['SALES_TO_HEADQUARTER_COUNTRY'] = us_extract['EXPORTS'] * self.imputation_exports_to_US_ratio
        us_extract['SALES_TO_OTHER_COUNTRY'] = us_extract['EXPORTS'] - us_extract['SALES_TO_HEADQUARTER_COUNTRY']

        us_extract.drop(
            columns=['GROSS_VALUE_ADDED', 'EXPORTS', 'IMPORTS', 'GROSS_OUTPUT_INCL_US', 'GROSS_OUTPUT_EXCL_US'],
            inplace=True
        )

        merged_df = pd.concat(
            [merged_df, us_extract],
            axis=0
        )

        return merged_df.reset_index(drop=True)

    def get_extended_foreign_analytical_amne_data(self):
        if self.oecd is None:
            raise Exception(
                "Before you may use this method, you have to load the OECD's CbCR data with the dedicated method."
            )

        aamne_foreign = self.get_unextended_foreign_analytical_amne_data()

        geographies = pd.read_csv(self.path_to_geographies)

        aamne_foreign = aamne_foreign.merge(
            geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='COUNTRY_CODE', right_on='CODE'
        )

        aamne_foreign.drop(columns=['CODE'], inplace=True)

        aamne_foreign['CONTINENT_CODE'] = aamne_foreign['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        aamne_foreign['CONTINENT_CODE'] = aamne_foreign['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        continent_imputations = {}

        columns_of_interest = [
            'SALES_TO_AFFILIATE_COUNTRY', 'SALES_TO_HEADQUARTER_COUNTRY', 'SALES_TO_OTHER_COUNTRY'
        ]

        for continent in aamne_foreign['CONTINENT_CODE'].unique():
            continent_imputations[continent] = {}

            restricted_df = aamne_foreign[aamne_foreign['CONTINENT_CODE'] == continent].copy()

            denominator = restricted_df[columns_of_interest].sum().sum()

            for column in columns_of_interest:
                suffix = column.replace('SALES_', '')
                new_column = 'PERC_' + suffix

                numerator = restricted_df[column].sum()

                continent_imputations[continent][new_column] = numerator / denominator

        continent_imputations['OTHER_GROUPS'] = {
            'PERC_TO_AFFILIATE_COUNTRY': 0,
            'PERC_TO_HEADQUARTER_COUNTRY': self.imputation_exports_to_US_ratio,
            'PERC_TO_OTHER_COUNTRY': 1 - self.imputation_exports_to_US_ratio
        }

        aamne_foreign['TOTAL_SALES'] = aamne_foreign[columns_of_interest].sum(axis=1)

        new_columns = []

        for column in columns_of_interest:
            suffix = column.replace('SALES_', '')
            new_column = 'PERC_' + suffix
            new_columns.append(new_column)

            aamne_foreign[new_column] = aamne_foreign[column] / aamne_foreign['TOTAL_SALES']

        aamne_foreign.drop(
            columns=columns_of_interest + ['TOTAL_SALES', 'CONTINENT_CODE'],
            inplace=True
        )

        partner_jurisdictions = self.oecd[
            ['AFFILIATE_COUNTRY_CODE', 'CONTINENT_CODE']
        ].drop_duplicates()

        partner_jurisdictions = partner_jurisdictions.merge(
            aamne_foreign,
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

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
        aamne_domestic = pd.read_excel(
            self.path_to_analytical_amne_domestic,
            sheet_name=self.domestic_aamne_tab,
            engine='openpyxl'
        )

        aamne_domestic = aamne_domestic[aamne_domestic['year'] == 2016].copy()
        aamne_domestic = aamne_domestic[aamne_domestic['own'] == 'MNE'].copy()
        aamne_domestic = aamne_domestic[aamne_domestic['cou'] != 'ROW'].copy()

        aamne_domestic.drop(
            columns=['flag_go', 'flag_gva', 'flag_exgr', 'flag_imgr', 'year', 'own'],
            inplace=True
        )

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
        if self.oecd is None:
            raise Exception(
                "Before you may use this method, you have to load the OECD's CbCR data with the dedicated method."
            )

        aamne_domestic = self.get_unextended_domestic_analytical_amne_data()

        geographies = pd.read_csv(self.path_to_geographies)

        aamne_domestic = aamne_domestic.merge(
            geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='COUNTRY_CODE', right_on='CODE'
        )

        aamne_domestic.drop(columns=['CODE'], inplace=True)

        aamne_domestic['CONTINENT_CODE'] = aamne_domestic['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        aamne_domestic['CONTINENT_CODE'] = aamne_domestic['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        continent_imputations = {}

        columns_of_interest = [
            'DOMESTIC_SALES', 'SALES_TO_OTHER_COUNTRY'
        ]

        for continent in aamne_domestic['CONTINENT_CODE'].unique():
            continent_imputations[continent] = {}

            restricted_df = aamne_domestic[aamne_domestic['CONTINENT_CODE'] == continent].copy()

            denominator = restricted_df[columns_of_interest].sum().sum()

            for column in columns_of_interest:
                suffix = column.replace('SALES_', '')
                new_column = 'PERC_' + suffix

                numerator = restricted_df[column].sum()

                continent_imputations[continent][new_column] = numerator / denominator

        aamne_domestic['TOTAL_SALES'] = aamne_domestic[columns_of_interest].sum(axis=1)

        new_columns = []

        for column in columns_of_interest:
            suffix = column.replace('SALES_', '')
            new_column = 'PERC_' + suffix
            new_columns.append(new_column)

            aamne_domestic[new_column] = aamne_domestic[column] / aamne_domestic['TOTAL_SALES']

        aamne_domestic.drop(
            columns=columns_of_interest + ['TOTAL_SALES', 'CONTINENT_CODE'],
            inplace=True
        )

        parent_jurisdictions = self.oecd[
            self.oecd['PARENT_COUNTRY_CODE'] == self.oecd['AFFILIATE_COUNTRY_CODE']
        ][['PARENT_COUNTRY_CODE', 'CONTINENT_CODE']].drop_duplicates()

        parent_jurisdictions = parent_jurisdictions.merge(
            aamne_domestic,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

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
