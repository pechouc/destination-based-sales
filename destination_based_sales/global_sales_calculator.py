"""
Following the model of "analyses.py", this module is the central module for the study of non-US multinational companies'
destination-based sales. It builds upon the logic encapsulated in "oecd_cbcr.py", "analytical_amne.py" and "trade_
statistics.py" to output the destination-based mapping of their worldwide revenues; this is covered in the "GlobalSales-
Calculator" Python class. Additionally, the "GlobalAnalysisProvider" class allows to reproduce the analyses that can be
found in the PDF report. In particular, it allows to re-estimate the revenue gains from the unilateral scenario of
Baraké et al. (2021).
"""


########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

from destination_based_sales.analytical_amne import AnalyticalAMNEPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor
from destination_based_sales.trade_statistics import TradeStatisticsProcessor
from destination_based_sales.analyses import SalesCalculator


########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_tax_deficits = os.path.join(path_to_dir, 'data', 'total_tax_deficits.xlsx')
path_to_EU_countries = os.path.join(path_to_dir, 'data', 'listofeucountries_csv.csv')

eu_28_country_codes = pd.read_csv(path_to_EU_countries, delimiter=';')
eu_28_country_codes = list(eu_28_country_codes['Alpha-3 code'].unique())
eu_27_country_codes = [c for c in eu_28_country_codes if c not in ['GBR', 'BGR', 'HRV', 'ROU', 'LTU']]


########################################################################################################################
# --- Content

class GlobalSalesCalculator:

    def __init__(self, winsorize_export_percs=True):

        self.aamne_preprocessor = AnalyticalAMNEPreprocessor()

        self.oecd_preprocessor = CbCRPreprocessor()
        self.oecd = self.oecd_preprocessor.get_preprocessed_revenue_data()

        self.trade_stat_processor = TradeStatisticsProcessor(
            year=2016,
            winsorize_export_percs=winsorize_export_percs,
            US_only=False
        )

        self.US_sales_calculator = SalesCalculator(year=2016)

        self.basic_revenue_columns = ['RELATED_PARTY_REVENUES', 'TOTAL_REVENUES', 'UNRELATED_PARTY_REVENUES'].copy()

        self.geo_columns = [
            'PARENT_COUNTRY_CODE', 'PARENT_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME'
        ].copy()

    def get_foreign_sales_split(self):

        foreign_aamne_data = self.aamne_preprocessor.get_extended_foreign_analytical_amne_data()

        oecd_foreign = self.oecd[
            self.oecd['PARENT_COUNTRY_CODE'] != self.oecd['AFFILIATE_COUNTRY_CODE']
        ].copy()

        merged_df_foreign = oecd_foreign.merge(
            foreign_aamne_data,
            how='left',
            on='AFFILIATE_COUNTRY_CODE'
        )

        for revenue_column in self.basic_revenue_columns:
            for perc_column in ['PERC_TO_AFFILIATE_COUNTRY', 'PERC_TO_HEADQUARTER_COUNTRY', 'PERC_TO_OTHER_COUNTRY']:

                suffix = perc_column.replace('PERC', '')

                merged_df_foreign[revenue_column + suffix] = \
                    merged_df_foreign[revenue_column] * merged_df_foreign[perc_column]

        merged_df_foreign.drop(
            columns=[
                'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES',
                'UNRELATED_PARTY_REVENUES', 'CONTINENT_CODE',
                'PERC_TO_AFFILIATE_COUNTRY', 'PERC_TO_HEADQUARTER_COUNTRY',
                'PERC_TO_OTHER_COUNTRY'
            ],
            inplace=True
        )

        return merged_df_foreign.copy()

    def get_domestic_sales_split(self):

        domestic_aamne_data = self.aamne_preprocessor.get_extended_domestic_analytical_amne_data()

        oecd_domestic = self.oecd[
            self.oecd['PARENT_COUNTRY_CODE'] == self.oecd['AFFILIATE_COUNTRY_CODE']
        ].copy()

        merged_df_domestic = oecd_domestic.merge(
            domestic_aamne_data,
            how='left',
            on='PARENT_COUNTRY_CODE'
        )

        for revenue_column in self.basic_revenue_columns:
            merged_df_domestic[revenue_column + '_TO_HEADQUARTER_COUNTRY'] = \
                merged_df_domestic['PERC_DOMESTIC_SALES'] * merged_df_domestic[revenue_column]

            merged_df_domestic[revenue_column + '_TO_AFFILIATE_COUNTRY'] = 0

            merged_df_domestic[revenue_column + '_TO_OTHER_COUNTRY'] = \
                merged_df_domestic['PERC_TO_OTHER_COUNTRY'] * merged_df_domestic[revenue_column]

        merged_df_domestic.drop(
            columns=[
                'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES',
                'UNRELATED_PARTY_REVENUES', 'CONTINENT_CODE',
                'PERC_DOMESTIC_SALES', 'PERC_TO_OTHER_COUNTRY'
            ],
            inplace=True
        )

        return merged_df_domestic.copy()

    def get_complete_revenue_split(self):

        merged_df_foreign = self.get_foreign_sales_split()
        merged_df_domestic = self.get_domestic_sales_split()

        complete_df = pd.concat([merged_df_domestic, merged_df_foreign], axis=0)

        headquarter_country_columns = []
        affiliate_country_columns = []
        other_country_columns = []

        for column in complete_df.columns:
            if column in self.geo_columns:
                continue

            elif 'HEADQUARTER' in column:
                headquarter_country_columns.append(column)

            elif 'AFFILIATE' in column:
                affiliate_country_columns.append(column)

            else:
                other_country_columns.append(column)

        headquarter_df = complete_df[self.geo_columns + headquarter_country_columns].copy()
        affiliate_df = complete_df[self.geo_columns + affiliate_country_columns].copy()
        other_country_df = complete_df[self.geo_columns + other_country_columns].copy()

        return headquarter_df, affiliate_df, other_country_df

    def get_already_attributed_sales(self):

        headquarter_df, affiliate_df, _ = self.get_complete_revenue_split()

        headquarter_df['ULTIMATE_DESTINATION_CODE'] = headquarter_df['PARENT_COUNTRY_CODE'].values
        headquarter_df['ULTIMATE_DESTINATION_NAME'] = headquarter_df['PARENT_COUNTRY_NAME'].values

        affiliate_df['ULTIMATE_DESTINATION_CODE'] = affiliate_df['AFFILIATE_COUNTRY_CODE'].values
        affiliate_df['ULTIMATE_DESTINATION_NAME'] = affiliate_df['AFFILIATE_COUNTRY_NAME'].values

        headquarter_df.rename(
            columns={
                'RELATED_PARTY_REVENUES_TO_HEADQUARTER_COUNTRY': 'RELATED_PARTY_REVENUES',
                'UNRELATED_PARTY_REVENUES_TO_HEADQUARTER_COUNTRY': 'UNRELATED_PARTY_REVENUES',
                'TOTAL_REVENUES_TO_HEADQUARTER_COUNTRY': 'TOTAL_REVENUES'
            },
            inplace=True
        )

        affiliate_df.rename(
            columns={
                'RELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY': 'RELATED_PARTY_REVENUES',
                'UNRELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY': 'UNRELATED_PARTY_REVENUES',
                'TOTAL_REVENUES_TO_AFFILIATE_COUNTRY': 'TOTAL_REVENUES'
            },
            inplace=True
        )

        already_attributed = pd.concat([headquarter_df, affiliate_df], axis=0)

        return already_attributed.copy()

    def attribute_other_country_sales(self):

        _, _, to_be_attributed = self.get_complete_revenue_split()

        df = self.trade_stat_processor.load_data_with_imputations()

        merged_df = to_be_attributed.merge(
            df,
            how='left',
            on='AFFILIATE_COUNTRY_CODE'
        )

        totals = {}
        merged_df['KEY'] = merged_df['PARENT_COUNTRY_CODE'] + merged_df['AFFILIATE_COUNTRY_CODE']

        for key in merged_df['KEY'].unique():
            restricted_df = merged_df[merged_df['KEY'] == key].copy()

            totals[key] = restricted_df[
                restricted_df['OTHER_COUNTRY_CODE'] != key[:3]
            ]['ALL_EXPORTS'].sum()

        merged_df['EXPORT_PERC'] = merged_df.apply(
            lambda row: row['ALL_EXPORTS'] / totals[
                row['PARENT_COUNTRY_CODE'] + row['AFFILIATE_COUNTRY_CODE']
            ],
            axis=1
        )

        merged_df = merged_df[merged_df['PARENT_COUNTRY_CODE'] != merged_df['OTHER_COUNTRY_CODE']].copy()

        columns_to_drop = []

        for revenue_column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']:
            column_to_drop = revenue_column + '_TO_OTHER_COUNTRY'

            columns_to_drop.append(column_to_drop)

            merged_df[revenue_column] = \
                merged_df[column_to_drop] * merged_df['EXPORT_PERC']

        columns_to_drop += ['ALL_EXPORTS', 'EXPORT_PERC', 'KEY']

        merged_df.drop(columns=columns_to_drop, inplace=True)

        merged_df.rename(
            columns={
                'OTHER_COUNTRY_CODE': 'ULTIMATE_DESTINATION_CODE'
            },
            inplace=True
        )

        return merged_df.copy()

    def get_simplified_sales_mapping_without_US(self):

        already_attributed = self.get_already_attributed_sales()

        already_attributed.drop(
            columns=[
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME',
                'PARENT_COUNTRY_NAME', 'ULTIMATE_DESTINATION_NAME'
            ],
            inplace=True
        )

        merged_df = self.attribute_other_country_sales()

        merged_df.drop(
            columns=['AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME', 'PARENT_COUNTRY_NAME'],
            inplace=True
        )

        merged_df = pd.concat(
            [already_attributed, merged_df],
            axis=0
        )

        merged_df = merged_df.groupby(
            ['PARENT_COUNTRY_CODE', 'ULTIMATE_DESTINATION_CODE']
        ).sum().reset_index()

        return merged_df[merged_df['PARENT_COUNTRY_CODE'] != 'USA'].copy()

    def get_simplified_sales_mapping(self):

        sales_mapping_excl_US = self.get_simplified_sales_mapping_without_US()

        US_sales_mapping = self.US_sales_calculator.get_final_dataframe()

        US_sales_mapping.drop(
            columns=['AFFILIATE_COUNTRY_CODE', 'EXPORT_PERC', 'AFFILIATE_COUNTRY_NAME'],
            inplace=True
        )

        US_sales_mapping = US_sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        US_sales_mapping['PARENT_COUNTRY_CODE'] = 'USA'

        US_sales_mapping.rename(
            columns={
                'OTHER_COUNTRY_CODE': 'ULTIMATE_DESTINATION_CODE'
            },
            inplace=True
        )

        sales_mapping = pd.concat([sales_mapping_excl_US, US_sales_mapping], axis=0)

        return sales_mapping.copy()


class GlobalAnalysisProvider:

    def __init__(self, path_to_tax_deficits=path_to_tax_deficits):
        self.global_sales_calculator = GlobalSalesCalculator()
        self.sales_mapping = self.global_sales_calculator.get_simplified_sales_mapping()

        self.path_to_tax_deficits = path_to_tax_deficits

    def compute_unilateral_scenario_gains(
        self,
        taxing_country_code,
        minimum_ETR=0.25,
        return_split=False
    ):
        if minimum_ETR not in [0.15, 0.21, 0.25, 0.3]:
            raise Exception('Only the 4 benchmark minimum ETRs of the June 1st report are accepted.')

        sales_mapping = self.sales_mapping.copy()

        tax_deficits = pd.read_excel(
            self.path_to_tax_deficits,
            sheet_name=f'{int(minimum_ETR * 100)}_percent',
            engine='openpyxl'
        )

        tax_deficits = tax_deficits[tax_deficits['Parent jurisdiction (alpha-3 code)'] != '..'].copy()

        attribution_ratios = []

        for country_code in tax_deficits['Parent jurisdiction (alpha-3 code)'].unique():
            restricted_sales_mapping = sales_mapping[
                sales_mapping['PARENT_COUNTRY_CODE'] == country_code
            ].copy()

            denominator = restricted_sales_mapping['UNRELATED_PARTY_REVENUES'].sum()

            if taxing_country_code == country_code:
                attribution_ratios.append(1)

            elif taxing_country_code in restricted_sales_mapping['ULTIMATE_DESTINATION_CODE'].unique():
                numerator = restricted_sales_mapping[
                    restricted_sales_mapping['ULTIMATE_DESTINATION_CODE'] == taxing_country_code
                ]['UNRELATED_PARTY_REVENUES'].iloc[0]

                attribution_ratios.append(numerator / denominator)

            else:
                attribution_ratios.append(0)

        tax_deficits['ATTRIBUTION_RATIOS'] = attribution_ratios
        tax_deficits['COLLECTIBLE_TAX_DEFICIT'] = tax_deficits['tax_deficit'] * tax_deficits['ATTRIBUTION_RATIOS']

        imputation = tax_deficits[
            ~tax_deficits['Parent jurisdiction (alpha-3 code)'].isin([taxing_country_code, 'USA'])
        ]['COLLECTIBLE_TAX_DEFICIT'].sum()

        imputation_adjusted = imputation

        if taxing_country_code == 'DEU':
            imputation_adjusted /= 2

        if not return_split:
            return tax_deficits['COLLECTIBLE_TAX_DEFICIT'].sum() + imputation_adjusted

        else:
            own_tax_deficit = tax_deficits[
                tax_deficits['Parent jurisdiction (alpha-3 code)'] == taxing_country_code
            ]['COLLECTIBLE_TAX_DEFICIT'].iloc[0]

            if taxing_country_code == 'USA':
                collected_from_the_US = 0

            else:
                collected_from_the_US = tax_deficits[
                    tax_deficits['Parent jurisdiction (alpha-3 code)'] == 'USA'
                ]['COLLECTIBLE_TAX_DEFICIT'].iloc[0]

            collected_from_other_multinationals = imputation + imputation_adjusted

            total = own_tax_deficit + collected_from_the_US + collected_from_other_multinationals

            return own_tax_deficit, collected_from_the_US, collected_from_other_multinationals, total

    def build_new_table_3(self, output_excel=False):

        output = {
            'OWN_TAX_DEFICIT': [],
            'COLLECTIBLE_FROM_US_MNEs': [],
            'COLLECTIBLE_FROM_OTHER_MNEs': [],
            'TOTAL': []
        }

        oecd = self.global_sales_calculator.oecd.copy()

        oecd_reporting_countries = list(oecd['PARENT_COUNTRY_CODE'].unique())
        oecd_reporting_countries += ['KOR', 'NLD', 'IRL', 'FIN']

        countries_to_display = sorted(eu_27_country_codes).copy()

        for country_code in sorted(oecd_reporting_countries):
            if country_code not in countries_to_display:
                countries_to_display.append(country_code)

            else:
                continue

        output['COUNTRY_CODE'] = countries_to_display.copy()

        for country_code in countries_to_display:

            own_tax_deficit, collected_from_the_US, collected_from_other_multinationals, total = \
                self.compute_unilateral_scenario_gains(country_code, return_split=True)

            output['OWN_TAX_DEFICIT'].append(own_tax_deficit)
            output['COLLECTIBLE_FROM_US_MNEs'].append(collected_from_the_US)
            output['COLLECTIBLE_FROM_OTHER_MNEs'].append(collected_from_other_multinationals)
            output['TOTAL'].append(total)

        output_df = pd.DataFrame.from_dict(output)

        if output_excel:
            with pd.ExcelWriter('/Users/Paul-Emmanuel/Desktop/new_table_3.xlsx', engine='xlsxwriter') as writer:
                output_df.to_excel(writer, sheet_name='revised_table', index=False)

        return output_df.copy()

    def build_new_table_3_formatted(self):

        output_df = self.build_new_table_3()

        columns = ['COUNTRY_CODE'] + list(output_df.columns[:-1])

        output_df = output_df[columns].copy()

        for column in output_df.columns[1:]:
            output_df[column] /= 10**9

            output_df[column] = output_df[column].map('{:.1f}'.format)

        return output_df.copy()
