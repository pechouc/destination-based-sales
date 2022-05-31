
########################################################################################################################
# --- Imports

import numpy as np
import pandas as pd

from destination_based_sales.sales_calculator import SimplifiedGlobalSalesCalculator
from destination_based_sales.utils import UK_CARIBBEAN_ISLANDS

from tax_deficit_simulator.calculator import TaxDeficitCalculator


########################################################################################################################
# --- Diverse

path = 'https://raw.githubusercontent.com/eutaxobservatory/tax-deficit-simulator/'
path += 'master/tax_deficit_simulator/data/listofeucountries_csv.csv'

eu_27_country_codes = pd.read_csv(
    path,
    delimiter=';'
)

eu_27_country_codes = list(eu_27_country_codes['Alpha-3 code'].unique())
eu_27_country_codes.remove('GBR')


########################################################################################################################
# --- Main code

class TaxReformSimulator:

    def __init__(self, year):

        self.year = year

        # Storing the global sales calculator and the unadjusted / adjusted sales mappings
        self.sales_calculator = SimplifiedGlobalSalesCalculator(
            year=self.year,
            aamne_domestic_sales_perc=False,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5,
            service_flows_to_exclude=[]
        )

        self.oecd_sales_mapping = self.sales_calculator.oecd.copy()

        self.adjusted_sales_mapping = self.sales_calculator.get_final_sales_mapping()

        # Storing the tax deficits to redistribute
        self.tax_deficit_calculator = TaxDeficitCalculator(
            year=self.year,
            add_AUT_AUT_row=True,
            sweden_treatment='adjust',
            belgium_treatment='replace',
            use_adjusted_profits=True,
            average_ETRs=True,
            carve_outs=False,
            de_minimis_exclusion=True,
            fetch_data_online=True
        )

        self.tax_deficit_calculator.load_clean_data()

        self.tax_deficits = self.tax_deficit_calculator.get_total_tax_deficits(minimum_ETR=0.15)

    def compute_Barake_et_al_unilateral_scenarios(self, elim_negative_revenues=True):

        if self.year == 2016:

            imputation_multiplier = 1
            correction_for_DEU = 2

        else:

            imputation_multiplier = 0.25
            correction_for_DEU = 1

        oecd_sales_mapping = self.oecd_sales_mapping.copy()
        adjusted_sales_mapping = self.adjusted_sales_mapping.copy()

        tax_deficits = self.tax_deficits.copy()

        tax_deficits = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'] != '..'
        ].copy()

        # ### Grouping the tax deficits of UK Caribbean Islands ---------------------------------------------------- ###
        UKI_extract = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(UK_CARIBBEAN_ISLANDS)
        ].copy()

        ser = UKI_extract.sum()

        ser.loc['Parent jurisdiction (whitespaces cleaned)'] = 'United Kingdom Islands, Caribbean'
        ser.loc['Parent jurisdiction (alpha-3 code)'] = 'UKI'

        UKI_extract = pd.DataFrame(ser).T.copy()

        restr_tax_deficits = tax_deficits[
            ~tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(UK_CARIBBEAN_ISLANDS)
        ].copy()

        restr_tax_deficits = pd.concat([restr_tax_deficits, UKI_extract], axis=0)
        restr_tax_deficits['tax_deficit'] = restr_tax_deficits['tax_deficit'].astype(float)

        # ### Preparing the unadjusted sales mapping --------------------------------------------------------------- ###
        restr_oecd_mapping = oecd_sales_mapping[
            [
                'PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES'
            ]
        ].copy()

        if elim_negative_revenues:
            restr_oecd_mapping['UNRELATED_PARTY_REVENUES'] = restr_oecd_mapping['UNRELATED_PARTY_REVENUES'].map(
                lambda x: max(x, 0)
            )

        parent_totals = restr_oecd_mapping.groupby('PARENT_COUNTRY_CODE').sum().to_dict()['UNRELATED_PARTY_REVENUES']

        restr_oecd_mapping['URP_PERCENTAGE'] = (
            restr_oecd_mapping['UNRELATED_PARTY_REVENUES']
            / restr_oecd_mapping['PARENT_COUNTRY_CODE'].map(parent_totals)
        )

        # We restrict to the affiliate countries for which we have computed a tax deficit already
        # Otherwise, the resulting estimated revenue gains will not be comparable
        restr_oecd_mapping = restr_oecd_mapping[
            restr_oecd_mapping['AFFILIATE_COUNTRY_CODE'].isin(
                restr_tax_deficits['Parent jurisdiction (alpha-3 code)'].unique()
            )
        ].copy()

        # Cases where the parent country and the affiliate country are the same are dealt with when taking the
        # pre-computed tax deficits
        restr_oecd_mapping = restr_oecd_mapping[
            restr_oecd_mapping['PARENT_COUNTRY_CODE'] != restr_oecd_mapping['AFFILIATE_COUNTRY_CODE']
        ].copy()

        restr_oecd_mapping = restr_oecd_mapping.drop(columns=['UNRELATED_PARTY_REVENUES'])

        # ### Preparing the adjusted sales mapping ----------------------------------------------------------------- ###

        restr_adjusted_mapping = adjusted_sales_mapping.groupby(
            ['PARENT_COUNTRY_CODE', 'OTHER_COUNTRY_CODE']
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        if elim_negative_revenues:
            restr_adjusted_mapping['UNRELATED_PARTY_REVENUES'] = restr_adjusted_mapping['UNRELATED_PARTY_REVENUES'].map(
                lambda x: max(x, 0)
            )

        parent_totals = restr_adjusted_mapping.groupby(
            'PARENT_COUNTRY_CODE'
        ).sum().to_dict()['UNRELATED_PARTY_REVENUES']

        restr_adjusted_mapping['URP_PERCENTAGE'] = (
            restr_adjusted_mapping['UNRELATED_PARTY_REVENUES']
            / restr_adjusted_mapping['PARENT_COUNTRY_CODE'].map(parent_totals)
        )

        # We restrict to the affiliate countries for which we have computed a tax deficit already
        # Otherwise, the resulting estimated revenue gains will not be comparable
        restr_adjusted_mapping = restr_adjusted_mapping[
            restr_adjusted_mapping['OTHER_COUNTRY_CODE'].isin(
                restr_tax_deficits['Parent jurisdiction (alpha-3 code)'].unique()
            )
        ].copy()

        # Cases where the parent country and the affiliate country are the same are dealt with when taking the
        # pre-computed tax deficits
        restr_adjusted_mapping = restr_adjusted_mapping[
            restr_adjusted_mapping['PARENT_COUNTRY_CODE'] != restr_adjusted_mapping['OTHER_COUNTRY_CODE']
        ].copy()

        restr_adjusted_mapping = restr_adjusted_mapping.drop(columns=['UNRELATED_PARTY_REVENUES'])

        # ### Deducing revenue gain estimates ---------------------------------------------------------------------- ###

        merged_df = restr_oecd_mapping.merge(
            restr_tax_deficits,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='Parent jurisdiction (alpha-3 code)'
        )
        merged_df = merged_df.drop(
            columns=[
                'Parent jurisdiction (alpha-3 code)', 'Parent jurisdiction (whitespaces cleaned)'
            ]
        )

        self.merged_df_unadj = merged_df.copy()

        merged_df['ATTRIBUTABLE_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']

        # Applying the imputation
        merged_df['MULTIPLIER'] = (merged_df['PARENT_COUNTRY_CODE'] != 'USA') * imputation_multiplier
        merged_df['MULTIPLIER'] = merged_df.apply(
            (
                lambda row: row['MULTIPLIER'] / correction_for_DEU
                if row['AFFILIATE_COUNTRY_CODE'] == 'DEU' else row['MULTIPLIER']
            ),
            axis=1
        )
        merged_df['ATTRIBUTABLE_REVENUES'] *= (1 + merged_df['MULTIPLIER'])

        temp_unadjusted = merged_df.groupby('AFFILIATE_COUNTRY_CODE').sum()['ATTRIBUTABLE_REVENUES'].to_dict()

        merged_df = restr_adjusted_mapping.merge(
            restr_tax_deficits,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='Parent jurisdiction (alpha-3 code)'
        )
        merged_df = merged_df.drop(
            columns=[
                'Parent jurisdiction (alpha-3 code)', 'Parent jurisdiction (whitespaces cleaned)'
            ]
        )

        self.merged_df_adj = merged_df.copy()

        merged_df['ATTRIBUTABLE_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']

        # Applying the imputation
        merged_df['MULTIPLIER'] = (merged_df['PARENT_COUNTRY_CODE'] != 'USA') * imputation_multiplier
        merged_df['MULTIPLIER'] = merged_df.apply(
            (
                lambda row: row['MULTIPLIER'] / correction_for_DEU
                if row['OTHER_COUNTRY_CODE'] == 'DEU' else row['MULTIPLIER']
            ),
            axis=1
        )
        merged_df['ATTRIBUTABLE_REVENUES'] *= (1 + merged_df['MULTIPLIER'])

        temp_adjusted = merged_df.groupby('OTHER_COUNTRY_CODE').sum()['ATTRIBUTABLE_REVENUES'].to_dict()

        # ### Gathering all types of revenue gains ----------------------------------------------------------------- ###

        restr_tax_deficits['UNADJ_ADDITIONAL_REVENUES'] = restr_tax_deficits['Parent jurisdiction (alpha-3 code)'].map(
            temp_unadjusted
        )
        restr_tax_deficits['ADJ_ADDITIONAL_REVENUES'] = restr_tax_deficits['Parent jurisdiction (alpha-3 code)'].map(
            temp_adjusted
        )

        restr_tax_deficits['UNADJ_TOTAL_TAX_DEFICITS'] = (
            restr_tax_deficits['tax_deficit'] + restr_tax_deficits['UNADJ_ADDITIONAL_REVENUES']
        )
        restr_tax_deficits['ADJ_TOTAL_TAX_DEFICITS'] = (
            restr_tax_deficits['tax_deficit'] + restr_tax_deficits['ADJ_ADDITIONAL_REVENUES']
        )

        return restr_tax_deficits.copy()

    def get_formatted_unilateral_scenario_comparison(self, table_type):
        output_df = self.compute_Barake_et_al_unilateral_scenarios()

        # Identifying EU countries and non-EU OECD-reporting countries
        output_df['Category'] = output_df['Parent jurisdiction (alpha-3 code)'].isin(eu_27_country_codes) * 1

        oecd_reporting_countries = list(self.oecd_sales_mapping['PARENT_COUNTRY_CODE'].unique())
        output_df['Category'] += np.logical_and(
            ~output_df['Parent jurisdiction (alpha-3 code)'].isin(eu_27_country_codes),
            output_df['Parent jurisdiction (alpha-3 code)'].isin(oecd_reporting_countries)
        ) * 2

        # ### Adding sub-totals

        # Full sample total
        ser = output_df.sum()

        ser.loc['Parent jurisdiction (whitespaces cleaned)'] = 'Full sample'
        ser.loc['Parent jurisdiction (alpha-3 code)'] = '..'
        ser.loc['Category'] = 3

        output_df = pd.concat(
            [
                output_df,
                pd.DataFrame(ser).T.copy()
            ]
        )

        # EU-wide total
        ser = output_df[output_df['Category'] == 1].sum()

        ser.loc['Parent jurisdiction (whitespaces cleaned)'] = 'EU Member-States'
        ser.loc['Parent jurisdiction (alpha-3 code)'] = '..'
        ser.loc['Category'] = 1.5

        output_df = pd.concat(
            [
                output_df,
                pd.DataFrame(ser).T.copy()
            ]
        )

        # Total for OECD-reporting countries
        ser = output_df[output_df['Parent jurisdiction (alpha-3 code)'].isin(oecd_reporting_countries)].sum()

        ser.loc['Parent jurisdiction (whitespaces cleaned)'] = 'OECD'
        ser.loc['Parent jurisdiction (alpha-3 code)'] = '..'
        ser.loc['Category'] = 2.5

        output_df = pd.concat(
            [
                output_df,
                pd.DataFrame(ser).T.copy()
            ]
        )

        # ### Finalising the formatting

        # Restricting the output to relevant countries and ordering based on the category
        output_df = output_df[output_df['Category'] > 0].copy()
        output_df = output_df.sort_values(by=['Category', 'Parent jurisdiction (whitespaces cleaned)'])
        output_df = output_df.reset_index(drop=True)

        if table_type == 'total_amounts':
            output_df = output_df[
                [
                    'Parent jurisdiction (whitespaces cleaned)', 'tax_deficit',
                    'UNADJ_TOTAL_TAX_DEFICITS', 'ADJ_TOTAL_TAX_DEFICITS'
                ]
            ].copy()

            for col in ['tax_deficit', 'UNADJ_TOTAL_TAX_DEFICITS', 'ADJ_TOTAL_TAX_DEFICITS']:
                output_df[col] /= 10**9

            output_df['Effect of the adjustment (%)'] = (
                (output_df['ADJ_TOTAL_TAX_DEFICITS'] - output_df['UNADJ_TOTAL_TAX_DEFICITS'])
                / output_df['UNADJ_TOTAL_TAX_DEFICITS']
            ) * 100

            output_df = output_df.rename(
                columns={
                    'Parent jurisdiction (whitespaces cleaned)': 'Taxing country',
                    'tax_deficit': 'Own tax deficit (€bn)',
                    'UNADJ_TOTAL_TAX_DEFICITS': 'Based on unadjusted sales (billion EUR)',
                    'ADJ_TOTAL_TAX_DEFICITS': 'Based on adjusted sales (billion EUR)'
                }
            )

            output_df['Taxing country'] = output_df['Taxing country'].map(
                lambda x: 'China' if x == "China (People's Republic of)" else x
            )

            for col in output_df.columns[1:]:
                output_df[col] = output_df[col].astype(float)

            return output_df.copy()

        elif table_type == 'focus_on_foreign_collection':
            output_df = output_df[
                [
                    'Parent jurisdiction (whitespaces cleaned)',
                    'UNADJ_ADDITIONAL_REVENUES', 'ADJ_ADDITIONAL_REVENUES'
                ]
            ].copy()

            for col in ['UNADJ_ADDITIONAL_REVENUES', 'ADJ_ADDITIONAL_REVENUES']:
                output_df[col] /= 10**6

            output_df['Effect of the adjustment (%)'] = (
                (output_df['ADJ_ADDITIONAL_REVENUES'] - output_df['UNADJ_ADDITIONAL_REVENUES'])
                / output_df['UNADJ_ADDITIONAL_REVENUES']
            ) * 100

            output_df = output_df.rename(
                columns={
                    'Parent jurisdiction (whitespaces cleaned)': 'Taxing country',
                    'UNADJ_ADDITIONAL_REVENUES': 'Based on unadjusted sales (million EUR)',
                    'ADJ_ADDITIONAL_REVENUES': 'Based on adjusted sales (million EUR)'
                }
            )

            output_df['Taxing country'] = output_df['Taxing country'].map(
                lambda x: 'China' if x == "China (People's Republic of)" else x
            )

            for col in output_df.columns[1:]:
                output_df[col] = output_df[col].astype(float)

            return output_df

        else:
            raise Exception(
                'Only two types of tables are supported with this method: '
                + '"total_amounts" and "focus_on_foreign_collection".'
            )

    def full_sales_apportionment(self, elim_negative_revenues=True):

        oecd_sales_mapping = self.oecd_sales_mapping.copy()
        adjusted_sales_mapping = self.adjusted_sales_mapping.copy()

        tax_deficits = self.tax_deficits.copy()

        tax_deficits = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'] != '..'
        ].copy()

        # ### Grouping the tax deficits of UK Caribbean Islands ---------------------------------------------------- ###

        UKI_extract = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(UK_CARIBBEAN_ISLANDS)
        ].copy()

        ser = UKI_extract.sum()

        ser.loc['Parent jurisdiction (whitespaces cleaned)'] = 'United Kingdom Islands, Caribbean'
        ser.loc['Parent jurisdiction (alpha-3 code)'] = 'UKI'

        UKI_extract = pd.DataFrame(ser).T.copy()

        restr_tax_deficits = tax_deficits[
            ~tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(UK_CARIBBEAN_ISLANDS)
        ].copy()

        restr_tax_deficits = pd.concat([restr_tax_deficits, UKI_extract], axis=0)
        restr_tax_deficits['tax_deficit'] = restr_tax_deficits['tax_deficit'].astype(float)

        # ### Preparing the unadjusted sales mapping --------------------------------------------------------------- ###

        restr_oecd_mapping = oecd_sales_mapping[
            [
                'PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES'
            ]
        ].copy()

        if elim_negative_revenues:
            restr_oecd_mapping['UNRELATED_PARTY_REVENUES'] = restr_oecd_mapping['UNRELATED_PARTY_REVENUES'].map(
                lambda x: max(x, 0)
            )

        parent_totals = restr_oecd_mapping.groupby('PARENT_COUNTRY_CODE').sum().to_dict()['UNRELATED_PARTY_REVENUES']

        restr_oecd_mapping['URP_PERCENTAGE'] = (
            restr_oecd_mapping['UNRELATED_PARTY_REVENUES']
            / restr_oecd_mapping['PARENT_COUNTRY_CODE'].map(parent_totals)
        )

        restr_oecd_mapping = restr_oecd_mapping.drop(columns=['UNRELATED_PARTY_REVENUES'])

        # ### Preparing the adjusted sales mapping ----------------------------------------------------------------- ###

        restr_adjusted_mapping = adjusted_sales_mapping.groupby(
            ['PARENT_COUNTRY_CODE', 'OTHER_COUNTRY_CODE']
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        if elim_negative_revenues:
            restr_adjusted_mapping['UNRELATED_PARTY_REVENUES'] = restr_adjusted_mapping['UNRELATED_PARTY_REVENUES'].map(
                lambda x: max(x, 0)
            )

        parent_totals = restr_adjusted_mapping.groupby(
            'PARENT_COUNTRY_CODE'
        ).sum().to_dict()['UNRELATED_PARTY_REVENUES']
        restr_adjusted_mapping['URP_PERCENTAGE'] = (
            restr_adjusted_mapping['UNRELATED_PARTY_REVENUES']
            / restr_adjusted_mapping['PARENT_COUNTRY_CODE'].map(parent_totals)
        )

        restr_adjusted_mapping = restr_adjusted_mapping.drop(columns=['UNRELATED_PARTY_REVENUES'])

        # ### Deducing the unadjusted distribution of revenues ----------------------------------------------------- ###

        merged_df = restr_oecd_mapping.merge(
            restr_tax_deficits,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='Parent jurisdiction (alpha-3 code)'
        ).drop(
            columns=[
                'Parent jurisdiction (alpha-3 code)',
                'Parent jurisdiction (whitespaces cleaned)'
            ]
        )

        merged_df['ATTRIBUTED_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        # unadjusted_temp = merged_df.groupby('AFFILIATE_COUNTRY_CODE').sum()['ATTRIBUTED_REVENUES'].to_dict()

        # ### Deducing the adjusted distribution of revenues ------------------------------------------------------- ###

        merged_df = restr_adjusted_mapping.merge(
            restr_tax_deficits,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='Parent jurisdiction (alpha-3 code)'
        ).drop(
            columns=[
                'Parent jurisdiction (alpha-3 code)',
                'Parent jurisdiction (whitespaces cleaned)'
            ]
        )

        merged_df['ATTRIBUTED_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        # adjusted_temp = merged_df.groupby('OTHER_COUNTRY_CODE').sum()['ATTRIBUTED_REVENUES']

        return restr_tax_deficits.copy()
