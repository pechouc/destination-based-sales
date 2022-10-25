
########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

from destination_based_sales.sales_calculator import SimplifiedGlobalSalesCalculator
from destination_based_sales.utils import UK_CARIBBEAN_ISLANDS, define_category, online_path_to_geo_file

from tax_deficit_simulator.calculator import TaxDeficitCalculator


########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


########################################################################################################################
# --- Main code

class TaxReformSimulator:

    def __init__(self, year, load_data_online=False):

        self.year = year

        self.path_to_geographies = path_to_geographies if not load_data_online else online_path_to_geo_file

        # Storing the global sales calculator and the unadjusted / adjusted sales mappings without any filtering based
        # on the detail of the bilateral breakdown provided in CbCR statistics
        self.sales_calculator = SimplifiedGlobalSalesCalculator(
            year=self.year,
            aamne_domestic_sales_perc=False,
            breakdown_threshold=0,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5,
            service_flows_to_exclude=[],
            load_data_online=load_data_online
        )

        self.oecd_sales_mapping = self.sales_calculator.oecd.copy()
        self.adjusted_sales_mapping = self.sales_calculator.get_final_sales_mapping()

        # Storing the global sales calculator and the unadjusted / adjusted sales mappings with the benchmark filtering
        # based on the detail of the bilateral breakdown provided in CbCR statistics
        self.restr_sales_calculator = SimplifiedGlobalSalesCalculator(
            year=self.year,
            aamne_domestic_sales_perc=False,
            breakdown_threshold=60,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5,
            service_flows_to_exclude=[],
            load_data_online=load_data_online
        )

        self.restr_oecd_sales_mapping = self.restr_sales_calculator.oecd.copy()
        self.restr_adjusted_sales_mapping = self.restr_sales_calculator.get_final_sales_mapping()

        # Storing the tax deficits to redistribute
        self.tax_deficit_calculator = TaxDeficitCalculator(
            year=self.year,
            add_AUT_AUT_row=True,
            sweden_treatment='adjust',
            belgium_treatment='replace',
            SGP_CYM_treatment='replace',
            use_adjusted_profits=True,
            average_ETRs=True,
            carve_outs=False,
            de_minimis_exclusion=True,
            fetch_data_online=True
        )

        self.tax_deficit_calculator.load_clean_data()

        tax_deficits = self.tax_deficit_calculator.get_total_tax_deficits(minimum_ETR=0.15)
        tax_deficits['tax_deficit'] /= self.tax_deficit_calculator.USD_to_EUR
        tax_deficits['tax_deficit'] /= self.tax_deficit_calculator.multiplier_2021

        self.tax_deficits = tax_deficits.copy()

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

        eu_27_country_codes = self.tax_deficit_calculator.eu_27_country_codes.copy()

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
                    'tax_deficit': 'Own tax deficit (billion USD)',
                    'UNADJ_TOTAL_TAX_DEFICITS': 'Based on unadjusted sales (billion USD)',
                    'ADJ_TOTAL_TAX_DEFICITS': 'Based on adjusted sales (billion USD)'
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
                    'UNADJ_ADDITIONAL_REVENUES': 'Based on unadjusted sales (million USD)',
                    'ADJ_ADDITIONAL_REVENUES': 'Based on adjusted sales (million USD)'
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

    def get_relevant_tax_deficits(self, verbose=True, formatted=False):
        """
        With this method, we restrict the estimated tax deficits to the set of countries for which we have prepared an
        adjusted sales mapping.
        """
        unique_parent_countries = self.restr_oecd_sales_mapping['PARENT_COUNTRY_CODE'].unique()

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

        # ### Restricting to countries for which we have sales mappings -------------------------------------------- ###

        restr_tax_deficits = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(unique_parent_countries)
        ].copy()

        if verbose:
            numerator = restr_tax_deficits['tax_deficit'].sum() / 10**9
            denominator = tax_deficits['tax_deficit'].sum() / 10**9
            ratio = numerator / denominator * 100

            print(
                f'We restrict the set of relevant tax deficit estimates to {len(unique_parent_countries)}'
                + ' unique parent countries for which we have satisfying unadjusted and adjusted sales mappings.'
            )

            print(
                f'They represent a total tax deficit of {round(numerator, 2)} billion USD.'
            )

            print(
                f'Vs. a full sample estimate of {round(denominator, 2)} billion USD.'
            )

            print(
                f'So the "coverage rate" for the estimated tax deficits is of {round(ratio, 1)}%.'
            )

        # ### Formatting the table --------------------------------------------------------------------------------- ###
        if formatted:
            restr_tax_deficits['tax_deficit'] /= 10**9

            restr_tax_deficits = restr_tax_deficits.drop(columns=['Parent jurisdiction (alpha-3 code)'])

            restr_tax_deficits = restr_tax_deficits.rename(
                columns={
                    'Parent jurisdiction (whitespaces cleaned)': 'Headquarter country',
                    'tax_deficit': 'Tax deficit to be allocated (billion USD)'
                }
            )

            restr_tax_deficits = restr_tax_deficits.reset_index(drop=True)

        return restr_tax_deficits.copy()

    def get_set_of_countries_focus_table(self, verbose=False):

        # Identifying the partner jurisdictions reported by all the relevant parent countries
        df = self.restr_oecd_sales_mapping.copy()

        df = df.groupby('AFFILIATE_COUNTRY_CODE').nunique()['PARENT_COUNTRY_CODE'].reset_index()

        df = df.rename(columns={'PARENT_COUNTRY_CODE': 'PARENT_COUNT'})

        df = df[
            df['PARENT_COUNT'] == self.restr_oecd_sales_mapping['PARENT_COUNTRY_CODE'].nunique()
        ].copy()

        recurring_partner_codes = list(
            df['AFFILIATE_COUNTRY_CODE'].unique()
        )

        # Full set of countries on which we focus
        full_set = np.unique(recurring_partner_codes)

        full_set = pd.DataFrame(full_set)
        full_set.columns = ['COUNTRY_CODE']

        geographies = pd.read_csv(self.path_to_geographies)
        geographies = geographies.groupby('CODE').first()['NAME'].reset_index()

        full_set = full_set.merge(
            geographies,
            how='left',
            left_on='COUNTRY_CODE', right_on='CODE'
        )

        full_set['NAME'] = full_set.apply(
            lambda row: {'UKI': 'UK Caribbean Islands'}.get(row['COUNTRY_CODE'], row['NAME']),
            axis=1
        )

        # Adding the IS_TAX_HAVEN indicator
        full_set['IS_TAX_HAVEN'] = full_set['COUNTRY_CODE'].isin(
            self.tax_deficit_calculator.tax_haven_country_codes.copy() + ['UKI']
        ) * 1

        # Finalising the table
        full_set = full_set.drop(columns=['COUNTRY_CODE'])

        full_set = full_set.sort_values(by='NAME').reset_index(drop=True)

        return full_set.copy()

    def get_full_sales_apportionment(self, elim_negative_revenues=True):

        oecd_sales_mapping = self.restr_oecd_sales_mapping.copy()
        adjusted_sales_mapping = self.restr_adjusted_sales_mapping.copy()

        restr_tax_deficits = self.get_relevant_tax_deficits(verbose=False, formatted=False)

        focus_set = self.get_set_of_countries_focus_table()
        focus_set_codes = focus_set['CODE'].unique()

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

        merged_df = merged_df[merged_df['AFFILIATE_COUNTRY_CODE'].isin(focus_set_codes)].copy()

        merged_df['ATTRIBUTED_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        unadjusted_temp = merged_df.groupby('AFFILIATE_COUNTRY_CODE').sum()['ATTRIBUTED_REVENUES'].to_dict()

        self.unadjusted_temp = unadjusted_temp.copy()

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

        merged_df = merged_df[merged_df['OTHER_COUNTRY_CODE'].isin(focus_set_codes)].copy()

        merged_df['ATTRIBUTED_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        adjusted_temp = merged_df.groupby('OTHER_COUNTRY_CODE').sum()['ATTRIBUTED_REVENUES'].to_dict()

        self.adjusted_temp = adjusted_temp.copy()

        # ### Deducing the unadjusted distribution of revenues from foreign MNEs only ------------------------------ ###

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

        merged_df = merged_df[merged_df['AFFILIATE_COUNTRY_CODE'].isin(focus_set_codes)].copy()

        merged_df = merged_df[merged_df['PARENT_COUNTRY_CODE'] != merged_df['AFFILIATE_COUNTRY_CODE']].copy()

        merged_df['ATTRIBUTED_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        unadjusted_foreign_temp = merged_df.groupby('AFFILIATE_COUNTRY_CODE').sum()['ATTRIBUTED_REVENUES'].to_dict()

        self.unadjusted_foreign_temp = unadjusted_foreign_temp.copy()

        # ### Deducing the unadjusted distribution of revenues from foreign MNEs only ------------------------------ ###

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

        merged_df = merged_df[merged_df['OTHER_COUNTRY_CODE'].isin(focus_set_codes)].copy()

        merged_df = merged_df[merged_df['PARENT_COUNTRY_CODE'] != merged_df['OTHER_COUNTRY_CODE']].copy()

        merged_df['ATTRIBUTED_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        adjusted_foreign_temp = merged_df.groupby('OTHER_COUNTRY_CODE').sum()['ATTRIBUTED_REVENUES'].to_dict()

        self.adjusted_foreign_temp = adjusted_foreign_temp.copy()

        # ### Gathering the different results in a single table ---------------------------------------------------- ###

        focus_set[
            'Full apportionment based on unadjusted sales (billion USD)'
        ] = focus_set['CODE'].map(
            lambda x: unadjusted_temp.get(x, 0)
        ) / 10**9

        focus_set[
            'Full apportionment based on adjusted sales (billion USD)'
        ] = focus_set['CODE'].map(
            lambda x: adjusted_temp.get(x, 0)
        ) / 10**9

        focus_set[
            'Foreign collection only, based on unadjusted sales (million USD)'
        ] = focus_set['CODE'].map(
            lambda x: unadjusted_foreign_temp.get(x, 0)
        ) / 10**6

        focus_set[
            'Foreign collection only, based on adjusted sales (million USD)'
        ] = focus_set['CODE'].map(
            lambda x: adjusted_foreign_temp.get(x, 0)
        ) / 10**6

        # ### Formatting the table --------------------------------------------------------------------------------- ###

        ser = focus_set.sum()
        ser.loc['NAME'] = 'Full sample - Total'
        focus_set = pd.concat([focus_set, pd.DataFrame(ser).T], axis=0)

        ser = focus_set[focus_set['IS_TAX_HAVEN'] == 0].sum()
        ser.loc['NAME'] = 'Non-havens - Total'
        ser.loc['IS_TAX_HAVEN'] = 0.5
        focus_set = pd.concat([focus_set, pd.DataFrame(ser).T], axis=0)

        ser = focus_set[focus_set['IS_TAX_HAVEN'] == 1].sum()
        ser.loc['NAME'] = 'Tax havens - Total'
        ser.loc['IS_TAX_HAVEN'] = 1.5
        focus_set = pd.concat([focus_set, pd.DataFrame(ser).T], axis=0)

        focus_set = focus_set.sort_values(by=['IS_TAX_HAVEN', 'NAME'])
        focus_set = focus_set.drop(columns=['IS_TAX_HAVEN', 'CODE'])
        focus_set = focus_set.rename(columns={'NAME': 'Country name'})

        return focus_set.copy()
