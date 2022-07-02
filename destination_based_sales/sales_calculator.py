########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.analytical_amne import AnalyticalAMNEPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor
from destination_based_sales.bea import ExtendedBEADataLoader
from destination_based_sales.trade_statistics import TradeStatisticsProcessor


class USSalesCalculator:

    def __init__(
        self,
        year,
        US_only,
        US_merchandise_exports_source,
        US_services_exports_source,
        non_US_merchandise_exports_source,
        non_US_services_exports_source,
        winsorize_export_percs,
        US_winsorizing_threshold=None,
        non_US_winsorizing_threshold=None,
        service_flows_to_exclude=None,
        load_data_online=False,
    ):
        self.year = year
        self.load_data_online = load_data_online

        # ### Loading the required data

        # IRS' country-by-country data
        irs_preprocessor = IRSDataPreprocessor(year=year, load_data_online=load_data_online)
        self.irs = irs_preprocessor.load_final_data()

        # Extended BEA data
        self.bea_loader = ExtendedBEADataLoader(year=year, load_data_online=load_data_online)
        self.sales_percentages = self.bea_loader.get_extended_sales_percentages()

        # Trade statistics (can be quite long!)
        self.trade_statistics, self.unrestricted_trade_statistics = self.load_trade_statistics(
            year=year,
            US_only=US_only,
            US_merchandise_exports_source=US_merchandise_exports_source,
            US_services_exports_source=US_services_exports_source,
            non_US_merchandise_exports_source=non_US_merchandise_exports_source,
            non_US_services_exports_source=non_US_services_exports_source,
            winsorize_export_percs=winsorize_export_percs,
            US_winsorizing_threshold=US_winsorizing_threshold,
            non_US_winsorizing_threshold=non_US_winsorizing_threshold,
            service_flows_to_exclude=service_flows_to_exclude
        )

    def load_trade_statistics(
        self,
        year,
        US_only,
        US_merchandise_exports_source,
        US_services_exports_source,
        non_US_merchandise_exports_source,
        non_US_services_exports_source,
        winsorize_export_percs,
        US_winsorizing_threshold,
        non_US_winsorizing_threshold,
        service_flows_to_exclude
    ):

        trade_stat_processor = TradeStatisticsProcessor(
            year=year,
            US_only=US_only,
            US_merchandise_exports_source=US_merchandise_exports_source,
            US_services_exports_source=US_services_exports_source,
            non_US_merchandise_exports_source=non_US_merchandise_exports_source,
            non_US_services_exports_source=non_US_services_exports_source,
            winsorize_export_percs=winsorize_export_percs,
            US_winsorizing_threshold=US_winsorizing_threshold,
            non_US_winsorizing_threshold=non_US_winsorizing_threshold,
            service_flows_to_exclude=service_flows_to_exclude,
            load_data_online=self.load_data_online
        )

        trade_statistics = trade_stat_processor.get_final_exports_distributions()

        trade_stats_extract = trade_statistics[trade_statistics['OTHER_COUNTRY_CODE'] != 'USA'].copy()

        new_totals = trade_stats_extract.groupby('AFFILIATE_COUNTRY_CODE').sum()['ALL_EXPORTS'].to_dict()

        trade_stats_extract['TOTAL_EXPORTS'] = trade_stats_extract['AFFILIATE_COUNTRY_CODE'].map(new_totals)
        trade_stats_extract['EXPORT_PERC'] = trade_stats_extract['ALL_EXPORTS'] / trade_stats_extract['TOTAL_EXPORTS']
        trade_stats_extract = trade_stats_extract.drop(columns=['TOTAL_EXPORTS'])

        return trade_stats_extract.copy(), trade_statistics.copy()

    def get_merged_df_with_absolute_amounts(self):

        # Merging the IRS and BEA DataFrames
        merged_df = self.irs.merge(
            self.sales_percentages,
            how='left',
            left_on='CODE', right_on='AFFILIATE_COUNTRY_CODE'
        )

        merged_df = merged_df.drop(
            columns=[
                'AFFILIATE_COUNTRY_NAME_x', 'CONTINENT_CODE_x',
                'CONTINENT_CODE_y', 'CODE', 'CONTINENT_NAME'
            ]
        )

        merged_df = merged_df.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME_y': 'AFFILIATE_COUNTRY_NAME'
            }
        )

        # Deducing the absolute amounts of sales
        absolute_amount_columns = []

        # We iterate over sales types
        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']:

            sales_type = column.split('_')[0]

            # And then over ultimate destination types
            for destination in ['US', 'OTHER_COUNTRY', 'AFFILIATE_COUNTRY']:

                new_column = column + '_TO_' + destination

                # And we construct the new column as (IRS absolute amounts * BEA sales percentages)
                merged_df[new_column] = (
                    merged_df[column] * merged_df['PERC_' + sales_type + '_' + destination]
                )

                absolute_amount_columns.append(new_column)

        # Cleaning the DataFrame
        merged_df = merged_df.drop(
            columns=self.bea_loader.percentage_columns.copy() + [
                'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES'
            ]
        )

        # Checking the overlap between splitted revenues and trade statistics
        # (logic to be moved to a separate unit test?)
        missing_overlap = (
            ~merged_df['AFFILIATE_COUNTRY_CODE'].isin(
                self.trade_statistics['AFFILIATE_COUNTRY_CODE'].unique()
            )
        ).sum()

        if missing_overlap != 0:
            raise Exception(
                'It seems that some countries in the splitted revenues are not covered by trade statistics yet.'
            )

        return merged_df.copy()

    def get_sales_to_other_foreign_countries(self):

        splitted_revenues = self.get_merged_df_with_absolute_amounts()

        other_country_sales = splitted_revenues[
            [
                'AFFILIATE_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE',
                'UNRELATED_PARTY_REVENUES_TO_OTHER_COUNTRY',
                'RELATED_PARTY_REVENUES_TO_OTHER_COUNTRY',
                'TOTAL_REVENUES_TO_OTHER_COUNTRY'
            ]
        ].copy()

        trade_statistics = self.trade_statistics.copy()

        merged_df = trade_statistics.merge(
            other_country_sales,
            how='inner',
            on='AFFILIATE_COUNTRY_CODE'
        )

        new_columns = []
        existing_columns = []

        for sales_type in ['UNRELATED', 'RELATED', 'TOTAL']:

            if sales_type != 'TOTAL':
                prefix = sales_type + '_PARTY'

            else:
                prefix = sales_type

            new_column = prefix + '_REVENUES'
            new_columns.append(new_column)

            existing_column = new_column + '_TO_OTHER_COUNTRY'
            existing_columns.append(existing_column)

            merged_df[new_column] = merged_df['EXPORT_PERC'] * merged_df[existing_column]

        merged_df.drop(
            columns=existing_columns + [
                'ALL_EXPORTS', 'EXPORT_PERC', 'AFFILIATE_COUNTRY_CONTINENT_CODE', 'OTHER_COUNTRY_CONTINENT_CODE'
            ],
            inplace=True
        )

        return merged_df.copy()

    def get_sales_to_affiliate_countries(self):

        splitted_revenues = self.get_merged_df_with_absolute_amounts()

        affiliate_country_sales = splitted_revenues[
            [
                'AFFILIATE_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY',
                'RELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY', 'TOTAL_REVENUES_TO_AFFILIATE_COUNTRY'
            ]
        ].copy()

        affiliate_country_sales.rename(
            columns={
                'UNRELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY': 'UNRELATED_PARTY_REVENUES',
                'RELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY': 'RELATED_PARTY_REVENUES',
                'TOTAL_REVENUES_TO_AFFILIATE_COUNTRY': 'TOTAL_REVENUES',
                'CODE': 'AFFILIATE_COUNTRY_CODE'
            },
            inplace=True
        )

        affiliate_country_sales['OTHER_COUNTRY_CODE'] = affiliate_country_sales['AFFILIATE_COUNTRY_CODE'].copy()

        return affiliate_country_sales.copy()

    def get_sales_to_the_US(self):

        splitted_revenues = self.get_merged_df_with_absolute_amounts()

        us_sales = splitted_revenues[
            [
                'AFFILIATE_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES_TO_US',
                'RELATED_PARTY_REVENUES_TO_US', 'TOTAL_REVENUES_TO_US'
            ]
        ].copy()

        us_sales.rename(
            columns={
                'UNRELATED_PARTY_REVENUES_TO_US': 'UNRELATED_PARTY_REVENUES',
                'RELATED_PARTY_REVENUES_TO_US': 'RELATED_PARTY_REVENUES',
                'TOTAL_REVENUES_TO_US': 'TOTAL_REVENUES',
                'CODE': 'AFFILIATE_COUNTRY_CODE'
            },
            inplace=True
        )

        us_sales['OTHER_COUNTRY_CODE'] = 'USA'

        return us_sales.copy()

    def get_final_sales_mapping(self):

        merged_df = self.get_sales_to_other_foreign_countries()
        affiliate_country_sales = self.get_sales_to_affiliate_countries()
        us_sales = self.get_sales_to_the_US()

        output_df = pd.concat(
            [merged_df, affiliate_country_sales, us_sales],
            axis=0
        )

        output_df = output_df.groupby(
            ['AFFILIATE_COUNTRY_CODE', 'OTHER_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME']
        ).sum().reset_index()

        return output_df.copy()


class SimplifiedGlobalSalesCalculator:

    def __init__(
        self,
        year,
        aamne_domestic_sales_perc,
        breakdown_threshold,
        US_merchandise_exports_source,
        US_services_exports_source,
        non_US_merchandise_exports_source,
        non_US_services_exports_source,
        winsorize_export_percs,
        US_winsorizing_threshold=None,
        non_US_winsorizing_threshold=None,
        service_flows_to_exclude=None,
        load_data_online=False
    ):
        if year not in [2016, 2017]:
            raise Exception(
                "Computations can only be run for 2016 and 2017, as we are constrained by the availability "
                + "of the OECD's aggregated and anonymized country-by-country data."
            )

        self.year = year
        self.aamne_domestic_sales_perc = aamne_domestic_sales_perc
        self.service_flows_to_exclude = service_flows_to_exclude
        self.load_data_online = load_data_online

        # ### Loading the required data

        # OECD's country-by-country data
        self.breakdown_threshold = breakdown_threshold
        cbcr_preprocessor = CbCRPreprocessor(
            year=year,
            breakdown_threshold=breakdown_threshold,
            load_data_online=load_data_online
        )
        self.oecd = cbcr_preprocessor.get_preprocessed_revenue_data()

        # Extended BEA data
        self.bea_loader = ExtendedBEADataLoader(year=year, load_data_online=load_data_online)
        self.sales_percentages = self.bea_loader.get_extended_sales_percentages()

        # US sales mapping and trade statistics
        calculator = USSalesCalculator(
            year=year,
            US_only=False,
            US_merchandise_exports_source=US_merchandise_exports_source,
            US_services_exports_source=US_services_exports_source,
            non_US_merchandise_exports_source=non_US_merchandise_exports_source,
            non_US_services_exports_source=non_US_services_exports_source,
            winsorize_export_percs=winsorize_export_percs,
            US_winsorizing_threshold=US_winsorizing_threshold,
            non_US_winsorizing_threshold=non_US_winsorizing_threshold,
            service_flows_to_exclude=service_flows_to_exclude,
            load_data_online=load_data_online
        )

        self.trade_statistics = calculator.unrestricted_trade_statistics.copy()
        self.US_sales_mapping = calculator.get_final_sales_mapping()

        if self.aamne_domestic_sales_perc:
            # If we want to use the Analytical AMNE database instead of BEA statistics for domestic sales percentages
            preprocessor = AnalyticalAMNEPreprocessor(load_data_online=load_data_online)

            aamne_domestic = preprocessor.get_unextended_domestic_analytical_amne_data()

            affiliate_country_ratios = aamne_domestic.copy()
            affiliate_country_ratios['AFFILIATE_COUNTRY'] = (
                affiliate_country_ratios['DOMESTIC_SALES'] /
                (
                    affiliate_country_ratios['DOMESTIC_SALES'] + affiliate_country_ratios['SALES_TO_OTHER_COUNTRY']
                )
            )
            affiliate_country_ratios = affiliate_country_ratios[['COUNTRY_CODE', 'AFFILIATE_COUNTRY']]
            affiliate_country_ratios = affiliate_country_ratios.set_index('COUNTRY_CODE').to_dict()['AFFILIATE_COUNTRY']

            export_ratios = aamne_domestic.copy()
            export_ratios['EXPORTS'] = (
                export_ratios['SALES_TO_OTHER_COUNTRY'] /
                (
                    export_ratios['DOMESTIC_SALES'] + export_ratios['SALES_TO_OTHER_COUNTRY']
                )
            )
            export_ratios = export_ratios[['COUNTRY_CODE', 'EXPORTS']]
            export_ratios = export_ratios.set_index('COUNTRY_CODE').to_dict()['EXPORTS']

            self.affiliate_country_ratios = affiliate_country_ratios.copy()
            self.export_ratios = export_ratios.copy()

    def get_adjusted_sales_percentages(self):

        sales_percentages = self.sales_percentages.copy()

        to_be_dropped = []

        for sales_type in ['RELATED', 'UNRELATED', 'TOTAL']:
            new_column = f'PERC_{sales_type}_EXPORTS'

            column_1 = f'PERC_{sales_type}_US'
            column_2 = f'PERC_{sales_type}_OTHER_COUNTRY'

            sales_percentages[new_column] = sales_percentages[column_1] + sales_percentages[column_2]

            to_be_dropped += [column_1, column_2]

        sales_percentages = sales_percentages.drop(columns=to_be_dropped)

        return sales_percentages.copy()

    def get_merged_df_with_absolute_amounts(self):

        oecd = self.oecd.copy()

        # We build the adjusted sales percentages thanks to the method defined above
        sales_percentages = self.get_adjusted_sales_percentages()

        # We add the sales percentages to the OECD's CbCR data
        merged_df = oecd.merge(
            sales_percentages.drop(
                columns=['AFFILIATE_COUNTRY_NAME', 'CONTINENT_CODE']
            ),
            how='left',
            on='AFFILIATE_COUNTRY_CODE'
        )

        if self.aamne_domestic_sales_perc:
            # If relevant, we replace the BEA domestic sales percentages with the ones from the Analytical AMNE database
            indices = merged_df[
                np.logical_and(
                    merged_df['PARENT_COUNTRY_CODE'] == merged_df['AFFILIATE_COUNTRY_CODE'],
                    merged_df['PARENT_COUNTRY_CODE'] != 'USA'
                )
            ].index

            # indices = merged_df[merged_df['PARENT_COUNTRY_CODE'] == merged_df['AFFILIATE_COUNTRY_CODE']].index

            for i in indices:
                for col in merged_df.columns:
                    if col.endswith('AFFILIATE_COUNTRY'):
                        merged_df.loc[i, col] = self.affiliate_country_ratios.get(
                            merged_df.loc[i, 'PARENT_COUNTRY_CODE'], merged_df.loc[i, col]
                        )

                    elif col.endswith('EXPORTS'):
                        merged_df.loc[i, col] = self.export_ratios.get(
                            merged_df.loc[i, 'PARENT_COUNTRY_CODE'], merged_df.loc[i, col]
                        )

                    else:
                        continue

        # Deducing the absolute amounts of sales
        absolute_amount_columns = []
        percentage_columns = []
        revenue_columns = ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']

        # We iterate over sales types
        for column in revenue_columns:

            sales_type = column.split('_')[0]

            # And then over ultimate destination types
            for destination in ['EXPORTS', 'AFFILIATE_COUNTRY']:

                new_column = column + '_TO_' + destination
                percentage_column = 'PERC_' + sales_type + '_' + destination

                # And we construct the new column as (IRS absolute amounts * BEA sales percentages)
                merged_df[new_column] = (
                    merged_df[column] * merged_df[percentage_column]
                )

                absolute_amount_columns.append(new_column)
                percentage_columns.append(percentage_column)

        merged_df = merged_df.drop(columns=percentage_columns + revenue_columns)

        merged_df = merged_df[merged_df['PARENT_COUNTRY_CODE'] != 'USA'].copy()

        # Checking the overlap between splitted revenues and trade statistics
        # (logic to be moved to a separate unit test?)
        missing_overlap = (
            ~merged_df['AFFILIATE_COUNTRY_CODE'].isin(
                self.trade_statistics['AFFILIATE_COUNTRY_CODE'].unique()
            )
        ).sum()

        if missing_overlap != 0:
            raise Exception(
                'It seems that some countries in the splitted revenues are not covered by trade statistics yet.'
            )

        return merged_df.copy()

    def get_export_sales(self):

        splitted_revenues = self.get_merged_df_with_absolute_amounts()

        export_sales = splitted_revenues[
            [
                'PARENT_COUNTRY_NAME', 'PARENT_COUNTRY_CODE',
                'AFFILIATE_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE',
                'UNRELATED_PARTY_REVENUES_TO_EXPORTS',
                'RELATED_PARTY_REVENUES_TO_EXPORTS',
                'TOTAL_REVENUES_TO_EXPORTS'
            ]
        ].copy()

        trade_statistics = self.trade_statistics.copy()

        merged_df = trade_statistics.merge(
            export_sales,
            how='inner',
            on='AFFILIATE_COUNTRY_CODE'
        )

        new_columns = []
        existing_columns = []

        for sales_type in ['UNRELATED', 'RELATED', 'TOTAL']:

            if sales_type != 'TOTAL':
                prefix = sales_type + '_PARTY'

            else:
                prefix = sales_type

            new_column = prefix + '_REVENUES'
            new_columns.append(new_column)

            existing_column = new_column + '_TO_EXPORTS'
            existing_columns.append(existing_column)

            merged_df[new_column] = merged_df['EXPORT_PERC'] * merged_df[existing_column]

        merged_df = merged_df.drop(
            columns=existing_columns + [
                'ALL_EXPORTS', 'EXPORT_PERC', 'AFFILIATE_COUNTRY_CONTINENT_CODE', 'OTHER_COUNTRY_CONTINENT_CODE'
            ]
        )

        return merged_df.copy()

    def get_sales_to_affiliate_countries(self):

        splitted_revenues = self.get_merged_df_with_absolute_amounts()

        affiliate_country_sales = splitted_revenues[
            [
                'PARENT_COUNTRY_CODE', 'PARENT_COUNTRY_NAME', 'AFFILIATE_COUNTRY_NAME',
                'AFFILIATE_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY',
                'RELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY', 'TOTAL_REVENUES_TO_AFFILIATE_COUNTRY'
            ]
        ].copy()

        affiliate_country_sales.rename(
            columns={
                'UNRELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY': 'UNRELATED_PARTY_REVENUES',
                'RELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY': 'RELATED_PARTY_REVENUES',
                'TOTAL_REVENUES_TO_AFFILIATE_COUNTRY': 'TOTAL_REVENUES'
            },
            inplace=True
        )

        affiliate_country_sales['OTHER_COUNTRY_CODE'] = affiliate_country_sales['AFFILIATE_COUNTRY_CODE'].copy()

        return affiliate_country_sales.copy()

    def get_final_sales_mapping(self):

        # We map sales to affiliate countries and sales that are in fact exports with the two previous methods
        merged_df = self.get_export_sales()
        affiliate_country_sales = self.get_sales_to_affiliate_countries()

        # We also make use of the mapping of US multinationals' sales
        US_sales_mapping = self.US_sales_mapping.copy()

        # Removing unnecessary columns
        merged_df = merged_df.drop(columns=['AFFILIATE_COUNTRY_NAME', 'PARENT_COUNTRY_NAME'])
        affiliate_country_sales = affiliate_country_sales.drop(
            columns=['PARENT_COUNTRY_NAME', 'AFFILIATE_COUNTRY_NAME']
        )
        US_sales_mapping = US_sales_mapping.drop(columns=['AFFILIATE_COUNTRY_NAME'])

        # Adding the parent country code column to the DataFrame with the US sales mapping
        US_sales_mapping['PARENT_COUNTRY_CODE'] = 'USA'

        output_df = pd.concat(
            [merged_df, affiliate_country_sales, US_sales_mapping],
            axis=0
        )

        output_df = output_df[
            [
                'PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE', 'OTHER_COUNTRY_CODE',
                'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES'
            ]
        ].sort_values(
            by=['PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE']
        ).reset_index(
            drop=True
        )

        return output_df.copy()
