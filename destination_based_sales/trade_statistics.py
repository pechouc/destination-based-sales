
########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

from destination_based_sales.comtrade import UNComtradeProcessor
from destination_based_sales.balanced_trade import BalancedTradeStatsProcessor
from destination_based_sales.bop import USBalanceOfPaymentsProcessor

from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor

from destination_based_sales.utils import UK_CARIBBEAN_ISLANDS, ensure_country_overlap_with_IRS, \
    ensure_country_overlap_with_OECD_CbCR


path_to_dir = os.path.dirname(os.path.abspath(__file__))
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class TradeStatisticsProcessor:

    def __init__(
        self,
        year,
        US_only,
        US_merchandise_exports_source,
        US_services_exports_source,
        non_US_merchandise_exports_source,
        non_US_services_exports_source,
        winsorize_export_percs,
        non_US_winsorizing_threshold=None,
        US_winsorizing_threshold=None,
        service_flows_to_exclude=None,
        path_to_geographies=path_to_geographies
    ):

        # Checking the chosen mix of data sources
        if US_merchandise_exports_source not in ['BIMTS', 'BoP', 'Comtrade']:
            raise Exception('US merchandise exports can only be sourced from "BIMTS", "BoP" or "Comtrade".')

        if US_services_exports_source not in ['BaTIS', 'BoP']:
            raise Exception('US merchandise exports can only be sourced from "BaTIS" or "BoP".')

        if non_US_merchandise_exports_source not in ['BIMTS', 'Comtrade']:
            raise Exception('Non-US merchandise exports can only be sourced from "BIMTS" or "Comtrade".')

        if non_US_services_exports_source not in ['BaTIS']:
            raise Exception('US merchandise exports can only be sourced from "BaTIS".')

        # Saving useful attributes
        self.year = year
        self.US_only = US_only
        self.winsorize_export_percs = winsorize_export_percs
        self.service_flows_to_exclude = service_flows_to_exclude

        if winsorize_export_percs:
            if non_US_winsorizing_threshold is None or US_winsorizing_threshold is None:
                raise Exception('If you want to winsorize export distributions, winsorizing thresholds must be passed.')

            self.winsorizing_threshold = non_US_winsorizing_threshold / 100
            self.winsorizing_threshold_US = US_winsorizing_threshold / 100

        # Saving the data source mix attributes
        self.US_merchandise_exports_source = US_merchandise_exports_source
        self.US_services_exports_source = US_services_exports_source
        self.non_US_merchandise_exports_source = non_US_merchandise_exports_source
        self.non_US_services_exports_source = non_US_services_exports_source

        # Loading geographies
        self.path_to_geographies = path_to_geographies
        self.geographies = pd.read_csv(path_to_geographies)

        # Loading BaTIS data once and for all if needed as it can be quite long
        if 'BaTIS' in [non_US_services_exports_source, US_services_exports_source]:
            processor = BalancedTradeStatsProcessor(
                year=self.year,
                service_flows_to_exclude=self.service_flows_to_exclude
            )
            self.services_batis = processor.load_clean_services_data()

        # Useful for the overlap with IRS and OECD country-by-country data
        self.UK_CARIBBEAN_ISLANDS = UK_CARIBBEAN_ISLANDS.copy()

        # If we are focusing only on the adjustment of US multinationals' sales, we match the IRS dataset
        # We also do if the year considered in 2018 since, for this one, we do not have the OECD's CbCR data yet
        if self.US_only or self.year == 2018:
            preprocessor = IRSDataPreprocessor(year=year)
            self.unique_IRS_country_codes = preprocessor.load_final_data()['CODE'].unique()

        else:
            oecd_preprocessor = CbCRPreprocessor(year=year)
            temp = oecd_preprocessor.get_preprocessed_revenue_data()
            self.unique_OECD_country_codes = temp['AFFILIATE_COUNTRY_CODE'].unique()

    def load_merchandise_and_services_data(self):

        # ### Loading US exports data

        # Merchandise
        if self.US_merchandise_exports_source == 'BIMTS':
            processor = BalancedTradeStatsProcessor(year=self.year)
            us_merchandise = processor.load_clean_merchandise_data()
            us_merchandise = us_merchandise[us_merchandise['AFFILIATE_COUNTRY_CODE'] == 'USA'].copy()

        elif self.US_merchandise_exports_source == 'BoP':
            processor = USBalanceOfPaymentsProcessor(year=self.year)
            us_merchandise = processor.load_final_merchandise_data()

        else:   # Alternative is to use Comtrade
            processor = UNComtradeProcessor(year=self.year)
            us_merchandise = processor.load_data_with_geographies()
            us_merchandise = us_merchandise[us_merchandise['AFFILIATE_COUNTRY_CODE'] == 'USA'].copy()

        # Services
        if self.US_services_exports_source == 'BaTIS':
            services_batis = self.services_batis.copy()
            us_services = services_batis[services_batis['AFFILIATE_COUNTRY_CODE'] == 'USA'].copy()

        else:   # Alternative is to use the US BoP
            processor = USBalanceOfPaymentsProcessor(year=self.year)
            us_services = processor.load_final_services_data()

        # ### Loading non-US exports data

        # Merchandise
        if self.non_US_merchandise_exports_source == 'BIMTS':
            processor = BalancedTradeStatsProcessor(year=self.year)
            merchandise = processor.load_clean_merchandise_data()
            merchandise = merchandise[merchandise['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

        else:   # Alternative is to use Comtrade
            processor = UNComtradeProcessor(year=self.year)
            merchandise = processor.load_data_with_geographies()
            merchandise = merchandise[merchandise['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

        # Services
        if self.non_US_services_exports_source == 'BaTIS':
            services_batis = self.services_batis.copy()
            services = services_batis[services_batis['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

        # Concatening the resulting DataFrames
        merchandise = pd.concat([merchandise, us_merchandise], axis=0)
        services = pd.concat([services, us_services], axis=0)

        return merchandise.copy(), services.copy()

    def load_overlapping_merchandise_and_exports_data(self):

        merchandise, services = self.load_merchandise_and_services_data()

        if self.US_only or self.year == 2018:
            # In that case, we must ensure consistency with the IRS' country-by-country data

            # For merchandise exports
            merchandise['OTHER_COUNTRY_CODE'] = merchandise.apply(
                lambda row: ensure_country_overlap_with_IRS(
                    row,
                    self.unique_IRS_country_codes,
                    self.UK_CARIBBEAN_ISLANDS
                ),
                axis=1
            )

            # For exports of services
            services['OTHER_COUNTRY_CODE'] = services.apply(
                lambda row: ensure_country_overlap_with_IRS(
                    row,
                    self.unique_IRS_country_codes,
                    self.UK_CARIBBEAN_ISLANDS
                ),
                axis=1
            )

        else:
            # In that case, we must ensure consistency with the OECD's country-by-country data

            merchandise['OTHER_COUNTRY_CODE'] = merchandise.apply(
                lambda row: ensure_country_overlap_with_OECD_CbCR(
                    row,
                    self.unique_OECD_country_codes,
                    self.UK_CARIBBEAN_ISLANDS
                ),
                axis=1
            )

            services['OTHER_COUNTRY_CODE'] = services.apply(
                lambda row: ensure_country_overlap_with_OECD_CbCR(
                    row,
                    self.unique_OECD_country_codes,
                    self.UK_CARIBBEAN_ISLANDS
                ),
                axis=1
            )

        merchandise = merchandise.groupby(
            [
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CONTINENT_CODE',
                'OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE'
            ]
        ).sum().reset_index()

        services = services.groupby(
            [
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CONTINENT_CODE',
                'OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE'
            ]
        ).sum().reset_index()

        return merchandise.copy(), services.copy()

    def load_merged_data(self):
        merchandise, services = self.load_overlapping_merchandise_and_exports_data()

        # We execute an OUTER merge
        merged_df = merchandise.merge(
            services,
            how='outer',
            on=[
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CONTINENT_CODE',
                'OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE'
            ]
        )

        # We replace missing values by 0 (assumptions described in the report)
        for column in ['MERCHANDISE_EXPORTS', 'SERVICES_EXPORTS']:
            merged_df[column] = merged_df[column].fillna(0)

        # We compute total exports as the sum of merchandise and services exports
        merged_df['ALL_EXPORTS'] = merged_df['MERCHANDISE_EXPORTS'] + merged_df['SERVICES_EXPORTS']

        return merged_df.copy()

    def compute_exports_per_continent(self):
        merged_df = self.load_merged_data()

        # We want a distribution of exports for each continent
        exports_per_continent = {}

        for continent in merged_df['AFFILIATE_COUNTRY_CONTINENT_CODE'].unique():
            # Iterating over unique affiliate country continent codes, we compute each distribution of exports
            restricted_df = merged_df[merged_df['AFFILIATE_COUNTRY_CONTINENT_CODE'] == continent].copy()

            restricted_df = restricted_df[
                ['OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE', 'ALL_EXPORTS']
            ].copy()

            restricted_df = restricted_df.groupby(
                ['OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE']
            ).sum().reset_index()

            # And we store it in the dedicated dictionary
            exports_per_continent[continent] = restricted_df.copy()

        # If we want to match the OECD's CbCR data, we also need a distribution of exports for the "Other groups"
        # We assume that it corresponds to the mean distribution of exports over all countries
        if not self.US_only:
            other_groups_df = merged_df[
                ['OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE', 'ALL_EXPORTS']
            ].copy()

            other_groups_df = other_groups_df.groupby(
                ['OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE']
            ).sum().reset_index()

            exports_per_continent['OTHER_GROUPS'] = other_groups_df.copy()

        return exports_per_continent.copy()

    def load_extended_exports_distributions(self):

        # We load unextended data and the exports distributions of each continent
        merged_df = self.load_merged_data()
        exports_per_continent = self.compute_exports_per_continent()

        # We first determine the complete set of affiliate countries that we want to cover
        if self.US_only or self.year == 2018:

            # In this case, we want to cover all the affiliate countries present in the IRS' country-by-country data
            processor = IRSDataPreprocessor(year=self.year)
            all_countries = processor.load_final_data()

            all_countries = all_countries.rename(
                columns={
                    'CODE': 'AFFILIATE_COUNTRY_CODE',
                    'CONTINENT_CODE': 'AFFILIATE_COUNTRY_CONTINENT_CODE'
                }
            )

            all_countries = all_countries[
                ['AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CONTINENT_CODE']
            ].drop_duplicates()

        else:

            # In this case, we want to cover all the affiliate countries present in the OECD's country-by-country data
            processor = CbCRPreprocessor(year=self.year)
            all_countries = processor.get_preprocessed_revenue_data()

            all_countries = all_countries.rename(
                columns={
                    'CONTINENT_CODE': 'AFFILIATE_COUNTRY_CONTINENT_CODE'
                }
            )

            all_countries = all_countries[
                ['AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CONTINENT_CODE']
            ].drop_duplicates()

        # For what affiliate countries are we missing a valid distribution of exports in our trade statistics?
        # Countries that are either absent from the DataFrame or that only display 0 exports for all partners
        # The latter case may happen in some cases depending on the chosen mix of data sources
        temp = merged_df.groupby('AFFILIATE_COUNTRY_CODE').sum()['ALL_EXPORTS']
        self.valid_exports_distributions = list(temp[temp > 0].index)

        missing_countries = all_countries[
            ~all_countries['AFFILIATE_COUNTRY_CODE'].isin(self.valid_exports_distributions)
        ].copy()

        # Starting from the exports data that we do have in the "merged_df" DataFrame, we add all the missing countries
        output_df = merged_df[merged_df['AFFILIATE_COUNTRY_CODE'].isin(self.valid_exports_distributions)].copy()

        output_df = output_df[
            [
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CONTINENT_CODE',
                'OTHER_COUNTRY_CODE', 'OTHER_COUNTRY_CONTINENT_CODE', 'ALL_EXPORTS'
            ]
        ].copy()

        for _, row in missing_countries.iterrows():

            # We will impute the distribution of exports of the corresponding continent
            imputed_df = exports_per_continent[row['AFFILIATE_COUNTRY_CONTINENT_CODE']].copy()

            # We simply eliminate the missing affiliate country from the destinations
            imputed_df = imputed_df[imputed_df['OTHER_COUNTRY_CODE'] != row['AFFILIATE_COUNTRY_CODE']].copy()

            # And we add the two columns that are missing to match the format of the merged_df DataFrame
            imputed_df['AFFILIATE_COUNTRY_CODE'] = row['AFFILIATE_COUNTRY_CODE']
            imputed_df['AFFILIATE_COUNTRY_CONTINENT_CODE'] = row['AFFILIATE_COUNTRY_CONTINENT_CODE']

            # We append the imputed DataFrame to the central one
            output_df = pd.concat([output_df, imputed_df], axis=0)

        return output_df.copy()

    def get_final_exports_distributions(self):

        # We load extended data
        output_df = self.load_extended_exports_distributions()

        # ### We first add the EXPORT_PERC column

        # We compute the total exports for each affiliate country in the dataset
        totals = {}

        for country_code in output_df['AFFILIATE_COUNTRY_CODE'].unique():

            restricted_df = output_df[output_df['AFFILIATE_COUNTRY_CODE'] == country_code].copy()
            totals[country_code] = restricted_df['ALL_EXPORTS'].sum()

            # TEMPORARY PRINT STATEMENT
            if totals[country_code] == 0:
                print(country_code)

        # This allows to add a temporary TOTAL_EXPORTS column
        output_df['TOTAL_EXPORTS'] = output_df['AFFILIATE_COUNTRY_CODE'].map(totals)

        # From which we deduce the EXPORT_PERC column
        output_df['EXPORT_PERC'] = output_df['ALL_EXPORTS'] / output_df['TOTAL_EXPORTS']

        # And the TOTAL_EXPORTS column is not useful anymore
        output_df = output_df.drop(columns=['TOTAL_EXPORTS'])

        if not self.winsorize_export_percs:
            # We do not need to winsorize and we can therefore return the DataFrame as is
            return output_df.reset_index(drop=True)

        else:

            # ### In this case, we need to winsorize the exports distributions

            # Winsorizing the US exports distribution
            us_extract = output_df[output_df['AFFILIATE_COUNTRY_CODE'] == 'USA'].copy()

            us_extract = us_extract[us_extract['EXPORT_PERC'] > self.winsorizing_threshold_US].copy()

            us_extract['EXPORT_PERC'] = us_extract['ALL_EXPORTS'] / us_extract['ALL_EXPORTS'].sum()

            # Winsorizing the non-US exports distributions
            non_us_extract = output_df[output_df['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

            non_us_extract = non_us_extract[non_us_extract['EXPORT_PERC'] > self.winsorizing_threshold].copy()

            new_totals = {}

            for country_code in non_us_extract['AFFILIATE_COUNTRY_CODE'].unique():

                restricted_df = non_us_extract[non_us_extract['AFFILIATE_COUNTRY_CODE'] == country_code].copy()
                new_totals[country_code] = restricted_df['ALL_EXPORTS'].sum()

            non_us_extract['TOTAL_EXPORTS'] = non_us_extract['AFFILIATE_COUNTRY_CODE'].map(new_totals)
            non_us_extract['EXPORT_PERC'] = non_us_extract['ALL_EXPORTS'] / non_us_extract['TOTAL_EXPORTS']
            non_us_extract = non_us_extract.drop(columns=['TOTAL_EXPORTS'])

            # We can concatenate the winsorized exports distributions and return the resulting DataFrame
            output_df = pd.concat([us_extract, non_us_extract], axis=0)

            return output_df.reset_index(drop=True)
