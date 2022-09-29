"""
Building upon the computation of destination-based sales, this module provides various analyses of the results and
allows to output the tables or graphs presented in the study (paper or appendix). It combines most of the data sources
used throughout this work.
"""

########################################################################################################################
# --- Imports

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor
from destination_based_sales.bea import ExtendedBEADataLoader
from destination_based_sales.trade_statistics import TradeStatisticsProcessor
from destination_based_sales.per_industry import PerIndustryAnalyser
from destination_based_sales.sales_calculator import USSalesCalculator, SimplifiedGlobalSalesCalculator
from destination_based_sales.utils import UK_CARIBBEAN_ISLANDS, online_path_to_geo_file, online_path_to_TH_list, \
    online_path_to_GNI_data, online_path_to_CONS_data


########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_GNI_data = os.path.join(path_to_dir, 'data', 'gross_national_income.csv')
# path_to_UNCTAD_consumption_exp = os.path.join(path_to_dir, 'data', 'us_gdpcomponent_34577843893623.csv')
path_to_UNCTAD_consumption_exp = os.path.join(path_to_dir, 'data', 'us_gdpcomponent_98866982281181.csv')

path_to_tax_haven_list = os.path.join(path_to_dir, 'data', 'tax_havens.csv')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


########################################################################################################################
# --- Content

class USAnalysesProvider:

    def __init__(
        self,
        year: int,
        US_merchandise_exports_source: str,
        US_services_exports_source: str,
        non_US_merchandise_exports_source: str,
        non_US_services_exports_source: str,
        winsorize_export_percs: bool,
        US_winsorizing_threshold: float = 0.5,
        non_US_winsorizing_threshold: float = 0.5,
        service_flows_to_exclude: Optional[list] = None,
        macro_indicator: str = 'CONS',
        load_data_online: bool = False
    ):
        """Encapsulates the logic behind the analysis of US multinational companies' non-adjusted and adjusted sales.

        :param year: year to consider for the analysis
        :type year: int
        :param US_merchandise_exports_source: data source for the US exports of goods
        :type US_merchandise_exports_source: str
        :param US_services_exports_source: data source for the US exports of services
        :type US_services_exports_source: str
        :param non_US_merchandise_exports_source: data source for the non-US exports of goods
        :type non_US_merchandise_exports_source: str
        :param non_US_services_exports_source: data source for the non-US exports of services
        :type non_US_services_exports_source: str
        :param winsorize_export_percs: whether to winsorize small export percentages
        :type winsorize_export_percs: bool
        :param US_winsorizing_threshold: lower-bound winsorizing threshold for US exports in %, defaults to 0.5
        :type US_winsorizing_threshold: float, optional
        :param non_US_winsorizing_threshold: lower-bound winsorizing threshold for non-US exports in %, defaults to 0.5
        :type non_US_winsorizing_threshold: float, optional
        :param service_flows_to_exclude: types of service exports to exclude from trade statistics, defaults to None
        :type service_flows_to_exclude: list, optional
        :param macro_indicator: macro indicator with which to compare the distribution of sales, defaults to "CONS"
        :type macro_indicator: str, optional
        :param load_data_online: whether to load the data online (True) or locally (False), defaults to False
        :type load_data_online: bool, optional

        :raises Exception: if macro_indicator neither equal to "CONS", nor to "GNI"

        :rtype: destination_based_sales.analyses_provider.USAnalysesProvider
        :return: object of the class USAnalysesProvider, allowing to analyse US multinationals' revenue variables
        """
        # Saving most arguments as attributes (used below in the code)
        self.year = year

        self.US_merchandise_exports_source = US_merchandise_exports_source
        self.US_services_exports_source = US_services_exports_source
        self.non_US_merchandise_exports_source = non_US_merchandise_exports_source
        self.non_US_services_exports_source = non_US_services_exports_source

        self.winsorize_export_percs = winsorize_export_percs
        self.US_winsorizing_threshold = US_winsorizing_threshold
        self.non_US_winsorizing_threshold = non_US_winsorizing_threshold

        self.service_flows_to_exclude = service_flows_to_exclude

        # Depending on whether we load the data from online sources, paths differ
        self.load_data_online = load_data_online

        if not load_data_online:
            self.path_to_geographies = path_to_geographies
            self.path_to_tax_haven_list = path_to_tax_haven_list

            # Loading the relevant macroeconomic indicator
            self.path_to_GNI_data = path_to_GNI_data
            self.path_to_UNCTAD_data = path_to_UNCTAD_consumption_exp

        else:
            self.path_to_geographies = online_path_to_geo_file
            self.path_to_tax_haven_list = online_path_to_TH_list

            # Loading the relevant macroeconomic indicator
            self.path_to_GNI_data = online_path_to_GNI_data
            self.path_to_UNCTAD_data = online_path_to_CONS_data

        # Loading the indicator to which we compare US firms' revenues depending on the macro_indicator argument
        if macro_indicator == 'GNI':
            self.macro_indicator = self.get_GNI_data()
            self.macro_indicator_prefix = 'GNI'
            self.macro_indicator_name = 'Gross National Income'

        elif macro_indicator == 'CONS':
            self.macro_indicator = self.get_consumption_expenditure_data()
            self.macro_indicator_prefix = 'CONS'
            self.macro_indicator_name = 'consumption expenditures'

        else:
            raise Exception(
                'Macroeconomic indicator can either be Gross National Income (pass "macro_indicator=GNI" as argument) '
                + 'or the UNCTAD consumption expenditure indicator (pass "macro_indicator=CONS" as argument).'
            )

        # Loading the list of tax havens
        tax_havens = pd.read_csv(self.path_to_tax_haven_list)
        self.tax_haven_country_codes = list(tax_havens['CODE'].unique()) + ['UKI']

        # Loading unadjusted revenue variables
        irs_preprocessor = IRSDataPreprocessor(year=year, load_data_online=load_data_online)
        self.irs = irs_preprocessor.load_final_data()

        # Loading adjusted revenue variables and trade statistics
        calculator = USSalesCalculator(
            year=self.year,
            US_only=True,
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
        self.sales_mapping = calculator.get_final_sales_mapping()

    def get_GNI_data(self) -> pd.DataFrame:
        """Loads Gross National Income data.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider

        :rtype: pandas.DataFrame
        :return: DataFrame containing the relevant Gross National Income series

        .. note:: This method relies on a dedicated data file, loaded online or locally and prepared preliminarily.
        """
        gross_national_income = pd.read_csv(self.path_to_GNI_data, delimiter=';')

        for column in gross_national_income.columns[2:]:
            gross_national_income[column] = gross_national_income[column].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            gross_national_income[column] = gross_national_income[column].astype(float)

        ser = gross_national_income[gross_national_income['COUNTRY_CODE'].isin(UK_CARIBBEAN_ISLANDS)].sum()
        ser['COUNTRY_NAME'] = 'UK Caribbean Islands'
        ser['COUNTRY_CODE'] = 'UKI'

        temp = ser.to_frame().T.copy()
        gross_national_income = pd.concat([gross_national_income, temp], axis=0).reset_index(drop=True)

        return gross_national_income.copy()

    def get_consumption_expenditure_data(self) -> pd.DataFrame:
        """Loading data on consumption expenditures.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider

        :rtype: pandas.DataFrame
        :return: DataFrame containing the relevant consumption expenditures series

        .. note:: This method relies on UNCTAD data, referred to in the OECD's draft rules for Pillar One Amount A.
        """
        # Reading the data file
        df = pd.read_csv(self.path_to_UNCTAD_data, encoding='latin')

        # Basic cleaning
        df = df.reset_index()
        df.columns = df.iloc[0]
        df = df.iloc[2:].copy()

        df = df.rename(
            columns={
                np.nan: 'COUNTRY_NAME',
                'YEAR': 'ITEM'
            }
        ).reset_index(drop=True)

        # Focusing on final consumption expenditures
        df = df[df['ITEM'].map(lambda x: x.strip()) == 'Final consumption expenditure'].copy()

        # Removing the rows with "_" for all the relevant years
        list_of_years = ['2016', '2017', '2018', '2019', '2020']
        df = df[df[list_of_years].sum(axis=1) != '_' * len(list_of_years)].copy()

        # Converting the columns to floats
        for col in list_of_years:
            df[col] = df[col].astype(float)

        df = df.drop(columns='ITEM')

        # Adding country and continent codes
        df['COUNTRY_NAME'] = df['COUNTRY_NAME'].map(lambda x: x.strip())

        df['COUNTRY_NAME'] = df['COUNTRY_NAME'].map(
            lambda country_name: {'France': 'France incl. Monaco'}.get(country_name, country_name)
        )
        df['COUNTRY_NAME'] = df['COUNTRY_NAME'].map(
            lambda country_name: {'France, metropolitan': 'France'}.get(country_name, country_name)
        )

        geographies = pd.read_csv(self.path_to_geographies)

        df = df.merge(
            geographies,
            how='left',
            left_on='COUNTRY_NAME', right_on='NAME'
        )

        df['CONTINENT_CODE'] = df['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        df['CONTINENT_CODE'] = df['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        df = df[['COUNTRY_NAME', 'CODE', 'CONTINENT_CODE', str(self.year)]].copy()

        # Adding UK Caribbean Islands
        temp = df[df['CODE'].isin(UK_CARIBBEAN_ISLANDS)][str(self.year)].sum()

        temp_1 = {
            'COUNTRY_NAME': 'United Kingdom Islands, Caribbean',
            'CODE': 'UKI',
            'CONTINENT_CODE': 'AMR',
            str(self.year): temp
        }

        df = df.append(temp_1, ignore_index=True)

        # Final step - Renaming the columns
        df = df.rename(
            columns={
                'CODE': 'COUNTRY_CODE',
                str(self.year): f'CONS_{str(self.year)}'
            }
        )

        return df.dropna().copy()

    def get_table_1(
        self,
        formatted: bool = True,
        sales_type: str = 'unrelated'
    ) -> pd.DataFrame:
        """Builds Table 1 that presents the split of domestic vs. foreign sales (also split by continent) for each year.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param formatted: whether to format floats into strings in the DataFrame, defaults to True
        :type formatted: bool, optional
        :param sales_type: revenue variable to consider, defaults to "unrelated"
        :type sales_type: str, optional

        :raises Exception: if sales_type equal neither to "unrelated", to "related", nor to "total"

        :rtype: pandas.DataFrame
        :return: Table 1, presenting the split of domestic vs. foreign sales (also split by continent) for each year
        """
        # Deducing the relevant variable from the sales_type argument
        if sales_type not in ['unrelated', 'related', 'total']:
            raise Exception(
                'The type of sales to consider for building Table 1 can only be: "related" (for related-party revenues)'
                + ', "unrelated" (for unrelated-party revenues) or "total" (for total revenues).'
            )

        sales_type_correspondence = {
            'unrelated': 'UNRELATED_PARTY_REVENUES',
            'related': 'RELATED_PARTY_REVENUES',
            'total': 'TOTAL_REVENUES'
        }

        column_name = sales_type_correspondence[sales_type.lower()]

        # Dictionaries storing the total domestic and foreign sales respectively
        us_totals = {}
        foreign_totals = {}

        # Getting 2016 figures
        preprocessor = IRSDataPreprocessor(year=2016, load_data_online=self.load_data_online)
        df = preprocessor.load_final_data()

        us_totals[2016] = df[df['CODE'] == 'USA'][column_name].iloc[0]

        df = df[df['CODE'] != 'USA'].copy()

        foreign_totals[2016] = df[column_name].sum()

        # Continent-specific sub-totals
        df = df.groupby('CONTINENT_NAME').sum()[[column_name]]
        df[column_name] /= (foreign_totals[2016] / 100)
        df.rename(columns={column_name: 2016}, inplace=True)

        # Repeating the process for the other relevant years
        for year in [2017, 2018, 2019]:

            preprocessor = IRSDataPreprocessor(year=year, load_data_online=self.load_data_online)
            df_temp = preprocessor.load_final_data()

            us_totals[year] = df_temp[df_temp['CODE'] == 'USA'][column_name].iloc[0]

            df_temp = df_temp[df_temp['CODE'] != 'USA'].copy()

            foreign_totals[year] = df_temp[column_name].sum()

            df_temp = df_temp.groupby('CONTINENT_NAME').sum()[[column_name]]
            df_temp[column_name] /= (foreign_totals[year] / 100)
            df_temp.rename(columns={column_name: year}, inplace=True)

            df = pd.concat([df, df_temp], axis=1)

        # Reformatting the DataFrame
        dict_df = df.to_dict()

        indices = ['Sales to the US (billion USD)', 'Sales abroad (billion USD)']

        for year in [2016, 2017, 2018, 2019]:
            dict_df[year][indices[0]] = us_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        # Sorting values, giving the priority to the latest year
        df.sort_values(by=[2019, 2018, 2017, 2016], ascending=False, inplace=True)

        # Formatting the floats into strings if relevant
        if formatted:

            for year in [2016, 2017, 2018, 2019]:
                df[year] = df[year].map('{:,.1f}'.format)

        # Final formatting step
        df.index = indices + [f'Of which {continent} (%)' for continent in df.index[2:]]

        return df.copy()

    def get_intermediary_dataframe_1(
        self,
        include_macro_indicator: bool,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Builds the first intermediary DataFrame, used for Table 2.a. and 2.b. and for Figure 1.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param include_macro_indicator: whether to include the relevant macro indicator in the table
        :type include_macro_indicator: bool
        :param verbose: whether to print optional results (number of countries disregarded, etc.), defaults to False
        :type verbose: bool, optional

        :rtype: pandas.DataFrame
        :return: first intermediary DataFrame, used for Table 2.a. and 2.b. and for Figure 1.
        """
        # Starting from the unadjusted revenue variables in the IRS' data
        irs = self.irs.copy()

        columns_of_interest = ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']

        # Including the macro indicator if relevant
        if include_macro_indicator:
            # Merging unadusted revenue variables and the macro indicator series
            merged_df = irs.merge(
                self.macro_indicator[['COUNTRY_CODE', f'{self.macro_indicator_prefix}_{self.year}']].copy(),
                how='left',
                left_on='CODE', right_on='COUNTRY_CODE'
            )

            # Printing various intermediary results
            if verbose:
                # Number of countries in the unadjusted revenue variables for which we lack the macro indicator
                print(
                    merged_df[f'{self.macro_indicator_prefix}_{self.year}'].isnull().sum(),
                    'foreign partner countries are eliminated because we lack the macroeconomic indicator for them.'
                )

                # For what share of the foreign unrelated-party revenues of US multinationals do they account?
                temp = merged_df[merged_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()
                temp = (
                    temp[
                        temp[f'{self.macro_indicator_prefix}_{self.year}'].isnull()
                    ]['UNRELATED_PARTY_REVENUES'].sum()
                    / temp['UNRELATED_PARTY_REVENUES'].sum() * 100
                )

                print(
                    'They represent',
                    round(temp, 2),
                    'of the foreign unrelated-party revenues in the table.'
                )

                # For what share of the final consumption expenditures do tax havens account (excluding the US)?
                temp = merged_df[merged_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()
                temp = (
                    temp[
                        temp['CODE'].isin(self.tax_haven_country_codes)
                    ][f'{self.macro_indicator_prefix}_{self.year}'].sum()
                    / temp[f'{self.macro_indicator_prefix}_{self.year}'].sum() * 100
                )

                print(
                    'Tax havens represent',
                    round(temp, 2),
                    'of the final consumption expenditures in the table.'
                )

            # Eliminating countries for which we lack a macro indicator
            merged_df = merged_df[~merged_df[f'{self.macro_indicator_prefix}_{self.year}'].isnull()].copy()

            columns_of_interest.append(f'{self.macro_indicator_prefix}_{self.year}')

        else:
            merged_df = irs.copy()

        # Excluding the US
        merged_df = merged_df[merged_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

        # Deducing shares from the absolute amounts
        new_columns = []

        for column in columns_of_interest:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            merged_df[new_column] = merged_df[column] / merged_df[column].sum() * 100

        return merged_df.copy()

    def get_table_2_a(
        self,
        formatted: bool = True
    ) -> pd.DataFrame:
        """Builds Table 2.a., that ranks the 20 largest partners of the US based on unadjusted unrelated-party revenues.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param formatted: whether to format the floats intro strings in the DataFrame, defaults to True
        :type formatted: bool, optional

        :rtype: pandas.DataFrame
        :return: Table 2.a., that ranks the 20 largest partners of the US based on unadjusted unrelated-party revenues
        """
        # Loading the relevant intermediary DataFrame without the macro indicator
        merged_df = self.get_intermediary_dataframe_1(include_macro_indicator=False)

        # Restricting the table to the relevant columns and ranking based on unrelated-party revenues
        output = merged_df[
            ['AFFILIATE_COUNTRY_NAME', 'UNRELATED_PARTY_REVENUES', 'SHARE_OF_UNRELATED_PARTY_REVENUES']
        ].sort_values(
            by='UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        # Moving from USD to billion USD
        output['UNRELATED_PARTY_REVENUES'] /= 10**9

        # Formatting floats into strings if relevant
        if formatted:
            for column in ['UNRELATED_PARTY_REVENUES', 'SHARE_OF_UNRELATED_PARTY_REVENUES']:
                output[column] = output[column].map('{:.1f}'.format)

        # (Almost) final step - Renaming the columns
        output.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME': 'Partner jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (USD billion)',
                'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
            },
            inplace=True
        )

        output.reset_index(drop=True, inplace=True)

        return output.copy()

    def get_table_2_b(
        self,
        formatted: bool = True,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Builds Table 2.b., that ranks the 20 largest partners of the US (unadjusted) with the macro indicator.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param formatted: whether to format the floats intro strings in the DataFrame, defaults to True
        :type formatted: bool, optional
        :param verbose: whether to print optional results, defaults to False
        :type verbose: bool, optional

        :rtype: pandas.DataFrame
        :return: Table 2.b., that ranks the 20 largest partners of the US (unadjusted) with the macro indicator
        """
        # Loading the relevant intermediary DataFrame with the macro indicator (and verbose if relevant)
        merged_df = self.get_intermediary_dataframe_1(include_macro_indicator=True, verbose=verbose)

        # Restricting the table to the relevant columns and ranking based on unrelated-party revenues
        output = merged_df[
            [
                'AFFILIATE_COUNTRY_NAME', 'SHARE_OF_UNRELATED_PARTY_REVENUES',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ]
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        # Formatting floats into strings if relevant
        if formatted:
            for column in ['SHARE_OF_UNRELATED_PARTY_REVENUES', f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}']:
                output[column] = output[column].map('{:.1f}'.format)

        # (Almost) final step - Renaming the columns
        output.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME': 'Partner jurisdiction',
                'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}': f'Share of {self.macro_indicator_name} (%)'
            },
            inplace=True
        )

        output.reset_index(drop=True, inplace=True)

        return output.copy()

    def plot_figure_1(
        self,
        kind: str,
        save_PNG: bool = False,
        path_to_folder: Optional[str] = None
    ):
        """Plots Figure 1, showing the relationship between the distributions of revenues and of the macro indicator.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param kind: type of graph to show (either a regplot, a scatterplot or an interactive chart)
        :type kind: str
        :param save_PNG: whether to save the graph as a PNG file, defaults to False
        :type save_PNG: bool, optional
        :param path_to_folder: path to the destination folder where to store the PNG file, defaults to None
        :type path_to_folder: str, optional

        :raises Exception: if kind equal neither to "regplot", to "scatter", nor to "interactive"
        :raises Exception: if save_PNG is True and path_to_folder is equal to None
        :raises Exception: if save_PNG is True and kind is differs from "regplot" (only one type of graph can be saved)

        :rtype: None (plt.show())
        :return: None (plt.show())
        """
        # Checking the value of the kind parameter, determining what type of graph to plot
        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        # Checking that we have a path to the destination folder if we need to save the graph as a PNG file
        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        # We load the intermediary DataFrame with the macro indicator
        merged_df = self.get_intermediary_dataframe_1(include_macro_indicator=True)

        # If we want to plot a regplot
        if kind == 'regplot':
            plot_df = merged_df.dropna().copy()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            # Adding a new column that determines the color of the dots (NAFTA members, tax havens, others)
            plot_df['Category'] = (
                plot_df['CODE'].isin(self.tax_haven_country_codes) * 1
                + plot_df['CODE'].isin(['CAN', 'MEX']) * 2
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven', 2: 'NAFTA member'})

            # Converting the shares of the macro indicator and the shares of unrelated-party revenues into floats
            plot_df[
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ] = plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'].astype(float)
            plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'] = plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'].astype(float)

            # Computing the correlation coefficient between the two series
            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'],
                plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )[1, 0]

            comment = (
                f'Correlation between unrelated-party revenues and {self.macro_indicator_name} '
                + f'in {self.year}: {round(correlation, 2)}'
            )

            # Matplotlib parameters determining the look of the graph
            plt.rcParams.update(
                {
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18,
                    'legend.fontsize': 18
                }
            )

            # Instantiating the figure
            plt.figure(figsize=(17, 10))

            # Changing column names
            col_name_init = f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            col_name_new = f'Share of total {self.year} {self.macro_indicator_name} (%)'
            plot_df.rename(
                columns={
                    col_name_init: col_name_new,
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
                },
                inplace=True
            )

            # Plotting the regression line
            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                ci=None,
                truncate=False
            )

            # Adding the scattered dots
            sns.scatterplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=100
            )

            if self.year == 2017:
                extract = plot_df.sort_values(
                    by='Share of foreign unrelated-party revenues (%)',
                    ascending=False
                ).head(10)

                for _, country in extract.iterrows():
                    if country['CODE'] == 'BRA':
                        x = country[col_name_new] - 0.3
                        y = country['Share of foreign unrelated-party revenues (%)'] + 0.4
                    else:
                        x = country[col_name_new] + 0.2
                        y = country['Share of foreign unrelated-party revenues (%)']

                    plt.text(
                        x=x,
                        y=y,
                        s=country['CODE'],
                        size=16,
                        bbox=dict(facecolor='gray', alpha=0.7)
                    )

                plt.xlim(0, plot_df[col_name_new].max() + 1)
                plt.ylim(-0.3, plot_df['Share of foreign unrelated-party revenues (%)'].max() + 1)

            # Adding the title with the correlation coefficient
            plt.title(comment)

            # Saving as a PNG file if relevant
            if save_PNG:
                plt.savefig(
                    os.path.join(
                        path_to_folder,
                        f'figure_1_{self.year}_US_only{"_GNI" if self.macro_indicator_prefix == "GNI" else ""}.png'
                    ),
                    bbox_inches='tight'
                )

            plt.show()

        # Other types of graphs
        else:
            merged_df['IS_TAX_HAVEN'] = merged_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes)

            plot_df = merged_df.dropna().copy()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            # Plotting a simple scatterplot
            if kind == 'scatter':
                plt.figure(figsize=(12, 7))

                sns.scatterplot(
                    x=f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    hue='IS_TAX_HAVEN',
                    data=plot_df
                )

                plt.show()

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

            # Plotting an interactive scatterplot with Plotly Express
            else:
                fig = px.scatter(
                    x=f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    color='IS_TAX_HAVEN',
                    color_discrete_sequence=['#636EFA', '#EF553B'],
                    data_frame=plot_df,
                    hover_name='AFFILIATE_COUNTRY_NAME'
                )

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

                fig.show()

    def get_table_4_intermediary(self) -> pd.DataFrame:
        """Builds the intermediary DataFrame used for Table 4 (continental split of the adjusted revenues).

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider

        :rtype: pandas.DataFrame
        :return: intermediary DataFrame used for Table 4, the latter showing the continental split of adjusted revenues
        """
        # Basic manipulation with the new sales mapping
        sales_mapping = self.sales_mapping.copy()

        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        # Cleaning geographies
        geographies = pd.read_csv(self.path_to_geographies)
        geographies = geographies[['CODE', 'CONTINENT_NAME']].drop_duplicates()

        geographies['CONTINENT_NAME'] = geographies['CONTINENT_NAME'].map(
            lambda x: 'Asia-Pacific' if x in ['Asia', 'Oceania'] or x is None else x
        )
        geographies['CONTINENT_NAME'] = geographies['CONTINENT_NAME'].map(
            lambda x: 'America' if x in ['South America', 'North America'] else x
        )

        # Merging the two DataFrames
        sales_mapping = sales_mapping.merge(
            geographies,
            how='left',
            left_on='OTHER_COUNTRY_CODE', right_on='CODE'
        )

        sales_mapping.drop(columns=['CODE'], inplace=True)

        continent_names_to_impute = {
            'OASIAOCN': 'Asia-Pacific',
            'UKI': 'America'
        }

        sales_mapping['CONTINENT_NAME'] = sales_mapping.apply(
            lambda row: continent_names_to_impute.get(row['OTHER_COUNTRY_CODE'], row['CONTINENT_NAME']),
            axis=1
        )

        return sales_mapping.copy()

    def get_table_4(
        self,
        sales_type: str = 'unrelated',
        formatted: bool = True
    ):
        """Builds Table 4 that presents the split of domestic vs. foreign adjusted sales (also split by continent).

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param sales_type: revenue variable to consider, defaults to "unrelated"
        :type sales_type: str, optional
        :param formatted: whether to format floats into strings in the DataFrame, defaults to True
        :type formatted: bool, optional

        :raises Exception: if sales_type equal neither to "unrelated", to "related", nor to "total"

        :rtype: pandas.DataFrame
        :return: Table 4, presenting the split of domestic vs. foreign adjusted sales (also split by continent)
        """
        # Instantiating the dictionaries that will store the total domestic and foreign sales
        us_totals = {}
        foreign_totals = {}

        # Deducing the relevant variable from the sales_type argument
        if sales_type not in ['unrelated', 'related', 'total']:
            raise Exception(
                'The type of sales to consider for building Table 4 can only be: "related" (for related-party revenues)'
                + ', "unrelated" (for unrelated-party revenues) or "total" (for total revenues).'
            )

        sales_type_correspondence = {
            'unrelated': 'UNRELATED_PARTY_REVENUES',
            'related': 'RELATED_PARTY_REVENUES',
            'total': 'TOTAL_REVENUES'
        }

        column_name = sales_type_correspondence[sales_type.lower()]

        # Manipulations for the year 2016
        analyser = USAnalysesProvider(
            year=2016,
            US_merchandise_exports_source=self.US_merchandise_exports_source,
            US_services_exports_source=self.US_services_exports_source,
            non_US_merchandise_exports_source=self.non_US_merchandise_exports_source,
            non_US_services_exports_source=self.non_US_services_exports_source,
            winsorize_export_percs=self.winsorize_export_percs,
            US_winsorizing_threshold=self.US_winsorizing_threshold,
            non_US_winsorizing_threshold=self.non_US_winsorizing_threshold,
            service_flows_to_exclude=self.service_flows_to_exclude
        )
        df = analyser.get_table_4_intermediary()

        us_totals[2016] = df[df['OTHER_COUNTRY_CODE'] == 'USA'][column_name].iloc[0]

        df = df[df['OTHER_COUNTRY_CODE'] != 'USA'].copy()

        foreign_totals[2016] = df[column_name].sum()

        df = df.groupby('CONTINENT_NAME').sum()[[column_name]]
        df[column_name] /= (foreign_totals[2016] / 100)
        df.rename(columns={column_name: 2016}, inplace=True)

        # Replicating these operations for the other years of interest
        for year in [2017, 2018, 2019]:
            analyser = USAnalysesProvider(
                year=year,
                US_merchandise_exports_source=self.US_merchandise_exports_source,
                US_services_exports_source=self.US_services_exports_source,
                non_US_merchandise_exports_source=self.non_US_merchandise_exports_source,
                non_US_services_exports_source=self.non_US_services_exports_source,
                winsorize_export_percs=self.winsorize_export_percs,
                US_winsorizing_threshold=self.US_winsorizing_threshold,
                non_US_winsorizing_threshold=self.non_US_winsorizing_threshold,
                service_flows_to_exclude=self.service_flows_to_exclude
            )
            df_temp = analyser.get_table_4_intermediary()

            us_totals[year] = df_temp[df_temp['OTHER_COUNTRY_CODE'] == 'USA'][column_name].iloc[0]

            df_temp = df_temp[df_temp['OTHER_COUNTRY_CODE'] != 'USA'].copy()

            foreign_totals[year] = df_temp[column_name].sum()

            df_temp = df_temp.groupby('CONTINENT_NAME').sum()[[column_name]]
            df_temp[column_name] /= (foreign_totals[year] / 100)
            df_temp.rename(columns={column_name: year}, inplace=True)

            df = pd.concat([df, df_temp], axis=1)

        # Formatting the DataFrame
        dict_df = df.to_dict()

        indices = ['Sales to the US (billion USD)', 'Sales abroad (billion USD)']

        # Moving from USD to billion USD
        for year in [2016, 2017, 2018, 2019]:
            dict_df[year][indices[0]] = us_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        # Sorting values based prioritarily on the latest year
        df.sort_values(by=[2019, 2018, 2017, 2016], ascending=False, inplace=True)

        # If relevant, formatting floats into strings with the proper number of decimals
        if formatted:
            for year in [2016, 2017, 2018, 2019]:
                df[year] = df[year].map('{:,.1f}'.format)

        # Final formatting step - Adding the missing row names
        df.index = indices + [f'Of which {continent} (%)' for continent in df.index[2:]]

        return df.copy()

    def get_intermediary_dataframe_2(
        self,
        include_macro_indicator: bool
    ) -> pd.DataFrame:
        """Builds the second intermediary DataFrame (based on adjusted sales), used for Table 5 and for Figure 2.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param include_macro_indicator: whether to include the relevant macro indicator in the table
        :type include_macro_indicator: bool

        :rtype: pandas.DataFrame
        :return: second intermediary DataFrame (based on adjusted sales), used for Table 5 and for Figure 2
        """
        # Starting from the adjusted revenue variables
        sales_mapping = self.sales_mapping.copy()

        # Aggregating the sales over final destinations
        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        # Including the relevant macro indicator depending on the include_macro_indicator argument
        if include_macro_indicator:
            # Merging the macro indicator series over the adjusted sales mapping
            sales_mapping = sales_mapping.merge(
                self.macro_indicator[
                    ['COUNTRY_CODE', 'COUNTRY_NAME', f'{self.macro_indicator_prefix}_{self.year}']
                ].copy(),
                how='left',
                left_on='OTHER_COUNTRY_CODE', right_on='COUNTRY_CODE'
            )

            sales_mapping.drop(columns='OTHER_COUNTRY_CODE', inplace=True)

            # Cleaning the macro indicator series and converting it into floats
            sales_mapping[
                f'{self.macro_indicator_prefix}_{self.year}'
            ] = sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            sales_mapping[
                f'{self.macro_indicator_prefix}_{self.year}'
            ] = sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].astype(float)

            # Removing countries for which we lack the macro indicator
            sales_mapping = sales_mapping[~sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].isnull()].copy()

        # Excluding the US from the destinations
        sales_mapping = sales_mapping[sales_mapping['COUNTRY_CODE'] != 'USA'].copy()

        # Moving from absolute amounts to shares (of the macro indicator and of sales)
        new_columns = []

        for column in [
            'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES',
            'TOTAL_REVENUES', f'{self.macro_indicator_prefix}_{self.year}'
        ]:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            sales_mapping[new_column] = sales_mapping[column] / sales_mapping[column].sum() * 100

        return sales_mapping.copy()

    def get_table_5(
        self,
        formatted: bool = True
    ) -> pd.DataFrame:
        """Builds Table 5, that ranks the 20 largest partners of the US based on adjusted unrelated-party revenues.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param formatted: whether to format the floats intro strings in the DataFrame, defaults to True
        :type formatted: bool, optional

        :rtype: pandas.DataFrame
        :return: Table 5, that ranks the 20 largest partners of the US based on adjusted unrelated-party revenues
        """
        # Loading the second intermediary DataFrame with the macro indicator
        merged_df = self.get_intermediary_dataframe_2(include_macro_indicator=True)

        # Restricting the DataFrame to the relevant columns and sorting values based on unrelated-party revenues
        output = merged_df[
            [
                'COUNTRY_NAME',
                'UNRELATED_PARTY_REVENUES',
                'SHARE_OF_UNRELATED_PARTY_REVENUES',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ]
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        output.reset_index(drop=True, inplace=True)

        # Moving from USD to billion USD
        output['UNRELATED_PARTY_REVENUES'] /= 10**9

        # If relevant, formatting floats into strings with the proper number of decimals
        if formatted:
            for column in output.columns[1:]:
                output[column] = output[column].map('{:.1f}'.format)

        # Final formatting step - Renaming columns
        output.rename(
            columns={
                'COUNTRY_NAME': 'Partner jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (USD billion)',
                'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of unrelated-party revenues (%)',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}': f'Share of {self.macro_indicator_name} (%)'
            },
            inplace=True
        )

        return output.copy()

    def plot_figure_2(
        self,
        kind: str,
        save_PNG: bool = False,
        path_to_folder: Optional[str] = None
    ):
        """Plots Figure 2, showing the relationship between adjusted unrelated-party revenues and the macro indicator.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param kind: type of graph to show (either a regplot, a scatterplot or an interactive chart)
        :type kind: str
        :param save_PNG: whether to save the graph as a PNG file, defaults to False
        :type save_PNG: bool, optional
        :param path_to_folder: path to the destination folder where to store the PNG file, defaults to None
        :type path_to_folder: str, optional

        :raises Exception: if kind equal neither to "regplot", to "scatter", nor to "interactive"
        :raises Exception: if save_PNG is True and path_to_folder is equal to None
        :raises Exception: if save_PNG is True and kind is differs from "regplot" (only one type of graph can be saved)

        :rtype: None (plt.show())
        :return: None (plt.show())
        """
        # Checking the value of the kind parameter, determining what type of graph to plot
        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        # Checking that we have a path to the destination folder if we need to save the graph as a PNG file
        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        # We load the second intermediary DataFrame with the macro indicator
        merged_df = self.get_intermediary_dataframe_2(include_macro_indicator=True)

        # If we want to plot a regplot
        if kind == 'regplot':

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['COUNTRY_NAME'] != 'United States'].copy()

            # Adding a new column that determines the color of the dots (NAFTA members, tax havens, others)
            plot_df['Category'] = (
                plot_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1
                + plot_df['COUNTRY_CODE'].isin(['CAN', 'MEX']) * 2
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven', 2: 'NAFTA member'})

            # Computing the correlation coefficient between the two series
            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'].astype(float),
                plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'].astype(float)
            )[1, 0]

            comment = (
                'Correlation between unrelated-party revenues and '
                + f'{self.macro_indicator_name} in {self.year}: {round(correlation, 2)}'
            )

            # Matplotlib parameters determining the look of the graph
            plt.rcParams.update(
                {
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18,
                    'legend.fontsize': 18
                }
            )

            # Instantiating the figure
            plt.figure(figsize=(17, 10))

            # Changing column names
            col_name_init = f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            col_name_new = f'Share of total {self.year} {self.macro_indicator_name} (%)'

            plot_df.rename(
                columns={
                    col_name_init: col_name_new,
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
                },
                inplace=True
            )

            # Plotting the regression line
            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                ci=None
            )

            # Adding scattered dots
            sns.scatterplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=100
            )

            if self.year == 2017:
                extract = plot_df.sort_values(
                    by='Share of foreign unrelated-party revenues (%)',
                    ascending=False
                ).head(10)
                for _, country in extract.iterrows():
                    if country['COUNTRY_CODE'] in ['SGP', 'NLD']:
                        x = country[col_name_new] - 0.1
                        y = country['Share of foreign unrelated-party revenues (%)'] + 0.35
                    else:
                        x = country[col_name_new] + 0.2
                        y = country['Share of foreign unrelated-party revenues (%)']

                    plt.text(
                        x=x,
                        y=y,
                        s=country['COUNTRY_CODE'],
                        size=16,
                        bbox=dict(facecolor='gray', alpha=0.7)
                    )

                plt.xlim(0, plot_df[col_name_new].max() + 1)
                plt.ylim(-0.3, plot_df['Share of foreign unrelated-party revenues (%)'].max() + 1)

            # Adding the title with the correlation coefficient
            plt.title(comment)

            # Saving as a PNG file if relevant
            if save_PNG:
                temp_bool = len(self.service_flows_to_exclude) > 0

                figure_name = f'figure_2_{self.year}_US_only{"_GNI" if self.macro_indicator_prefix == "GNI" else ""}'
                figure_name += f'{"_excl" if temp_bool else ""}.png'

                plt.savefig(
                    os.path.join(
                        path_to_folder,
                        figure_name
                    ),
                    bbox_inches='tight'
                )

            plt.show()

        # Other types of graphs
        else:
            merged_df['IS_TAX_HAVEN'] = merged_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['COUNTRY_NAME'] != 'United States'].copy()

            # Plotting a simple scatterplot
            if kind == 'scatter':

                plt.figure(figsize=(12, 7))

                sns.scatterplot(
                    x=f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    hue='IS_TAX_HAVEN',
                    data=plot_df
                )

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

                plt.show()

            # Plotting an interactive scatterplot with Plotly Express
            else:
                colors = plot_df['IS_TAX_HAVEN'].map(
                    lambda x: 'blue' if x == 0 else 'red'
                )

                fig = px.scatter(
                    x=f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    color=colors,
                    data_frame=plot_df,
                    hover_name='COUNTRY_NAME'
                )

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

                fig.show()

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Builds a DataFrame with each country's adjusted and unadjusted unrelated-party revenues for comparison.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider

        :rtype: pandas.DataFrame
        :return: DataFrame with each country's adjusted and unadjusted unrelated-party revenues
        """
        # Unadjusted sales mapping
        irs = self.irs.copy()

        # Adjusted sales mapping
        sales_mapping = self.sales_mapping.copy()
        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        # Merging the two unrelated-party revenues series
        irs = irs[['AFFILIATE_COUNTRY_NAME', 'CODE', 'UNRELATED_PARTY_REVENUES']].copy()

        merged_df = irs.merge(
            sales_mapping[['OTHER_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES']],
            how='inner',
            left_on='CODE', right_on='OTHER_COUNTRY_CODE'
        )

        merged_df.drop(columns=['OTHER_COUNTRY_CODE'], inplace=True)

        # Final step - Renaming columns
        merged_df.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME': 'COUNTRY_NAME',
                'CODE': 'COUNTRY_CODE',
                'UNRELATED_PARTY_REVENUES_x': 'UPR_IRS',
                'UNRELATED_PARTY_REVENUES_y': 'UPR_ADJUSTED'
            },
            inplace=True
        )

        return merged_df.copy()

    def get_focus_on_tax_havens(self) -> pd.DataFrame:
        """Builds a DataFrame that allows to compare the unadjusted and adjusted revenues booked in tax havens.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider

        :rtype: pandas.DataFrame
        :return: DataFrame focused on the impact of the adjustment on tax havens
        """
        # We load the DataFrame allowing to compare each partner's non-adjusted and adjusted revenues
        merged_df = self.get_comparison_dataframe()

        # Restricting the set of partner countries to tax havens
        restricted_df = merged_df[
            merged_df['COUNTRY_CODE'].isin(
                self.tax_haven_country_codes
            )
        ].copy()

        # Ranking based on non-adjusted unrelated-party revenues (from largest to smallest)
        restricted_df.sort_values(by='UPR_IRS', ascending=False, inplace=True)

        restricted_df.reset_index(drop=True, inplace=True)

        # We add a row to the DataFrame that shows the total (non-adjusted and adjusted) revenues in tax havens
        dict_df = restricted_df.to_dict()

        dict_df[restricted_df.columns[0]][len(restricted_df)] = 'Total for tax havens'
        dict_df[restricted_df.columns[1]][len(restricted_df)] = '..'
        dict_df[restricted_df.columns[2]][len(restricted_df)] = restricted_df['UPR_IRS'].sum()
        dict_df[restricted_df.columns[3]][len(restricted_df)] = restricted_df['UPR_ADJUSTED'].sum()

        restricted_df = pd.DataFrame.from_dict(dict_df)

        # Moving from absolute amounts to shares
        restricted_df['SHARE_OF_UPR_IRS'] = restricted_df['UPR_IRS'] / merged_df['UPR_IRS'].sum() * 100
        restricted_df['SHARE_OF_UPR_ADJUSTED'] = restricted_df['UPR_ADJUSTED'] / merged_df['UPR_ADJUSTED'].sum() * 100

        # Expressing the absolute amounts in million USD, instead of plain USD
        for column in ['UPR_IRS', 'UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column] / 10**6
            # restricted_df[column] = restricted_df[column].map(round)

        # Rounding revenue shares with three decimals
        for column in ['SHARE_OF_UPR_IRS', 'SHARE_OF_UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column].map(
                lambda x: round(x, 3)
            )

        # (Almost) final step - Renaming columns
        restricted_df.rename(
            columns={
                'COUNTRY_NAME': 'Country name',
                'UPR_IRS': 'Unrelated-party revenues based on IRS ($m)',
                'UPR_ADJUSTED': 'Adjusted unrelated-party revenues ($m)',
                'SHARE_OF_UPR_IRS': 'Share of UPR based on IRS (%)',
                'SHARE_OF_UPR_ADJUSTED': 'Share of adjusted UPR (%)'
            },
            inplace=True
        )

        restricted_df.drop(columns=['COUNTRY_CODE'], inplace=True)

        return restricted_df.copy()

    def plot_focus_on_tax_havens(
        self,
        orient: str = 'horizontal',
        save_PNG: bool = False,
        path_to_folder: Optional[str] = None
    ):
        """Plots Figure 3, that shows the evolution of the unrelated-party revenues attributed to in-sample tax havens.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param orient: orientation of the bars in the graph, defaults to "horizontal"
        :type orient: str, optional
        :param save_PNG: whether to save the graph as a PNG file, defaults to False
        :type save_PNG: bool, optional
        :param path_to_folder: path to the destination folder where to store the PNG file, defaults to None
        :type path_to_folder: str, optional

        :raises Exception: if orient is specified and equal neither to "horizontal", nor to "vertical"
        :raises Exception: if save_PNG is True and path_to_folder is equal to None

        :rtype: None (plt.show())
        :return: None (plt.show())
        """
        # Checking that if one wants to save the graph as a PNG file, a path to the destination folder was specified
        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        # Setting some variables that will determine the appearance of the graph depending on the orientation chosen
        if orient == 'horizontal':
            figsize = (12, 12)
            ascending = False

        elif orient == 'vertical':
            figsize = (12, 8)
            ascending = True

        # orient must either be equal to "horizontal" (default value) or to "vertical"
        else:
            raise Exception('Orientation of the graph can only be "horizontal" or "vertical".')

        # We start from the DataFrame that tracks the evolution of tax havens
        df = self.get_focus_on_tax_havens()

        # We compute each tax haven's % change in unrelated-party revenues through the adjustment
        df['Change in unrelated-party revenues (%)'] = (df[df.columns[2]] / df[df.columns[1]] - 1) * 100

        # If, for some tax havens, the absolute % change is over 100%, they will not be shown on the graph
        # Instead, we print the name of the tax haven(s), its non-adjusted and adjusted UPR and the % change
        if (np.abs(df[df.columns[-1]]) >= 100).sum() > 0:
            for _, row in df[np.abs(df[df.columns[-1]]) >= 100].iterrows():
                print(
                    row['Country name'], '-', row['Unrelated-party revenues based on IRS ($m)'],
                    '-', row['Adjusted unrelated-party revenues ($m)'], '-', row[df.columns[-1]]
                )

        # Eliminating the tax haven(s) concerned
        df = df[np.abs(df[df.columns[-1]]) < 100].copy()

        # Sorting values based on the % change
        df_sorted = df.sort_values(
            by=df.columns[-1],
            ascending=ascending
        ).copy()

        # Settings for the aspect of the graph
        plt.rcParams.update(
            {
                'axes.titlesize': 20,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18
            }
        )

        # Instantiating the figure
        plt.figure(figsize=figsize)

        # Bars associated with a decrease are displayed in red; the others are displayed in blue
        y_pos = np.arange(len(df_sorted))
        colors = df_sorted[df_sorted.columns[-1]].map(
            lambda x: 'darkred' if x < 0 else 'darkblue'
        )

        # If the bars are oriented horizontally
        if orient == 'horizontal':
            plt.barh(
                y_pos,
                df_sorted[df_sorted.columns[-1]],
                color=colors
            )

            plt.yticks(
                ticks=y_pos,
                labels=df_sorted['Country name']
            )

            plt.xlabel(df_sorted.columns[-1])

            file_name_suffix = ''

        # If the bars are oriented vertically
        else:
            plt.bar(
                y_pos,
                df_sorted[df_sorted.columns[-1]],
                color=colors
            )

            # We shorten some country names
            df_sorted['Country name'] = df_sorted['Country name'].map(
                lambda x: 'UK Caribbean Islands' if x == 'United Kingdom Islands, Caribbean' else x
            )
            df_sorted['Country name'] = df_sorted['Country name'].map(
                lambda x: 'St. Vincent and the Gr.' if x == 'Saint Vincent and the Grenadines' else x
            )

            plt.xticks(
                ticks=y_pos,
                labels=df_sorted['Country name'],
                rotation=90
            )

            plt.ylabel(df_sorted.columns[-1])

            # To distinguish the PNG files associated with horizontal and vertical graphs, we add a specific suffix
            file_name_suffix = 'v'

        # Restricting the white spaces on the sides of the graph
        plt.tight_layout()

        # Saving the graph as a PNG file if relevant
        if save_PNG:
            temp_bool = len(self.service_flows_to_exclude) > 0

            plt.savefig(
                os.path.join(
                    path_to_folder,
                    f'focus_on_tax_havens_{self.year}_US_only{"_excl" if temp_bool else ""}_{file_name_suffix}.png'
                ),
                bbox_inches='tight'
            )

        plt.show()

    def get_table_6(
        self,
        country_code: str,
        formatted: bool = True
    ) -> pd.DataFrame:
        """Builds Table 6 for any given in-sample partner country, showing where its adjusted revenues come from.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param country_code: alpha-3 code of the country on which we want to focus
        :type country_code: str
        :param formatted: whether to format floats into strings with the right number of decimals, defaults to False
        :type formatted: bool, optional

        :rtype: pandas.DataFrame
        :return: table showing, for the country considered, where its adjusted revenues come from
        """
        # For some countries for which this table is especially relevant and useful, we change the code into a full name
        if country_code == 'BEL':
            country_name = 'Belgium'

        elif country_code == 'LBN':
            country_name = 'Lebanon'

        elif country_code == 'NLD':
            country_name = 'the Netherlands'

        else:
            country_name = country_code

        # We start from the adjusted sales mapping
        sales_mapping = self.sales_mapping.copy()

        # We restrict it to the destination / partner country of interest
        focus = sales_mapping[sales_mapping['OTHER_COUNTRY_CODE'] == country_code].copy()

        focus = focus.groupby(
            'AFFILIATE_COUNTRY_CODE'
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        # We also look into trade statistics and more specifically, at the exports to the chosen destination country
        trade_statistics = self.trade_statistics.copy()
        trade_statistics_extract = trade_statistics[trade_statistics['OTHER_COUNTRY_CODE'] == country_code].copy()
        trade_statistics_extract = trade_statistics_extract[
            ['AFFILIATE_COUNTRY_CODE', 'EXPORT_PERC']
        ].drop_duplicates(
        ).copy()

        # We add export percentages to the initial DataFrame
        focus = focus.merge(
            trade_statistics_extract,
            how='left',
            on='AFFILIATE_COUNTRY_CODE'
        )

        # We convert the unrelated-party revenues from plain USD to million USD and round to 1 decimal
        focus['UNRELATED_PARTY_REVENUES'] /= 10**6
        focus['UNRELATED_PARTY_REVENUES'] = focus['UNRELATED_PARTY_REVENUES'].map(lambda x: round(x, 1))

        # If relevant, we convert the export percentages into strings with the relevant number of decimals
        if formatted:
            focus['EXPORT_PERC'] = (focus['EXPORT_PERC'] * 100).map('{:.2f}'.format)
            focus['EXPORT_PERC'] = focus['EXPORT_PERC'].map(lambda x: '..' if x == 'nan' else x)
        else:
            focus['EXPORT_PERC'] *= 100
            focus['EXPORT_PERC'] = focus['EXPORT_PERC'].map(lambda x: '..' if np.isnan(x) else x)

        # Renaming columns
        focus.rename(
            columns={
                'AFFILIATE_COUNTRY_CODE': 'Affiliate jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (million USD)',
                'EXPORT_PERC': f'Share of {country_name} in exports (%)'
            },
            inplace=True
        )

        # Sorting the countries of origin of revenues from the largest to the smallest; restricting to the 10 largest
        focus = focus.sort_values(
            by='Unrelated-party revenues (million USD)',
            ascending=False
        ).head(10).reset_index(drop=True)

        # Adding a brief methodological note
        print(
            'Note that the export percentages are computed including the US in the destinations. '
            + 'Export percentages actually used in the computations (that exclude the US from the set of destinations) '
            + 'are therefore higher by, say, a few percentage points (except for the US themselves).'
        )

        return focus.copy()

    def get_country_profile(
        self,
        country_code: str
    ):
        """Prints and returns statistics and tables that delineate the impact of the adjustment for a specific partner.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param country_code: alpha-3 code of the country on which we want to focus
        :type country_code: str

        :rtype: tuple (3 different Pandas DataFrames)
        :return: 3 different Pandas DataFrames that allow to explore the impact of the adjustment for a given partner
        """
        # We will need the relevant income year
        year = self.year

        # ### Collecting the necessary data

        # We need the adjusted sales mapping
        sales_mapping = self.sales_mapping.copy()

        # We need the non-adjusted sales mapping
        irs_sales = self.irs.copy()

        # We need the trade statistics actually used in the adjustment
        trade_statistics = self.trade_statistics.copy()

        # We need the export percentages obtained from the BEA's statistics on the activities of US MNEs
        bea_preprocessor = ExtendedBEADataLoader(year=year, load_data_online=self.load_data_online)
        bea = bea_preprocessor.get_extended_sales_percentages()

        # We need non-winsorized trade statistics
        trade_stats_processor = TradeStatisticsProcessor(
            year=year,
            US_only=True,
            US_merchandise_exports_source=self.US_merchandise_exports_source,
            US_services_exports_source=self.US_services_exports_source,
            non_US_merchandise_exports_source=self.non_US_merchandise_exports_source,
            non_US_services_exports_source=self.non_US_services_exports_source,
            winsorize_export_percs=False,
            service_flows_to_exclude=self.service_flows_to_exclude,
            load_data_online=self.load_data_online
        )
        trade_stats_non_winsorized = trade_stats_processor.load_merged_data()

        # ### Country profile computations

        # Ex-ante and ex-post unrelated-party sales booked in the country
        unadjusted_upr = irs_sales[irs_sales['CODE'] == country_code]['UNRELATED_PARTY_REVENUES'].sum()

        print(
            f"Based on the IRS' country-by-country data, in {year}, US multinational companies booked",
            round(unadjusted_upr / 10**6, 2),
            f"million USD of unrelated-party revenues in {country_code}.\n"
        )

        adjusted_upr = sales_mapping[
            sales_mapping['OTHER_COUNTRY_CODE'] == country_code
        ]['UNRELATED_PARTY_REVENUES'].sum()

        print(
            f"Based on the adjusted country-by-country data, in {year}, US multinational companies booked",
            round(adjusted_upr / 10**6, 2),
            f"million USD of unrelated-party revenues in {country_code}.\n"
        )

        # Deducing the % change in the revenues attributed to this partner country due to the adjustment
        if adjusted_upr < unadjusted_upr:
            print(
                "This represents a decrease by",
                round(- (adjusted_upr - unadjusted_upr) / unadjusted_upr * 100, 2),
                "%.\n"
            )
        else:
            print(
                "This represents an increase by",
                round((adjusted_upr - unadjusted_upr) / unadjusted_upr * 100, 2),
                "%.\n"
            )

        # BEA sales percentages
        local_perc = bea.set_index('AFFILIATE_COUNTRY_CODE').loc[country_code, 'PERC_UNRELATED_AFFILIATE_COUNTRY']
        us_perc = bea.set_index('AFFILIATE_COUNTRY_CODE').loc[country_code, 'PERC_UNRELATED_US']
        other_country_perc = bea.set_index('AFFILIATE_COUNTRY_CODE').loc[country_code, 'PERC_UNRELATED_OTHER_COUNTRY']

        print(
            f"According to BEA data for the year {year}, local sales accounted for",
            round(local_perc * 100, 2),
            f"% of the unrelated-party sales of US multinational companies in {country_code}.",
            "Sales to the US represented",
            round(us_perc * 100, 2),
            "% of unrelated-party sales and eventually, sales to any other country accounted for",
            round(other_country_perc * 100, 2),
            f"% of the unrelated-party sales of US multinational companies in {country_code}.\n"
        )

        # Where do the sales attributed to the country in the adjusted database come from?
        print(
            f"We can track the unrelated-party revenues booked in {country_code} according to the adjusted database",
            f"back to the affiliate countries whose sales are attributed to {country_code}.\n"
        )

        top_ten = list(
            sales_mapping[
                sales_mapping['OTHER_COUNTRY_CODE'] == country_code
            ].sort_values(
                by='UNRELATED_PARTY_REVENUES',
                ascending=False
            ).head(
                10
            )[
                'AFFILIATE_COUNTRY_NAME'
            ].unique()
        )

        print(
            "From the largest to the lowest unrelated-party revenues, the top 10 affiliate countries from which",
            f"the sales attributed to {country_code} in the adjusted database are sourced are:",
            ', '.join(top_ten) + '.\n'
        )

        # ### Preparing the outputs

        # Table showing where the adjusted revenues come from
        sales_origin = sales_mapping[
            sales_mapping['OTHER_COUNTRY_CODE'] == country_code
        ].sort_values(
            by='UNRELATED_PARTY_REVENUES',
            ascending=False
        )

        # Relevant winsorized and non-winsorized trade statistics
        trade_stats_extract = trade_statistics[
            trade_statistics['OTHER_COUNTRY_CODE'] == country_code
        ].copy()

        trade_stats_non_winsorized = trade_stats_non_winsorized[
            trade_stats_non_winsorized['OTHER_COUNTRY_CODE'] == country_code
        ].copy()
        trade_stats_non_winsorized = trade_stats_non_winsorized.sort_values(
            by='ALL_EXPORTS', ascending=False
        )

        return sales_origin.copy(), trade_stats_extract.copy(), trade_stats_non_winsorized.copy()


class GlobalAnalysesProvider:

    def __init__(
        self,
        year: int,
        aamne_domestic_sales_perc: bool,
        breakdown_threshold: int,
        US_merchandise_exports_source: str,
        US_services_exports_source: str,
        non_US_merchandise_exports_source: str,
        non_US_services_exports_source: str,
        winsorize_export_percs: bool,
        US_winsorizing_threshold: float = 0.5,
        non_US_winsorizing_threshold: float = 0.5,
        service_flows_to_exclude: Optional[list] = None,
        macro_indicator: str = 'CONS',
        load_data_online: bool = False
    ):
        """Encapsulates the logic behind the analysis of global multinationals' non-adjusted and adjusted sales.

        :param year: year to consider for the analysis
        :type year: int
        :param aamne_domestic_sales_perc: whether to use the Analytical AMNE database for multinationals' domestic sales
        :type aamne_domestic_sales_perc: bool
        :param breakdown_threshold: minimum number of partners for a parent country to be included in the adjustment
        :type breakdown_threshold: int
        :param US_merchandise_exports_source: data source for the US exports of goods
        :type US_merchandise_exports_source: str
        :param US_services_exports_source: data source for the US exports of services
        :type US_services_exports_source: str
        :param non_US_merchandise_exports_source: data source for the non-US exports of goods
        :type non_US_merchandise_exports_source: str
        :param non_US_services_exports_source: data source for the non-US exports of services
        :type non_US_services_exports_source: str
        :param winsorize_export_percs: whether to winsorize small export percentages
        :type winsorize_export_percs: bool
        :param US_winsorizing_threshold: lower-bound winsorizing threshold for US exports in %, defaults to 0.5
        :type US_winsorizing_threshold: float, optional
        :param non_US_winsorizing_threshold: lower-bound winsorizing threshold for non-US exports in %, defaults to 0.5
        :type non_US_winsorizing_threshold: float, optional
        :param service_flows_to_exclude: types of service exports to exclude from trade statistics, defaults to None
        :type service_flows_to_exclude: list, optional
        :param macro_indicator: macro indicator with which to compare the distribution of sales, defaults to "CONS"
        :type macro_indicator: str, optional
        :param load_data_online: whether to load the data online (True) or locally (False), defaults to False
        :type load_data_online: bool, optional

        :raises Exception: if macro_indicator neither equal to "CONS", nor to "GNI"

        :rtype: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :return: object of the class USAnalysesProvider, allowing to analyse US multinationals' revenue variables
        """
        # Saving most arguments as attributes (used below in the code)
        self.year = year

        self.aamne_domestic_sales_perc = aamne_domestic_sales_perc

        self.US_merchandise_exports_source = US_merchandise_exports_source
        self.US_services_exports_source = US_services_exports_source
        self.non_US_merchandise_exports_source = non_US_merchandise_exports_source
        self.non_US_services_exports_source = non_US_services_exports_source

        self.winsorize_export_percs = winsorize_export_percs
        self.US_winsorizing_threshold = US_winsorizing_threshold
        self.non_US_winsorizing_threshold = non_US_winsorizing_threshold

        self.service_flows_to_exclude = service_flows_to_exclude

        # Depending on whether we load the data from online sources, paths differ
        self.load_data_online = load_data_online

        if not load_data_online:
            self.path_to_geographies = path_to_geographies
            self.path_to_tax_haven_list = path_to_tax_haven_list

            # Loading the relevant macroeconomic indicator
            self.path_to_GNI_data = path_to_GNI_data
            self.path_to_UNCTAD_data = path_to_UNCTAD_consumption_exp

        else:
            self.path_to_geographies = online_path_to_geo_file
            self.path_to_tax_haven_list = online_path_to_TH_list

            # Loading the relevant macroeconomic indicator
            self.path_to_GNI_data = online_path_to_GNI_data
            self.path_to_UNCTAD_data = online_path_to_CONS_data

        # Loading the indicator to which we compare firms' revenues depending on the macro_indicator argument
        if macro_indicator == 'GNI':
            self.macro_indicator = self.get_GNI_data()
            self.macro_indicator_prefix = 'GNI'
            self.macro_indicator_name = 'Gross National Income'

        elif macro_indicator == 'CONS':
            self.macro_indicator = self.get_consumption_expenditure_data()
            self.macro_indicator_prefix = 'CONS'
            self.macro_indicator_name = 'consumption expenditures'

        else:
            raise Exception(
                'Macroeconomic indicator can either be Gross National Income (pass "macro_indicator=GNI" as argument) '
                + 'or the UNCTAD consumption expenditure indicator (pass "macro_indicator=CONS" as argument).'
            )

        # Loading the list of tax havens
        tax_havens = pd.read_csv(self.path_to_tax_haven_list)
        self.tax_haven_country_codes = list(tax_havens['CODE'].unique()) + ['UKI']

        # Loading unadjusted revenue variables - Depends on the threshold retained as minimum number of partners
        self.breakdown_threshold = breakdown_threshold
        cbcr_preprocessor = CbCRPreprocessor(
            year=year,
            breakdown_threshold=breakdown_threshold,
            load_data_online=load_data_online
        )
        self.oecd = cbcr_preprocessor.get_preprocessed_revenue_data()

        # Loading adjusted revenue variables and trade statistics
        calculator = SimplifiedGlobalSalesCalculator(
            year=self.year,
            aamne_domestic_sales_perc=aamne_domestic_sales_perc,
            breakdown_threshold=breakdown_threshold,
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
        self.trade_statistics = calculator.trade_statistics.copy()
        self.sales_mapping = calculator.get_final_sales_mapping()

    def get_GNI_data(self) -> pd.DataFrame:
        """Loads Gross National Income data.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider

        :rtype: pandas.DataFrame
        :return: DataFrame containing the relevant Gross National Income series

        .. note:: This method relies on a dedicated data file, loaded online or locally and prepared preliminarily.
        """
        gross_national_income = pd.read_csv(self.path_to_GNI_data, delimiter=';')

        for column in gross_national_income.columns[2:]:
            gross_national_income[column] = gross_national_income[column].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            gross_national_income[column] = gross_national_income[column].astype(float)

        ser = gross_national_income[gross_national_income['COUNTRY_CODE'].isin(UK_CARIBBEAN_ISLANDS)].sum()
        ser['COUNTRY_NAME'] = 'UK Caribbean Islands'
        ser['COUNTRY_CODE'] = 'UKI'

        temp = ser.to_frame().T.copy()
        gross_national_income = pd.concat([gross_national_income, temp], axis=0).reset_index(drop=True)

        return gross_national_income.copy()

    def get_consumption_expenditure_data(self) -> pd.DataFrame:
        """Loading data on consumption expenditures.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider

        :rtype: pandas.DataFrame
        :return: DataFrame containing the relevant consumption expenditures series

        .. note:: This method relies on UNCTAD data, referred to in the OECD's draft rules for Pillar One Amount A.
        """
        # Reading the data file
        df = pd.read_csv(self.path_to_UNCTAD_data, encoding='latin')

        # Basic cleaning
        df = df.reset_index()
        df.columns = df.iloc[0]
        df = df.iloc[2:].copy()

        df = df.rename(
            columns={
                np.nan: 'COUNTRY_NAME',
                'YEAR': 'ITEM'
            }
        ).reset_index(drop=True)

        # Focusing on final consumption expenditures
        df = df[df['ITEM'].map(lambda x: x.strip()) == 'Final consumption expenditure'].copy()

        # Removing the rows with "_" for all the relevant years
        list_of_years = ['2016', '2017', '2018', '2019', '2020']
        df = df[df[list_of_years].sum(axis=1) != '_' * len(list_of_years)].copy()

        # Converting the columns to floats
        for col in list_of_years:
            df[col] = df[col].astype(float)

        df = df.drop(columns='ITEM')

        # Adding country and continent codes
        df['COUNTRY_NAME'] = df['COUNTRY_NAME'].map(lambda x: x.strip())

        df['COUNTRY_NAME'] = df['COUNTRY_NAME'].map(
            lambda country_name: {'France': 'France incl. Monaco'}.get(country_name, country_name)
        )
        df['COUNTRY_NAME'] = df['COUNTRY_NAME'].map(
            lambda country_name: {'France, metropolitan': 'France'}.get(country_name, country_name)
        )

        geographies = pd.read_csv(self.path_to_geographies)

        df = df.merge(
            geographies,
            how='left',
            left_on='COUNTRY_NAME', right_on='NAME'
        )

        df['CONTINENT_CODE'] = df['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        df['CONTINENT_CODE'] = df['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        df = df[['COUNTRY_NAME', 'CODE', 'CONTINENT_CODE', str(self.year)]].copy()

        # Adding UK Caribbean Islands
        temp = df[df['CODE'].isin(UK_CARIBBEAN_ISLANDS)][str(self.year)].sum()

        temp_1 = {
            'COUNTRY_NAME': 'United Kingdom Islands, Caribbean',
            'CODE': 'UKI',
            'CONTINENT_CODE': 'AMR',
            str(self.year): temp
        }

        df = df.append(temp_1, ignore_index=True)

        # Final step - Renaming the columns
        df = df.rename(
            columns={
                'CODE': 'COUNTRY_CODE',
                str(self.year): f'CONS_{str(self.year)}'
            }
        )

        return df.copy()

    def get_table_with_relevant_parents(self) -> pd.DataFrame:
        """Returns the list of selected partner countries (based on the breakdown threshold) and numbers of partners.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider

        :rtype: pandas.DataFrame
        :return: table presenting the parent countries selected for the adjustment and their number of partner countries
        """
        # Starting from the non-adjusted revenue variables
        df = self.oecd.copy()

        # For each parent country, we determine the number of unique affiliate countries
        df = df.groupby(
            ['PARENT_COUNTRY_CODE', 'PARENT_COUNTRY_NAME']
        ).nunique(
        )[
            'AFFILIATE_COUNTRY_CODE'
        ].reset_index()

        # We sort values (i) by the number of unique affiliate countries (descending) and (ii) alphabetically
        table_methodology = df[
            ['PARENT_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE']
        ].sort_values(
            by=['AFFILIATE_COUNTRY_CODE', 'PARENT_COUNTRY_NAME'],
            ascending=[False, True]
        ).rename(
            columns={
                'PARENT_COUNTRY_NAME': 'Parent country',
                'AFFILIATE_COUNTRY_CODE': 'Number of partner jurisdictions'
            }
        )

        # Changing the name of China in the table (shorter option)
        table_methodology['Parent country'] = table_methodology['Parent country'].map(
            lambda country_name: 'China' if country_name == "China (People's Republic of)" else country_name
        )

        return table_methodology.reset_index(drop=True)

    def get_table_1(
        self,
        formatted: bool = True,
        sales_type: str = 'unrelated'
    ) -> pd.DataFrame:
        """Builds Table 1 that presents the split of domestic vs. foreign sales (also split by continent) for each year.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param formatted: whether to format floats into strings in the DataFrame, defaults to True
        :type formatted: bool, optional
        :param sales_type: revenue variable to consider, defaults to "unrelated"
        :type sales_type: str, optional

        :raises Exception: if sales_type equal neither to "unrelated", to "related", nor to "total"

        :rtype: pandas.DataFrame
        :return: Table 1, presenting the split of domestic vs. foreign sales (also split by continent) for each year
        """
        # Deducing the relevant variable from the sales_type argument
        if sales_type not in ['unrelated', 'related', 'total']:
            raise Exception(
                'The type of sales to consider for building Table 1 can only be: "related" (for related-party revenues)'
                + ', "unrelated" (for unrelated-party revenues) or "total" (for total revenues).'
            )

        sales_type_correspondence = {
            'unrelated': 'UNRELATED_PARTY_REVENUES',
            'related': 'RELATED_PARTY_REVENUES',
            'total': 'TOTAL_REVENUES'
        }

        column_name = sales_type_correspondence[sales_type.lower()]

        # Dictionaries storing the total domestic and foreign sales respectively
        domestic_totals = {}
        foreign_totals = {}

        # Getting 2016 figures
        preprocessor = CbCRPreprocessor(
            year=2016,
            breakdown_threshold=self.breakdown_threshold,
            load_data_online=self.load_data_online
        )
        df = preprocessor.get_preprocessed_revenue_data()

        domestic_totals[2016] = df[df['PARENT_COUNTRY_CODE'] == df['AFFILIATE_COUNTRY_CODE']][column_name].sum()

        df = df[df['PARENT_COUNTRY_CODE'] != df['AFFILIATE_COUNTRY_CODE']].copy()

        foreign_totals[2016] = df[column_name].sum()

        # Continent-specific sub-totals
        df = df.groupby('CONTINENT_CODE').sum()[[column_name]]
        df[column_name] /= (foreign_totals[2016] / 100)
        df.rename(columns={column_name: 2016}, inplace=True)

        # Repeating the process for the other relevant year(s)
        for year in [2017]:

            preprocessor = CbCRPreprocessor(
                year=year,
                breakdown_threshold=self.breakdown_threshold,
                load_data_online=self.load_data_online
            )
            df_temp = preprocessor.get_preprocessed_revenue_data()

            domestic_totals[year] = df_temp[
                df_temp['PARENT_COUNTRY_CODE'] == df_temp['AFFILIATE_COUNTRY_CODE']
            ][column_name].sum()

            df_temp = df_temp[df_temp['PARENT_COUNTRY_CODE'] != df_temp['AFFILIATE_COUNTRY_CODE']].copy()

            foreign_totals[year] = df_temp[column_name].sum()

            df_temp = df_temp.groupby('CONTINENT_CODE').sum()[[column_name]]
            df_temp[column_name] /= (foreign_totals[year] / 100)
            df_temp.rename(columns={column_name: year}, inplace=True)

            df = pd.concat([df, df_temp], axis=1)

        # Reformatting the DataFrame
        dict_df = df.to_dict()

        indices = ['Domestic sales (billion USD)', 'Foreign sales (billion USD)']

        for year in [2016, 2017]:
            dict_df[year][indices[0]] = domestic_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        # Sorting values, giving the priority to the latest year
        df.sort_values(by=[2017, 2016], ascending=False, inplace=True)

        continent_names = {
            'EUR': 'Europe',
            'AMR': 'America',
            'APAC': 'Asia-Pacific',
            'AFR': 'Africa'
        }

        df.index = indices + [
            f'Of which {continent_names.get(continent, continent)} (%)' for continent in df.index[2:]
        ]

        # Formatting the floats into strings if relevant
        if formatted:
            for year in [2016, 2017]:
                df[year] = df[year].map('{:,.1f}'.format)

        return df.iloc[:-1, ].copy()  # We exclude the row corresponding to "Other Groups" in the table we output

    def get_intermediary_dataframe_1(
        self,
        include_macro_indicator: bool,
        exclude_US_from_parents: bool
    ) -> pd.DataFrame:
        """Builds the first intermediary DataFrame, used for Table 2.a. and 2.b. and for Figure 1 (global).

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param include_macro_indicator: whether to include the relevant macro indicator in the table
        :type include_macro_indicator: bool
        :param exclude_US_from_parents: whether to exclude the US from the set of parent countries considered
        :type exclude_US_from_parents: bool

        :rtype: pandas.DataFrame
        :return: first intermediary DataFrame, used for Table 2.a. and 2.b. and for Figure 1 (global)
        """
        # Starting from the unadjusted revenue variables in the OECD's data
        oecd = self.oecd.copy()

        columns_of_interest = ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']

        # Focusing on foreign revenues
        oecd = oecd[
            oecd['PARENT_COUNTRY_CODE'] != oecd['AFFILIATE_COUNTRY_CODE']
        ].copy()

        # If relevant, excluding the US from the set of parent countries considered
        if exclude_US_from_parents:
            oecd = oecd[oecd['PARENT_COUNTRY_CODE'] != 'USA'].copy()

        # For each affiliate / partner country, summing the revenues over the set of parent countries
        oecd = oecd.groupby(
            [
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME'
            ]
        ).sum()[
            columns_of_interest
        ].reset_index()

        # Including the macro indicator if relevant
        if include_macro_indicator:
            # Merging unadusted revenue variables and the macro indicator series
            merged_df = oecd.merge(
                self.macro_indicator[['COUNTRY_CODE', f'{self.macro_indicator_prefix}_{self.year}']].copy(),
                how='left',
                left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
            )

            # Eliminating countries for which we lack a macro indicator
            merged_df = merged_df[~merged_df[f'{self.macro_indicator_prefix}_{self.year}'].isnull()]

            columns_of_interest.append(f'{self.macro_indicator_prefix}_{self.year}')

        else:

            merged_df = oecd.copy()

        # Deducing shares from the absolute amounts
        new_columns = []

        for column in columns_of_interest:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            if column == f'{self.macro_indicator_prefix}_{self.year}':
                temp = merged_df[
                    ['AFFILIATE_COUNTRY_CODE', column]
                ].groupby(
                    'AFFILIATE_COUNTRY_CODE'
                ).first().reset_index()
                temp[new_column] = temp[column] / temp[column].sum() * 100

                merged_df = merged_df.merge(
                    temp,
                    how='inner',
                    on='AFFILIATE_COUNTRY_CODE'
                )

            else:
                merged_df[new_column] = merged_df[column] / merged_df[column].sum() * 100

        return merged_df.copy()

    def get_table_2_a(
        self,
        formatted: bool = True
    ) -> pd.DataFrame:
        """Builds Table 2.a., that ranks the 20 largest partner countries based on unadjusted unrelated-party revenues.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param formatted: whether to format the floats intro strings in the DataFrame, defaults to True
        :type formatted: bool, optional

        :rtype: pandas.DataFrame
        :return: Table 2.a., that ranks the 20 largest partner countries based on unadjusted unrelated-party revenues
        """
        # Loading the relevant intermediary DataFrame without the macro indicator
        merged_df = self.get_intermediary_dataframe_1(
            include_macro_indicator=False, exclude_US_from_parents=False
        )

        # Restricting the table to the relevant columns and ranking based on unrelated-party revenues
        output = merged_df[
            ['AFFILIATE_COUNTRY_NAME', 'UNRELATED_PARTY_REVENUES', 'SHARE_OF_UNRELATED_PARTY_REVENUES']
        ].sort_values(
            by='UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        # Moving from USD to billion USD
        output['UNRELATED_PARTY_REVENUES'] /= 10**9

        # Renaming the columns
        output.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME': 'Partner jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (USD billion)',
                'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
            },
            inplace=True
        )

        # Formatting floats into strings if relevant
        if formatted:
            for column in ['Unrelated-party revenues (USD billion)', 'Share of foreign unrelated-party revenues (%)']:
                output[column] = output[column].map('{:,.1f}'.format)

        output.reset_index(drop=True, inplace=True)

        return output.copy()

    def get_table_2_b(
        self,
        exclude_US_from_parents: bool = False,
        formatted: bool = True
    ) -> pd.DataFrame:
        """Builds Table 2.b., that ranks the 20 largest partners (unadjusted) with the macro indicator.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param exclude_US_from_parents: whether to exclude the US from the set of parent countries, defaults to False
        :type exclude_US_from_parents: bool, optional
        :param formatted: whether to format the floats intro strings in the DataFrame, defaults to True
        :type formatted: bool, optional

        :rtype: pandas.DataFrame
        :return: Table 2.b., that ranks the 20 largest partners (unadjusted) with the macro indicator
        """
        # Loading the relevant intermediary DataFrame with the macro indicator
        merged_df = self.get_intermediary_dataframe_1(
            include_macro_indicator=True, exclude_US_from_parents=exclude_US_from_parents
        )

        # Restricting the table to the relevant columns and ranking based on unrelated-party revenues
        output = merged_df[
            [
                'AFFILIATE_COUNTRY_NAME', 'SHARE_OF_UNRELATED_PARTY_REVENUES',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ]
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        # Formatting floats into strings if relevant
        if formatted:
            for column in ['SHARE_OF_UNRELATED_PARTY_REVENUES', f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}']:
                output[column] = output[column].map('{:,.1f}'.format)

        # (Almost) final step - Renaming the columns
        output.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME': 'Partner jurisdiction',
                'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}': f'Share of {self.macro_indicator_name} (%)'
            },
            inplace=True
        )

        output.reset_index(drop=True, inplace=True)

        return output.copy()

    def plot_figure_1(
        self,
        kind: str,
        exclude_US_from_parents: bool,
        save_PNG: bool = False,
        path_to_folder: Optional[str] = None
    ):
        """Plots Figure 1, showing the relationship between the distributions of revenues and of the macro indicator.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param kind: type of graph to show (either a regplot, a scatterplot or an interactive chart)
        :type kind: str
        :param exclude_US_from_parents: whether to exclude the US from the set of parent countries considered
        :type exclude_US_from_parents: bool
        :param save_PNG: whether to save the graph as a PNG file, defaults to False
        :type save_PNG: bool, optional
        :param path_to_folder: path to the destination folder where to store the PNG file, defaults to None
        :type path_to_folder: str, optional

        :raises Exception: if kind equal neither to "regplot", to "scatter", nor to "interactive"
        :raises Exception: if save_PNG is True and path_to_folder is equal to None
        :raises Exception: if save_PNG is True and kind is differs from "regplot" (only one type of graph can be saved)

        :rtype: None (plt.show())
        :return: None (plt.show())
        """
        # Checking the value of the kind parameter, determining what type of graph to plot
        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        # Checking that we have a path to the destination folder if we need to save the graph as a PNG file
        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        # We load the first intermediary DataFrame with the macro indicator
        merged_df = self.get_intermediary_dataframe_1(
            include_macro_indicator=True, exclude_US_from_parents=exclude_US_from_parents
        )

        # If we want to plot a regplot
        if kind == 'regplot':
            plot_df = merged_df.dropna().copy()

            # Adding a new column that determines the color of the dots (tax havens and others)
            plot_df['Category'] = (
                plot_df['AFFILIATE_COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven'})

            # Converting the shares of the macro indicator and the shares of unrelated-party revenues into floats
            plot_df[
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ] = plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'].astype(float)
            plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'] = plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'].astype(float)

            # Computing the correlation coefficient between the two series
            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'],
                plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )[1, 0]

            comment = (
                'Correlation between unrelated-party revenues and '
                + f'{self.macro_indicator_name} in {self.year}: {round(correlation, 2)}'
            )

            # Matplotlib parameters determining the look of the graph
            plt.rcParams.update(
                {
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18,
                    'legend.fontsize': 18
                }
            )

            # Instantiating the figure
            plt.figure(figsize=(17, 10))

            # Changing column names
            col_name_init = f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            col_name_new = f'Share of total {self.year} {self.macro_indicator_name} (%)'
            plot_df.rename(
                columns={
                    col_name_init: col_name_new,
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
                },
                inplace=True
            )

            # Plotting the regression line
            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                ci=None,
                truncate=False
            )

            # Adding scattered dots
            sns.scatterplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=100
            )

            if self.year == 2017:
                extract = plot_df.sort_values(
                    by='Share of foreign unrelated-party revenues (%)',
                    ascending=False
                ).head(8)

                for _, country in extract.iterrows():
                    if country['AFFILIATE_COUNTRY_CODE'] == 'SGP':
                        x = country[col_name_new] - 0.05
                        y = country['Share of foreign unrelated-party revenues (%)'] + 0.5
                    elif country['AFFILIATE_COUNTRY_CODE'] == 'HKG':
                        x = country[col_name_new]
                        y = country['Share of foreign unrelated-party revenues (%)'] - 0.8
                    else:
                        x = country[col_name_new] + 0.35
                        y = country['Share of foreign unrelated-party revenues (%)']

                    plt.text(
                        x=x,
                        y=y,
                        s=country['AFFILIATE_COUNTRY_CODE'],
                        size=16,
                        bbox=dict(facecolor='gray', alpha=0.7)
                    )

                plt.xlim(0, plot_df[col_name_new].max() + 2)
                plt.ylim(-1, plot_df['Share of foreign unrelated-party revenues (%)'].max() + 2)

            # Adding the title with the correlation coefficient
            plt.title(comment)

            # Saving as a PNG file if relevant
            if save_PNG:
                file_name = f'figure_1_{self.year}_global'

                if exclude_US_from_parents:
                    file_name += '_excl_US_from_parents'

                file_name += '.png'

                plt.savefig(
                    os.path.join(
                        path_to_folder,
                        file_name
                    ),
                    bbox_inches='tight'
                )

            plt.show()

        # Other types of graphs
        else:
            merged_df['IS_TAX_HAVEN'] = merged_df['AFFILIATE_COUNTRY_CODE'].isin(self.tax_haven_country_codes)

            plot_df = merged_df.dropna().copy()

            # Plotting a simple scatterplot
            if kind == 'scatter':
                plt.figure(figsize=(12, 7))

                sns.scatterplot(
                    x=f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    hue='IS_TAX_HAVEN',
                    data=plot_df
                )

                plt.show()

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

            # Plotting an interactive scatterplot with Plotly Express
            else:

                # colors = plot_df['IS_TAX_HAVEN'].map(
                #     lambda x: 'patate' if x == 0 else 'fraise'
                # )

                fig = px.scatter(
                    x=f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    color='IS_TAX_HAVEN',
                    color_discrete_sequence=['#EF553B', '#636EFA'],
                    data_frame=plot_df,
                    hover_name='AFFILIATE_COUNTRY_NAME'
                )

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

                fig.show()

    def get_table_4_intermediary(self) -> pd.DataFrame:
        """Builds the intermediary DataFrame used for Table 4 (continental split of the adjusted revenues).

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider

        :rtype: pandas.DataFrame
        :return: intermediary DataFrame used for Table 4, the latter showing the continental split of adjusted revenues
        """
        # Basic manipulation with the new sales mapping
        sales_mapping = self.sales_mapping.copy()

        sales_mapping = sales_mapping.groupby(
            ['PARENT_COUNTRY_CODE', 'OTHER_COUNTRY_CODE']
        ).sum().reset_index()

        # Cleaning geographies
        geographies = pd.read_csv(self.path_to_geographies)
        geographies = geographies[['CODE', 'CONTINENT_NAME']].drop_duplicates()

        geographies['CONTINENT_NAME'] = geographies['CONTINENT_NAME'].map(
            lambda x: 'Asia-Pacific' if x in ['Asia', 'Oceania'] or x is None else x
        )
        geographies['CONTINENT_NAME'] = geographies['CONTINENT_NAME'].map(
            lambda x: 'America' if x in ['South America', 'North America'] else x
        )

        # Merging the two DataFrames
        sales_mapping = sales_mapping.merge(
            geographies,
            how='left',
            left_on='OTHER_COUNTRY_CODE', right_on='CODE'
        )

        sales_mapping.drop(columns=['CODE'], inplace=True)

        continent_names_to_impute = {
            'OASIAOCN': 'Asia-Pacific',
            'UKI': 'America'
        }

        sales_mapping['CONTINENT_NAME'] = sales_mapping.apply(
            lambda row: continent_names_to_impute.get(row['OTHER_COUNTRY_CODE'], row['CONTINENT_NAME']),
            axis=1
        )

        return sales_mapping.copy()

    def get_table_4(
        self,
        sales_type: str = 'unrelated',
        formatted: bool = True
    ) -> pd.DataFrame:
        """Builds Table 4 that presents the split of domestic vs. foreign adjusted sales (also split by continent).

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param sales_type: revenue variable to consider, defaults to "unrelated"
        :type sales_type: str, optional
        :param formatted: whether to format floats into strings in the DataFrame, defaults to True
        :type formatted: bool, optional

        :raises Exception: if sales_type equal neither to "unrelated", to "related", nor to "total"

        :rtype: pandas.DataFrame
        :return: Table 4, presenting the split of domestic vs. foreign adjusted sales (also split by continent)
        """
        # Instantiating the dictionaries that will store the total domestic and foreign sales
        domestic_totals = {}
        foreign_totals = {}

        # Deducing the relevant variable from the sales_type argument
        if sales_type not in ['unrelated', 'related', 'total']:
            raise Exception(
                'The type of sales to consider for building Table 4 can only be: "related" (for related-party revenues)'
                + ', "unrelated" (for unrelated-party revenues) or "total" (for total revenues).'
            )

        sales_type_correspondence = {
            'unrelated': 'UNRELATED_PARTY_REVENUES',
            'related': 'RELATED_PARTY_REVENUES',
            'total': 'TOTAL_REVENUES'
        }

        column_name = sales_type_correspondence[sales_type.lower()]

        # Manipulations for the year 2016
        analyser = GlobalAnalysesProvider(
            year=2016,
            aamne_domestic_sales_perc=self.aamne_domestic_sales_perc,
            US_merchandise_exports_source=self.US_merchandise_exports_source,
            US_services_exports_source=self.US_services_exports_source,
            non_US_merchandise_exports_source=self.non_US_merchandise_exports_source,
            non_US_services_exports_source=self.non_US_services_exports_source,
            winsorize_export_percs=self.winsorize_export_percs,
            non_US_winsorizing_threshold=self.non_US_winsorizing_threshold,
            US_winsorizing_threshold=self.US_winsorizing_threshold,
            service_flows_to_exclude=self.service_flows_to_exclude
        )
        df = analyser.get_table_4_intermediary()

        domestic_totals[2016] = df[df['PARENT_COUNTRY_CODE'] == df['OTHER_COUNTRY_CODE']][column_name].sum()

        df = df[df['PARENT_COUNTRY_CODE'] != df['OTHER_COUNTRY_CODE']].copy()

        foreign_totals[2016] = df[column_name].sum()

        df = df.groupby('CONTINENT_NAME').sum()[[column_name]]
        df[column_name] /= (foreign_totals[2016] / 100)
        df.rename(columns={column_name: 2016}, inplace=True)

        # Replicating these operations for the other years of interest
        for year in [2017]:

            analyser = GlobalAnalysesProvider(
                year=year,
                aamne_domestic_sales_perc=self.aamne_domestic_sales_perc,
                US_merchandise_exports_source=self.US_merchandise_exports_source,
                US_services_exports_source=self.US_services_exports_source,
                non_US_merchandise_exports_source=self.non_US_merchandise_exports_source,
                non_US_services_exports_source=self.non_US_services_exports_source,
                winsorize_export_percs=self.winsorize_export_percs,
                non_US_winsorizing_threshold=self.non_US_winsorizing_threshold,
                US_winsorizing_threshold=self.US_winsorizing_threshold,
                service_flows_to_exclude=self.service_flows_to_exclude
            )
            df_temp = analyser.get_table_4_intermediary()

            domestic_totals[year] = df_temp[
                df_temp['PARENT_COUNTRY_CODE'] == df_temp['OTHER_COUNTRY_CODE']
            ][column_name].sum()

            df_temp = df_temp[df_temp['PARENT_COUNTRY_CODE'] != df_temp['OTHER_COUNTRY_CODE']].copy()

            foreign_totals[year] = df_temp[column_name].sum()

            df_temp = df_temp.groupby('CONTINENT_NAME').sum()[[column_name]]
            df_temp[column_name] /= (foreign_totals[year] / 100)
            df_temp.rename(columns={column_name: year}, inplace=True)

            df = pd.concat([df, df_temp], axis=1)

        # Formatting the DataFrame
        dict_df = df.to_dict()

        indices = ['Domestic sales (billion USD)', 'Sales abroad (billion USD)']

        # Moving from USD to billion USD
        for year in [2016, 2017]:
            dict_df[year][indices[0]] = domestic_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        # Sorting values based prioritarily on the latest year
        df.sort_values(by=[2017, 2016], ascending=False, inplace=True)

        # If relevant, formatting floats into strings with the proper number of decimals
        if formatted:
            for year in [2016, 2017]:
                df[year] = df[year].map('{:,.1f}'.format)

        # Final formatting step - Adding the missing row names
        df.index = indices + [f'Of which {continent} (%)' for continent in df.index[2:]]

        return df.copy()

    def get_intermediary_dataframe_2(
        self,
        include_macro_indicator: bool,
        exclude_US_from_parents: bool
    ) -> pd.DataFrame:
        """Builds the second intermediary DataFrame (based on adjusted sales), used for Table 5 and for Figure 2.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param include_macro_indicator: whether to include the relevant macro indicator in the table
        :type include_macro_indicator: bool
        :param exclude_US_from_parents: whether to exclude the US from the set of parent countries considered
        :type exclude_US_from_parents: bool

        :rtype: pandas.DataFrame
        :return: second intermediary DataFrame (based on adjusted sales), used for Table 5 and for Figure 2
        """
        # Starting from the adjusted revenue variables
        sales_mapping = self.sales_mapping.copy()

        # Focusing on foreign revenues
        sales_mapping = sales_mapping[
            sales_mapping['PARENT_COUNTRY_CODE'] != sales_mapping['OTHER_COUNTRY_CODE']
        ].copy()

        # If relevant, excluding the US from the set of parent countries considered
        if exclude_US_from_parents:
            sales_mapping = sales_mapping[sales_mapping['PARENT_COUNTRY_CODE'] != 'USA'].copy()

        # Aggregating the sales over final destinations
        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        # Including the relevant macro indicator depending on the include_macro_indicator argument
        if include_macro_indicator:
            # Merging the macro indicator series over the adjusted sales mapping
            sales_mapping = sales_mapping.merge(
                self.macro_indicator[
                    ['COUNTRY_CODE', 'COUNTRY_NAME', f'{self.macro_indicator_prefix}_{self.year}']
                ].copy(),
                how='left',
                left_on='OTHER_COUNTRY_CODE', right_on='COUNTRY_CODE'
            )

            sales_mapping.drop(columns='OTHER_COUNTRY_CODE', inplace=True)

            # Cleaning the macro indicator series and converting it into floats
            sales_mapping[
                f'{self.macro_indicator_prefix}_{self.year}'
            ] = sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            sales_mapping[
                f'{self.macro_indicator_prefix}_{self.year}'
            ] = sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].astype(float)

            # Removing countries for which we lack the macro indicator
            sales_mapping = sales_mapping[~sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].isnull()].copy()

        # Moving from absolute amounts to shares (of the macro indicator and of sales)
        new_columns = []

        for column in [
            'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES',
            'TOTAL_REVENUES', f'{self.macro_indicator_prefix}_{self.year}'
        ]:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            if column == f'{self.macro_indicator_prefix}_{self.year}':
                temp = sales_mapping[
                    ['COUNTRY_CODE', column]
                ].groupby(
                    'COUNTRY_CODE'
                ).first().reset_index()
                temp[new_column] = temp[column] / temp[column].sum() * 100

                sales_mapping = sales_mapping.merge(
                    temp,
                    how='inner',
                    on='COUNTRY_CODE'
                )

            else:
                sales_mapping[new_column] = sales_mapping[column] / sales_mapping[column].sum() * 100

        return sales_mapping.copy()

    def plot_figure_2(
        self,
        kind: str,
        exclude_US_from_parents: bool,
        save_PNG: bool = False,
        path_to_folder: Optional[str] = None
    ):
        """Plots Figure 2, showing the relationship between adjusted revenues and the macro indicator.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param kind: type of graph to show (either a regplot, a scatterplot or an interactive chart)
        :type kind: str
        :param exclude_US_from_parents: whether to exclude the US from the set of parent countries considered
        :type exclude_US_from_parents: bool
        :param save_PNG: whether to save the graph as a PNG file, defaults to False
        :type save_PNG: bool, optional
        :param path_to_folder: path to the destination folder where to store the PNG file, defaults to None
        :type path_to_folder: str, optional

        :raises Exception: if kind equal neither to "regplot", to "scatter", nor to "interactive"
        :raises Exception: if save_PNG is True and path_to_folder is equal to None
        :raises Exception: if save_PNG is True and kind is differs from "regplot" (only one type of graph can be saved)

        :rtype: None (plt.show())
        :return: None (plt.show())
        """
        # Checking the value of the kind parameter, determining what type of graph to plot
        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        # Checking that we have a path to the destination folder if we need to save the graph as a PNG file
        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        # We load the second intermediary DataFrame with the macro indicator
        merged_df = self.get_intermediary_dataframe_2(
            include_macro_indicator=True, exclude_US_from_parents=exclude_US_from_parents
        )

        # If we want to plot a regplot
        if kind == 'regplot':
            plot_df = merged_df.dropna().copy()

            # Adding a new column that determines the color of the dots (tax havens and others)
            plot_df['Category'] = (
                plot_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven'})

            # Computing the correlation coefficient between the two series
            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'].astype(float),
                plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'].astype(float)
            )[1, 0]

            comment = (
                'Correlation between unrelated-party revenues and '
                + f'{self.macro_indicator_name} in {self.year}: {round(correlation, 2)}'
            )

            # Matplotlib parameters determining the look of the graph
            plt.rcParams.update(
                {
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18,
                    'legend.fontsize': 18
                }
            )

            # Instantiating the figure
            plt.figure(figsize=(17, 10))

            # Changing column names
            col_name_init = f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            col_name_new = f'Share of total {self.year} {self.macro_indicator_name} (%)'
            plot_df.rename(
                columns={
                    col_name_init: col_name_new,
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
                },
                inplace=True
            )

            # Plotting the regression line
            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                ci=None,
                truncate=False
            )

            # Adding the scattered dots
            sns.scatterplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=100
            )

            if self.year == 2017:
                extract = plot_df.sort_values(
                    by='Share of foreign unrelated-party revenues (%)',
                    ascending=False
                ).head(10)
                for _, country in extract.iterrows():
                    if country['COUNTRY_CODE'] == 'HKG':
                        x = country[col_name_new] - 0.4
                        y = country['Share of foreign unrelated-party revenues (%)'] + 0.5
                    elif country['COUNTRY_CODE'] == 'SGP':
                        x = country[col_name_new] - 1.4
                        y = country['Share of foreign unrelated-party revenues (%)'] - 0.1
                    else:
                        x = country[col_name_new] + 0.4
                        y = country['Share of foreign unrelated-party revenues (%)'] - 0.1

                    plt.text(
                        x=x,
                        y=y,
                        s=country['COUNTRY_CODE'],
                        size=16,
                        bbox=dict(facecolor='gray', alpha=0.7)
                    )

                plt.xlim(-1.5, plot_df[col_name_new].max() + 2)
                plt.ylim(-0.3, plot_df['Share of foreign unrelated-party revenues (%)'].max() + 0.8)

            # Adding the title with the correlation coefficient
            plt.title(comment)

            # Saving as a PNG file if relevant
            if save_PNG:

                file_name = f'figure_2_{self.year}_global{"_AAMNE" if self.aamne_domestic_sales_perc else ""}'

                if exclude_US_from_parents:
                    file_name += '_excl_US_from_parents'

                file_name += '.png'

                plt.savefig(
                    os.path.join(
                        path_to_folder,
                        file_name
                    ),
                    bbox_inches='tight'
                )

            plt.show()

        # Other types of graphs
        else:
            merged_df['IS_TAX_HAVEN'] = merged_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['COUNTRY_NAME'] != 'United States'].copy()

            # Plotting a simple scatterplot
            if kind == 'scatter':

                plt.figure(figsize=(12, 7))

                sns.scatterplot(
                    x=f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    hue='IS_TAX_HAVEN',
                    data=plot_df
                )

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

                plt.show()

            # Plotting an interactive scatterplot with Plotly Express
            else:
                colors = plot_df['IS_TAX_HAVEN'].map(
                    lambda x: 'blue' if x == 0 else 'red'
                )

                fig = px.scatter(
                    x=f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    color=colors,
                    data_frame=plot_df,
                    hover_name='COUNTRY_NAME'
                )

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

                fig.show()

    def get_table_5(
        self,
        formatted: bool = True
    ) -> pd.DataFrame:
        """Builds Table 5, that ranks the 20 largest partner countries based on adjusted unrelated-party revenues.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param formatted: whether to format the floats intro strings in the DataFrame, defaults to True
        :type formatted: bool, optional

        :rtype: pandas.DataFrame
        :return: Table 5, that ranks the 20 largest partners countries based on adjusted unrelated-party revenues
        """
        # Loading the second intermediary DataFrame with the macro indicator
        merged_df = self.get_intermediary_dataframe_2(
            include_macro_indicator=True, exclude_US_from_parents=False
        )

        # Restricting the DataFrame to the relevant columns and sorting values based on unrelated-party revenues
        output = merged_df[
            [
                'COUNTRY_NAME',
                'UNRELATED_PARTY_REVENUES',
                'SHARE_OF_UNRELATED_PARTY_REVENUES',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ]
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        output.reset_index(drop=True, inplace=True)

        # Moving from USD to billion USD
        output['UNRELATED_PARTY_REVENUES'] /= 10**9

        # If relevant, formatting floats into strings with the proper number of decimals
        if formatted:
            for column in output.columns[1:]:
                output[column] = output[column].map('{:.1f}'.format)

        # Final formatting step - Renaming columns
        output.rename(
            columns={
                'COUNTRY_NAME': 'Partner jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (USD billion)',
                'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of unrelated-party revenues (%)',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}': f'Share of {self.macro_indicator_name} (%)'
            },
            inplace=True
        )

        return output.copy()

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Builds a DataFrame with each country's adjusted and unadjusted unrelated-party revenues for comparison.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider

        :rtype: pandas.DataFrame
        :return: DataFrame with each country's adjusted and unadjusted unrelated-party revenues
        """
        # Unadjusted sales mapping
        oecd = self.oecd.copy()
        oecd = oecd.groupby(
            ['AFFILIATE_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE']
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        # Adjusted sales mapping
        sales_mapping = self.sales_mapping.copy()
        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        # Merging the two unrelated-party revenues series
        merged_df = oecd.merge(
            sales_mapping[['OTHER_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES']],
            how='inner',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='OTHER_COUNTRY_CODE'
        )

        merged_df.drop(columns=['OTHER_COUNTRY_CODE'], inplace=True)

        # Final step - Renaming columns
        merged_df.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME': 'COUNTRY_NAME',
                'AFFILIATE_COUNTRY_CODE': 'COUNTRY_CODE',
                'UNRELATED_PARTY_REVENUES_x': 'UPR_OECD',
                'UNRELATED_PARTY_REVENUES_y': 'UPR_ADJUSTED'
            },
            inplace=True
        )

        return merged_df.copy()

    def get_focus_on_tax_havens(self) -> pd.DataFrame:
        """Builds a DataFrame that allows to compare the unadjusted and adjusted revenues booked in tax havens.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider

        :rtype: pandas.DataFrame
        :return: DataFrame focused on the impact of the adjustment on tax havens
        """
        # We load the DataFrame allowing to compare each partner's non-adjusted and adjusted revenues
        merged_df = self.get_comparison_dataframe()

        # Restricting the set of partner countries to tax havens
        restricted_df = merged_df[
            merged_df['COUNTRY_CODE'].isin(
                self.tax_haven_country_codes
            )
        ].copy()

        # Ranking based on non-adjusted unrelated-party revenues (from largest to smallest)
        restricted_df.sort_values(by='UPR_OECD', ascending=False, inplace=True)

        restricted_df.reset_index(drop=True, inplace=True)

        # We add a row to the DataFrame that shows the total (non-adjusted and adjusted) revenues in tax havens
        dict_df = restricted_df.to_dict()

        dict_df[restricted_df.columns[0]][len(restricted_df)] = 'Total for tax havens'
        dict_df[restricted_df.columns[1]][len(restricted_df)] = '..'
        dict_df[restricted_df.columns[2]][len(restricted_df)] = restricted_df['UPR_OECD'].sum()
        dict_df[restricted_df.columns[3]][len(restricted_df)] = restricted_df['UPR_ADJUSTED'].sum()

        restricted_df = pd.DataFrame.from_dict(dict_df)

        # Moving from absolute amounts to shares
        restricted_df['SHARE_OF_UPR_OECD'] = restricted_df['UPR_OECD'] / merged_df['UPR_OECD'].sum() * 100
        restricted_df['SHARE_OF_UPR_ADJUSTED'] = restricted_df['UPR_ADJUSTED'] / merged_df['UPR_ADJUSTED'].sum() * 100

        # Expressing the absolute amounts in million USD, instead of plain USD
        for column in ['UPR_OECD', 'UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column] / 10**6
            # restricted_df[column] = restricted_df[column].map(round)

        # Rounding revenue shares with three decimals
        for column in ['SHARE_OF_UPR_OECD', 'SHARE_OF_UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column].map(
                lambda x: round(x, 3)
            )

        # (Almost) final step - Renaming columns
        restricted_df.rename(
            columns={
                'COUNTRY_NAME': 'Country name',
                'UPR_OECD': 'Unrelated-party revenues based on OECD ($m)',
                'UPR_ADJUSTED': 'Adjusted unrelated-party revenues ($m)',
                'SHARE_OF_UPR_OECD': 'Share of UPR based on OECD (%)',
                'SHARE_OF_UPR_ADJUSTED': 'Share of adjusted UPR (%)'
            },
            inplace=True
        )

        restricted_df.drop(columns=['COUNTRY_CODE'], inplace=True)

        return restricted_df.copy()

    def plot_focus_on_tax_havens(
        self,
        orient: str = 'horizontal',
        save_PNG: bool = False,
        path_to_folder: Optional[str] = None
    ):
        """Plots Figure 3, that shows the evolution of the unrelated-party revenues attributed to in-sample tax havens.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param orient: orientation of the bars in the graph, defaults to "horizontal"
        :type orient: str, optional
        :param save_PNG: whether to save the graph as a PNG file, defaults to False
        :type save_PNG: bool, optional
        :param path_to_folder: path to the destination folder where to store the PNG file, defaults to None
        :type path_to_folder: str, optional

        :raises Exception: if orient is specified and equal neither to "horizontal", nor to "vertical"
        :raises Exception: if save_PNG is True and path_to_folder is equal to None

        :rtype: None (plt.show())
        :return: None (plt.show())
        """
        # Checking that if one wants to save the graph as a PNG file, a path to the destination folder was specified
        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        # Setting some variables that will determine the appearance of the graph depending on the orientation chosen
        if orient == 'horizontal':
            figsize = (12, 12)
            ascending = False

        elif orient == 'vertical':
            figsize = (12, 8)
            ascending = True

        # orient must either be equal to "horizontal" (default value) or to "vertical"
        else:
            raise Exception('Orientation of the graph can only be "horizontal" or "vertical".')

        # We start from the DataFrame that tracks the evolution of tax havens
        df = self.get_focus_on_tax_havens()

        # We compute each tax haven's % change in unrelated-party revenues through the adjustment
        df['Change in unrelated-party revenues (%)'] = (df[df.columns[2]] / df[df.columns[1]] - 1) * 100

        # If, for some tax havens, the absolute % change is over 100%, they will not be shown on the graph
        # Instead, we print the name of the tax haven(s), its non-adjusted and adjusted UPR and the % change
        if (np.abs(df[df.columns[-1]]) >= 100).sum() > 0:
            for _, row in df[np.abs(df[df.columns[-1]]) >= 100].iterrows():
                print(
                    row['Country name'], '-', row['Unrelated-party revenues based on OECD ($m)'],
                    '-', row['Adjusted unrelated-party revenues ($m)'], '-', row[df.columns[-1]]
                )

        # Eliminating the tax haven(s) concerned
        df = df[np.abs(df[df.columns[-1]]) < 100].copy()

        # Sorting values based on the % change
        df_sorted = df.sort_values(
            by=df.columns[-1],
            ascending=ascending
        ).copy()

        # Settings for the aspect of the graph
        plt.rcParams.update(
            {
                'axes.titlesize': 20,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18
            }
        )

        # Instantiating the figure
        plt.figure(figsize=figsize)

        # Bars associated with a decrease are displayed in red; the others are displayed in blue
        y_pos = np.arange(len(df_sorted))
        colors = df_sorted[df_sorted.columns[-1]].map(
            lambda x: 'darkred' if x < 0 else 'darkblue'
        )

        # If the bars are oriented horizontally
        if orient == 'horizontal':
            plt.barh(
                y_pos,
                df_sorted[df_sorted.columns[-1]],
                color=colors
            )

            plt.yticks(
                ticks=y_pos,
                labels=df_sorted['Country name']
            )

            plt.xlabel(df_sorted.columns[-1])

            file_name_suffix = ''

        # If the bars are oriented vertically
        else:
            plt.bar(
                y_pos,
                df_sorted[df_sorted.columns[-1]],
                color=colors
            )

            # We shorten some country names
            df_sorted['Country name'] = df_sorted['Country name'].map(
                lambda x: 'UK Caribbean Islands' if x == 'United Kingdom Islands, Caribbean' else x
            )
            df_sorted['Country name'] = df_sorted['Country name'].map(
                lambda x: 'St. Vincent and the Gr.' if x == 'Saint Vincent and the Grenadines' else x
            )

            plt.xticks(
                ticks=y_pos,
                labels=df_sorted['Country name'],
                rotation=90
            )

            plt.ylabel(df_sorted.columns[-1])

            # To distinguish the PNG files associated with horizontal and vertical graphs, we add a specific suffix
            file_name_suffix = 'v'

        # Restricting the white spaces on the sides of the graph
        plt.tight_layout()

        # Saving the graph as a PNG file if relevant
        if save_PNG:
            plt.savefig(
                os.path.join(
                    path_to_folder,
                    (
                        f'focus_on_tax_havens_{self.year}_global{"_AAMNE" if self.aamne_domestic_sales_perc else ""}'
                        + f'_{file_name_suffix}.png'
                    )
                ),
                bbox_inches='tight'
            )

        plt.show()

    def get_table_6(
        self,
        country_code: str
    ) -> pd.DataFrame:
        """Builds Table 6 for any given in-sample partner country, showing where its adjusted revenues come from.

        :param self: the GlobalAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.GlobalAnalysesProvider
        :param country_code: alpha-3 code of the country on which we want to focus
        :type country_code: str

        :rtype: pandas.DataFrame
        :return: table showing, for the country considered, where its adjusted revenues come from
        """
        # For some countries for which this table is especially relevant and useful, we change the code into a full name
        if country_code == 'BEL':
            country_name = 'Belgium'

        elif country_code == 'LBN':
            country_name = 'Lebanon'

        elif country_code == 'NLD':
            country_name = 'the Netherlands'

        else:
            country_name = country_code

        # We start from the adjusted sales mapping
        sales_mapping = self.sales_mapping.copy()

        # We restrict it to the destination / partner country of interest
        focus = sales_mapping[sales_mapping['OTHER_COUNTRY_CODE'] == country_code].copy()

        focus = focus.groupby(
            'AFFILIATE_COUNTRY_CODE'
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        # We also look into trade statistics and more specifically, at the exports to the chosen destination country
        trade_statistics = self.trade_statistics.copy()
        trade_statistics_extract = trade_statistics[trade_statistics['OTHER_COUNTRY_CODE'] == country_code].copy()
        trade_statistics_extract = trade_statistics_extract[
            ['AFFILIATE_COUNTRY_CODE', 'EXPORT_PERC']
        ].drop_duplicates(
        ).copy()

        # We add export percentages to the initial DataFrame
        focus = focus.merge(
            trade_statistics_extract,
            how='left',
            on='AFFILIATE_COUNTRY_CODE'
        )

        # We convert the unrelated-party revenues from plain USD to million USD and round to 1 decimal
        focus['UNRELATED_PARTY_REVENUES'] /= 10**6
        focus['UNRELATED_PARTY_REVENUES'] = focus['UNRELATED_PARTY_REVENUES'].map(lambda x: round(x, 1))

        # We convert the export percentages into strings with the relevant number of decimals
        focus['EXPORT_PERC'] = (focus['EXPORT_PERC'] * 100).map('{:.2f}'.format)
        focus['EXPORT_PERC'] = focus['EXPORT_PERC'].map(lambda x: '..' if x == 'nan' else x)

        # Renaming columns
        focus.rename(
            columns={
                'AFFILIATE_COUNTRY_CODE': 'Affiliate jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (million USD)',
                'EXPORT_PERC': f'Share of {country_name} in exports (%)'
            },
            inplace=True
        )

        # Sorting the countries of origin of revenues from the largest to the smallest; restricting to the 10 largest
        focus = focus.sort_values(
            by='Unrelated-party revenues (million USD)',
            ascending=False
        ).head(10).reset_index(drop=True)

        return focus.copy()

    def get_country_profile(
        self,
        country_code: str
    ):
        """Prints and returns statistics and tables that delineate the impact of the adjustment for a specific partner.

        :param self: the USAnalysesProvider object itself (method)
        :type self: destination_based_sales.analyses_provider.USAnalysesProvider
        :param country_code: alpha-3 code of the country on which we want to focus
        :type country_code: str

        :rtype: tuple (3 different Pandas DataFrames)
        :return: 3 different Pandas DataFrames that allow to explore the impact of the adjustment for a given partner
        """
        # We will need the relevant income year
        year = self.year

        # ### Collecting the necessary data

        # We need the adjusted sales mapping
        sales_mapping = self.sales_mapping.copy()

        # We need the non-adjusted sales mapping
        oecd_sales = self.oecd.copy()

        # We need the trade statistics actually used in the adjustment
        trade_statistics = self.trade_statistics.copy()

        # We need the export percentages obtained from the BEA's statistics on the activities of US MNEs
        bea_preprocessor = ExtendedBEADataLoader(year=year, load_data_online=self.load_data_online)
        bea = bea_preprocessor.get_extended_sales_percentages()

        # We need non-winsorized trade statistics
        trade_stats_processor = TradeStatisticsProcessor(
            year=year,
            US_only=True,
            US_merchandise_exports_source=self.US_merchandise_exports_source,
            US_services_exports_source=self.US_services_exports_source,
            non_US_merchandise_exports_source=self.non_US_merchandise_exports_source,
            non_US_services_exports_source=self.non_US_services_exports_source,
            winsorize_export_percs=False,
            service_flows_to_exclude=self.service_flows_to_exclude,
            load_data_online=self.load_data_online
        )
        trade_stats_non_winsorized = trade_stats_processor.load_merged_data()

        # ### Country profile computations

        # Ex-ante and ex-post unrelated-party sales booked in the country
        unadjusted_upr = oecd_sales[
            oecd_sales['AFFILIATE_COUNTRY_CODE'] == country_code
        ]['UNRELATED_PARTY_REVENUES'].sum()

        print(
            f"Based on the OECD's country-by-country data, in {year}, multinational companies booked",
            round(unadjusted_upr / 10**6, 2),
            f"million USD of unrelated-party revenues in {country_code}.\n"
        )

        adjusted_upr = sales_mapping[
            sales_mapping['OTHER_COUNTRY_CODE'] == country_code
        ]['UNRELATED_PARTY_REVENUES'].sum()

        print(
            f"Based on the adjusted country-by-country data, in {year}, multinational companies booked",
            round(adjusted_upr / 10**6, 2),
            f"million USD of unrelated-party revenues in {country_code}.\n"
        )

        # Deducing the % change in the revenues attributed to this partner country due to the adjustment
        if adjusted_upr < unadjusted_upr:
            print(
                "This represents a decrease by",
                round(- (adjusted_upr - unadjusted_upr) / unadjusted_upr * 100, 2),
                "%.\n"
            )
        else:
            print(
                "This represents an increase by",
                round((adjusted_upr - unadjusted_upr) / unadjusted_upr * 100, 2),
                "%.\n"
            )

        # BEA sales percentages
        local_perc = bea.set_index('AFFILIATE_COUNTRY_CODE').loc[country_code, 'PERC_UNRELATED_AFFILIATE_COUNTRY']
        us_perc = bea.set_index('AFFILIATE_COUNTRY_CODE').loc[country_code, 'PERC_UNRELATED_US']
        other_country_perc = bea.set_index('AFFILIATE_COUNTRY_CODE').loc[country_code, 'PERC_UNRELATED_OTHER_COUNTRY']

        print(
            f"According to BEA data for the year {year}, local sales accounted for",
            round(local_perc * 100, 2),
            f"% of the unrelated-party sales of US multinational companies in {country_code}.",
            "Sales to the US represented",
            round(us_perc * 100, 2),
            "% of unrelated-party sales and eventually, sales to any other country accounted for",
            round(other_country_perc * 100, 2),
            f"% of the unrelated-party sales of US multinational companies in {country_code}.\n"
        )

        print(
            "For the extension to non-US multinational companies, we therefore consider that",
            round(local_perc * 100, 2),
            f"% of unrelated-party sales booked in {country_code} are local sales and that",
            round((us_perc + other_country_perc) * 100, 2),
            f"% of unrelated-party sales booked in {country_code} are generated in any foreign jurisdiction.",
        )

        # Where do the sales attributed to the country in the adjusted database come from?
        print(
            f"We can track the unrelated-party revenues booked in {country_code} according to the adjusted database",
            f"back to the affiliate countries whose sales are attributed to {country_code}.\n"
        )

        top_ten = list(
            sales_mapping[
                sales_mapping['OTHER_COUNTRY_CODE'] == country_code
            ].sort_values(
                by='UNRELATED_PARTY_REVENUES',
                ascending=False
            ).head(
                10
            )[
                'AFFILIATE_COUNTRY_CODE'
            ].unique()
        )

        print(
            "From the largest to the lowest unrelated-party revenues, the top 10 affiliate countries from which",
            f"the sales attributed to {country_code} in the adjusted database are sourced are:",
            ', '.join(top_ten) + '.\n'
        )

        # ### Preparing the outputs

        # Table showing where the adjusted revenues come from
        sales_origin = sales_mapping[
            sales_mapping['OTHER_COUNTRY_CODE'] == country_code
        ].sort_values(
            by='UNRELATED_PARTY_REVENUES',
            ascending=False
        )

        # Relevant winsorized and non-winsorized trade statistics
        trade_stats_extract = trade_statistics[
            trade_statistics['OTHER_COUNTRY_CODE'] == country_code
        ].copy()

        trade_stats_non_winsorized = trade_stats_non_winsorized[
            trade_stats_non_winsorized['OTHER_COUNTRY_CODE'] == country_code
        ].copy()

        trade_stats_non_winsorized = trade_stats_non_winsorized.sort_values(
            by='ALL_EXPORTS', ascending=False
        )

        return sales_origin.copy(), trade_stats_extract.copy(), trade_stats_non_winsorized.copy()


if __name__ == '__main__':
    final_output = {}

    path_to_folder = sys.argv[1]

    for year in [2016, 2017, 2018]:

        analyser = USAnalysesProvider(
            year=year,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            winsorize_export_percs=True,
            service_flows_to_exclude=[]
        )
        industry_analyser = PerIndustryAnalyser(year=year)

        table_1 = analyser.get_table_1(formatted=False)
        table_2_a = analyser.get_table_2_a(formatted=False)
        table_2_b = analyser.get_table_2_b(formatted=False)
        table_4 = analyser.get_table_4(formatted=False)
        table_5 = analyser.get_table_5(formatted=False)
        table_6 = analyser.get_table_6(country_code='BEL')

        final_output[f'table_1_{year}'] = table_1.copy()
        final_output[f'table_2_a_{year}'] = table_2_a.copy()
        final_output[f'table_2_b_{year}'] = table_2_b.copy()
        final_output[f'table_4_{year}'] = table_4.copy()
        final_output[f'table_5_{year}'] = table_5.copy()
        final_output[f'table_6_{year}'] = table_6.copy()

        del table_1
        del table_2_a
        del table_2_b
        del table_4
        del table_5
        del table_6

    with pd.ExcelWriter(os.path.join(path_to_folder, 'tables_PYTHON_OUTPUT.xlsx'), engine='xlsxwriter') as writer:
        for key, value in final_output.items():
            value.to_excel(writer, sheet_name=key, index=True)
