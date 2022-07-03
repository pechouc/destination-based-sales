########################################################################################################################
# --- Imports

import os
import sys

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
        year,
        US_merchandise_exports_source,
        US_services_exports_source,
        non_US_merchandise_exports_source,
        non_US_services_exports_source,
        winsorize_export_percs,
        US_winsorizing_threshold=0.5,
        non_US_winsorizing_threshold=0.5,
        service_flows_to_exclude=None,
        macro_indicator='CONS',
        load_data_online=False
    ):

        self.year = year

        self.US_merchandise_exports_source = US_merchandise_exports_source
        self.US_services_exports_source = US_services_exports_source
        self.non_US_merchandise_exports_source = non_US_merchandise_exports_source
        self.non_US_services_exports_source = non_US_services_exports_source

        self.winsorize_export_percs = winsorize_export_percs
        self.US_winsorizing_threshold = US_winsorizing_threshold
        self.non_US_winsorizing_threshold = non_US_winsorizing_threshold

        self.service_flows_to_exclude = service_flows_to_exclude

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

        irs_preprocessor = IRSDataPreprocessor(year=year, load_data_online=load_data_online)
        self.irs = irs_preprocessor.load_final_data()

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

    def get_GNI_data(self):
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

    def get_consumption_expenditure_data(self):
        df = pd.read_csv(self.path_to_UNCTAD_data, encoding='latin')

        df = df.reset_index()
        df.columns = df.iloc[0]
        df = df.iloc[2:].copy()

        df = df.rename(
            columns={
                np.nan: 'COUNTRY_NAME',
                'YEAR': 'ITEM'
            }
        ).reset_index(drop=True)

        df = df[df['ITEM'].map(lambda x: x.strip()) == 'Final consumption expenditure'].copy()

        list_of_years = ['2016', '2017', '2018', '2019', '2020']
        df = df[df[list_of_years].sum(axis=1) != '_' * len(list_of_years)].copy()

        for col in list_of_years:
            df[col] = df[col].astype(float)

        df = df.drop(columns='ITEM')

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

        df = df.rename(
            columns={
                'CODE': 'COUNTRY_CODE',
                str(self.year): f'CONS_{str(self.year)}'
            }
        )

        return df.dropna().copy()

    def get_table_1(self, formatted=True, sales_type='unrelated'):

        sales_type_correspondence = {
            'unrelated': 'UNRELATED_PARTY_REVENUES',
            'related': 'RELATED_PARTY_REVENUES',
            'total': 'TOTAL_REVENUES'
        }

        column_name = sales_type_correspondence[sales_type.lower()]

        us_totals = {}
        foreign_totals = {}

        preprocessor = IRSDataPreprocessor(year=2016, load_data_online=self.load_data_online)
        df = preprocessor.load_final_data()

        us_totals[2016] = df[df['CODE'] == 'USA'][column_name].iloc[0]

        df = df[df['CODE'] != 'USA'].copy()

        foreign_totals[2016] = df[column_name].sum()

        df = df.groupby('CONTINENT_NAME').sum()[[column_name]]
        df[column_name] /= (foreign_totals[2016] / 100)
        df.rename(columns={column_name: 2016}, inplace=True)

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

        dict_df = df.to_dict()

        indices = ['Sales to the US (billion USD)', 'Sales abroad (billion USD)']

        for year in [2016, 2017, 2018, 2019]:
            dict_df[year][indices[0]] = us_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        df.sort_values(by=[2019, 2018, 2017, 2016], ascending=False, inplace=True)

        if formatted:

            for year in [2016, 2017, 2018, 2019]:
                df[year] = df[year].map('{:,.1f}'.format)

        df.index = indices + [f'Of which {continent} (%)' for continent in df.index[2:]]

        return df.copy()

    def get_intermediary_dataframe_1(self, include_macro_indicator, verbose=False):

        irs = self.irs.copy()

        columns_of_interest = ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']

        if include_macro_indicator:

            merged_df = irs.merge(
                self.macro_indicator[['COUNTRY_CODE', f'{self.macro_indicator_prefix}_{self.year}']].copy(),
                how='left',
                left_on='CODE', right_on='COUNTRY_CODE'
            )

            if verbose:
                print(
                    merged_df[f'{self.macro_indicator_prefix}_{self.year}'].isnull().sum(),
                    'foreign partner countries are eliminated because we lack the macroeconomic indicator for them.'
                )

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

            merged_df = merged_df[~merged_df[f'{self.macro_indicator_prefix}_{self.year}'].isnull()].copy()

            columns_of_interest.append(f'{self.macro_indicator_prefix}_{self.year}')

        else:

            merged_df = irs.copy()

        merged_df = merged_df[merged_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

        new_columns = []

        for column in columns_of_interest:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            merged_df[new_column] = merged_df[column] / merged_df[column].sum() * 100

        return merged_df.copy()

    def get_table_2_a(self, formatted=True):

        merged_df = self.get_intermediary_dataframe_1(include_macro_indicator=False)

        output = merged_df[
            ['AFFILIATE_COUNTRY_NAME', 'UNRELATED_PARTY_REVENUES', 'SHARE_OF_UNRELATED_PARTY_REVENUES']
        ].sort_values(
            by='UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        output['UNRELATED_PARTY_REVENUES'] /= 10**9

        if formatted:

            for column in ['UNRELATED_PARTY_REVENUES', 'SHARE_OF_UNRELATED_PARTY_REVENUES']:
                output[column] = output[column].map('{:.1f}'.format)

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

    def get_table_2_b(self, formatted=True, verbose=False):

        merged_df = self.get_intermediary_dataframe_1(include_macro_indicator=True, verbose=verbose)

        output = merged_df[
            [
                'AFFILIATE_COUNTRY_NAME', 'SHARE_OF_UNRELATED_PARTY_REVENUES',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ]
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        if formatted:

            for column in ['SHARE_OF_UNRELATED_PARTY_REVENUES', f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}']:
                output[column] = output[column].map('{:.1f}'.format)

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

    def plot_figure_1(self, kind, save_PNG=False, path_to_folder=None):

        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        merged_df = self.get_intermediary_dataframe_1(include_macro_indicator=True)

        if kind == 'regplot':

            plot_df = merged_df.dropna().copy()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            plot_df['Category'] = (
                plot_df['CODE'].isin(self.tax_haven_country_codes) * 1
                + plot_df['CODE'].isin(['CAN', 'MEX']) * 2
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven', 2: 'NAFTA member'})

            plot_df[
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ] = plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'].astype(float)
            plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'] = plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'].astype(float)

            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'],
                plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )[1, 0]

            comment = (
                f'Correlation between unrelated-party revenues and {self.macro_indicator_name} '
                + f'in {self.year}: {round(correlation, 2)}'
            )

            plt.rcParams.update(
                {
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18,
                    'legend.fontsize': 18
                }
            )

            plt.figure(figsize=(17, 10))

            col_name_init = f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            col_name_new = f'Share of total {self.year} {self.macro_indicator_name} (%)'
            plot_df.rename(
                columns={
                    col_name_init: col_name_new,
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
                },
                inplace=True
            )

            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                ci=None
            )

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

            plt.title(comment)

            if save_PNG:
                plt.savefig(
                    os.path.join(
                        path_to_folder,
                        f'figure_1_{self.year}_US_only{"_GNI" if self.macro_indicator_prefix == "GNI" else ""}.png'
                    ),
                    bbox_inches='tight'
                )

            plt.show()

        else:

            merged_df['IS_TAX_HAVEN'] = merged_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes)

            plot_df = merged_df.dropna().copy()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

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

    def get_table_4_intermediary(self):

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

    def get_table_4(self, sales_type='unrelated', formatted=True):

        us_totals = {}
        foreign_totals = {}

        # Determining the revenue variable on which we are focusing
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

        dict_df = df.to_dict()

        indices = ['Sales to the US (billion USD)', 'Sales abroad (billion USD)']

        for year in [2016, 2017, 2018, 2019]:
            dict_df[year][indices[0]] = us_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        df.sort_values(by=[2019, 2018, 2017, 2016], ascending=False, inplace=True)

        if formatted:

            for year in [2016, 2017, 2018, 2019]:
                df[year] = df[year].map('{:,.1f}'.format)

        df.index = indices + [f'Of which {continent} (%)' for continent in df.index[2:]]

        return df.copy()

    def get_intermediary_dataframe_2(self, include_macro_indicator):

        sales_mapping = self.sales_mapping.copy()

        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        if include_macro_indicator:

            sales_mapping = sales_mapping.merge(
                self.macro_indicator[
                    ['COUNTRY_CODE', 'COUNTRY_NAME', f'{self.macro_indicator_prefix}_{self.year}']
                ].copy(),
                how='left',
                left_on='OTHER_COUNTRY_CODE', right_on='COUNTRY_CODE'
            )

            sales_mapping.drop(columns='OTHER_COUNTRY_CODE', inplace=True)

            sales_mapping[
                f'{self.macro_indicator_prefix}_{self.year}'
            ] = sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            sales_mapping[
                f'{self.macro_indicator_prefix}_{self.year}'
            ] = sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].astype(float)

            sales_mapping = sales_mapping[~sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].isnull()].copy()

        sales_mapping = sales_mapping[sales_mapping['COUNTRY_CODE'] != 'USA'].copy()

        new_columns = []

        for column in [
            'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES',
            'TOTAL_REVENUES', f'{self.macro_indicator_prefix}_{self.year}'
        ]:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            sales_mapping[new_column] = sales_mapping[column] / sales_mapping[column].sum() * 100

        return sales_mapping.copy()

    def get_table_5(self, formatted=True):

        merged_df = self.get_intermediary_dataframe_2(include_macro_indicator=True)

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

        output['UNRELATED_PARTY_REVENUES'] /= 10**9

        if formatted:

            for column in output.columns[1:]:
                output[column] = output[column].map('{:.1f}'.format)

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

    def plot_figure_2(self, kind, save_PNG=False, path_to_folder=None):

        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        merged_df = self.get_intermediary_dataframe_2(include_macro_indicator=True)

        if kind == 'regplot':

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['COUNTRY_NAME'] != 'United States'].copy()

            plot_df['Category'] = (
                plot_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1
                + plot_df['COUNTRY_CODE'].isin(['CAN', 'MEX']) * 2
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven', 2: 'NAFTA member'})

            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'].astype(float),
                plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'].astype(float)
            )[1, 0]

            comment = (
                'Correlation between unrelated-party revenues and '
                + f'{self.macro_indicator_name} in {self.year}: {round(correlation, 2)}'
            )

            plt.rcParams.update(
                {
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18,
                    'legend.fontsize': 18
                }
            )

            plt.figure(figsize=(17, 10))

            col_name_init = f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            col_name_new = f'Share of total {self.year} {self.macro_indicator_name} (%)'

            plot_df.rename(
                columns={
                    col_name_init: col_name_new,
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)'
                },
                inplace=True
            )

            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of total unrelated-party revenues (%)',
                data=plot_df,
                ci=None
            )

            sns.scatterplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of total unrelated-party revenues (%)',
                data=plot_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=100
            )

            plt.title(comment)

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

        else:

            merged_df['IS_TAX_HAVEN'] = merged_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['COUNTRY_NAME'] != 'United States'].copy()

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

    def get_comparison_dataframe(self):

        irs = self.irs.copy()

        sales_mapping = self.sales_mapping.copy()
        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        irs = irs[['AFFILIATE_COUNTRY_NAME', 'CODE', 'UNRELATED_PARTY_REVENUES']].copy()

        merged_df = irs.merge(
            sales_mapping[['OTHER_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES']],
            how='inner',
            left_on='CODE', right_on='OTHER_COUNTRY_CODE'
        )

        merged_df.drop(columns=['OTHER_COUNTRY_CODE'], inplace=True)

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

    def get_focus_on_tax_havens(self):

        merged_df = self.get_comparison_dataframe()

        restricted_df = merged_df[
            merged_df['COUNTRY_CODE'].isin(
                self.tax_haven_country_codes
            )
        ].copy()

        restricted_df.sort_values(by='UPR_IRS', ascending=False, inplace=True)

        restricted_df.reset_index(drop=True, inplace=True)

        dict_df = restricted_df.to_dict()

        dict_df[restricted_df.columns[0]][len(restricted_df)] = 'Total for tax havens'
        dict_df[restricted_df.columns[1]][len(restricted_df)] = '..'
        dict_df[restricted_df.columns[2]][len(restricted_df)] = restricted_df['UPR_IRS'].sum()
        dict_df[restricted_df.columns[3]][len(restricted_df)] = restricted_df['UPR_ADJUSTED'].sum()

        restricted_df = pd.DataFrame.from_dict(dict_df)

        restricted_df['SHARE_OF_UPR_IRS'] = restricted_df['UPR_IRS'] / merged_df['UPR_IRS'].sum() * 100
        restricted_df['SHARE_OF_UPR_ADJUSTED'] = restricted_df['UPR_ADJUSTED'] / merged_df['UPR_ADJUSTED'].sum() * 100

        for column in ['UPR_IRS', 'UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column] / 10**6
            # restricted_df[column] = restricted_df[column].map(round)

        for column in ['SHARE_OF_UPR_IRS', 'SHARE_OF_UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column].map(
                lambda x: round(x, 3)
            )

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

    def plot_focus_on_tax_havens(self, orient='horizontal', save_PNG=False, path_to_folder=None):

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        if orient == 'horizontal':
            figsize = (12, 12)
            ascending = False

        elif orient == 'vertical':
            figsize = (12, 8)
            ascending = True

        else:
            raise Exception('Orientation of the graph can only be "horizontal" or "vertical".')

        df = self.get_focus_on_tax_havens()

        df['Change in unrelated-party revenues (%)'] = (df[df.columns[2]] / df[df.columns[1]] - 1) * 100

        if (np.abs(df[df.columns[-1]]) >= 100).sum() > 0:
            for _, row in df[np.abs(df[df.columns[-1]]) >= 100].iterrows():
                print(
                    row['Country name'], '-', row['Unrelated-party revenues based on IRS ($m)'],
                    '-', row['Adjusted unrelated-party revenues ($m)'], '-', row[df.columns[-1]]
                )

        df = df[np.abs(df[df.columns[-1]]) < 100].copy()

        df_sorted = df.sort_values(
            by=df.columns[-1],
            ascending=ascending
        ).copy()

        plt.rcParams.update(
            {
                'axes.titlesize': 20,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18
            }
        )

        plt.figure(figsize=figsize)

        y_pos = np.arange(len(df_sorted))
        colors = df_sorted[df_sorted.columns[-1]].map(
            lambda x: 'darkred' if x < 0 else 'darkblue'
        )

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

        else:
            plt.bar(
                y_pos,
                df_sorted[df_sorted.columns[-1]],
                color=colors
            )

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

            file_name_suffix = 'v'

        plt.tight_layout()

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

    def get_table_6(self, country_code, formatted=True):

        if country_code == 'BEL':
            country_name = 'Belgium'

        elif country_code == 'LBN':
            country_name = 'Lebanon'

        elif country_code == 'NLD':
            country_name = 'the Netherlands'

        else:
            country_name = country_code

        sales_mapping = self.sales_mapping.copy()

        focus = sales_mapping[sales_mapping['OTHER_COUNTRY_CODE'] == country_code].copy()

        focus = focus.groupby(
            'AFFILIATE_COUNTRY_CODE'
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        trade_statistics = self.trade_statistics.copy()
        trade_statistics_extract = trade_statistics[trade_statistics['OTHER_COUNTRY_CODE'] == country_code].copy()
        trade_statistics_extract = trade_statistics_extract[
            ['AFFILIATE_COUNTRY_CODE', 'EXPORT_PERC']
        ].drop_duplicates(
        ).copy()

        focus = focus.merge(
            trade_statistics_extract,
            how='left',
            on='AFFILIATE_COUNTRY_CODE'
        )

        focus['UNRELATED_PARTY_REVENUES'] /= 10**6
        focus['UNRELATED_PARTY_REVENUES'] = focus['UNRELATED_PARTY_REVENUES'].map(lambda x: round(x, 1))

        if formatted:
            focus['EXPORT_PERC'] = (focus['EXPORT_PERC'] * 100).map('{:.2f}'.format)
            focus['EXPORT_PERC'] = focus['EXPORT_PERC'].map(lambda x: '..' if x == 'nan' else x)
        else:
            focus['EXPORT_PERC'] *= 100
            focus['EXPORT_PERC'] = focus['EXPORT_PERC'].map(lambda x: '..' if np.isnan(x) else x)

        focus.rename(
            columns={
                'AFFILIATE_COUNTRY_CODE': 'Affiliate jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (million USD)',
                'EXPORT_PERC': f'Share of {country_name} in exports (%)'
            },
            inplace=True
        )

        focus = focus.sort_values(
            by='Unrelated-party revenues (million USD)',
            ascending=False
        ).head(10).reset_index(drop=True)

        print(
            'Note that the export percentages are computed including the US in the destinations. '
            + 'Export percentages actually used in the computations (that exclude the US from the set of destinations) '
            + 'are therefore higher by, say, a few percentage points (except for the US themselves).'
        )

        return focus.copy()

    def get_country_profile(self, country_code):

        year = self.year

        # ### Collecting the necessary data

        sales_mapping = self.sales_mapping.copy()

        irs_sales = self.irs.copy()

        trade_statistics = self.trade_statistics.copy()

        bea_preprocessor = ExtendedBEADataLoader(year=year, load_data_online=self.load_data_online)
        bea = bea_preprocessor.get_extended_sales_percentages()

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

        sales_origin = sales_mapping[
            sales_mapping['OTHER_COUNTRY_CODE'] == country_code
        ].sort_values(
            by='UNRELATED_PARTY_REVENUES',
            ascending=False
        )

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
        year,
        aamne_domestic_sales_perc,
        breakdown_threshold,
        US_merchandise_exports_source,
        US_services_exports_source,
        non_US_merchandise_exports_source,
        non_US_services_exports_source,
        winsorize_export_percs,
        US_winsorizing_threshold=0.5,
        non_US_winsorizing_threshold=0.5,
        service_flows_to_exclude=None,
        macro_indicator='CONS',
        load_data_online=False
    ):

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

        tax_havens = pd.read_csv(self.path_to_tax_haven_list)
        self.tax_haven_country_codes = list(tax_havens['CODE'].unique()) + ['UKI']

        self.breakdown_threshold = breakdown_threshold
        cbcr_preprocessor = CbCRPreprocessor(
            year=year,
            breakdown_threshold=breakdown_threshold,
            load_data_online=load_data_online
        )
        self.oecd = cbcr_preprocessor.get_preprocessed_revenue_data()

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

    def get_GNI_data(self):
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

    def get_consumption_expenditure_data(self):
        df = pd.read_csv(self.path_to_UNCTAD_data, encoding='latin')

        df = df.reset_index()
        df.columns = df.iloc[0]
        df = df.iloc[2:].copy()

        df = df.rename(
            columns={
                np.nan: 'COUNTRY_NAME',
                'YEAR': 'ITEM'
            }
        ).reset_index(drop=True)

        df = df[df['ITEM'].map(lambda x: x.strip()) == 'Final consumption expenditure'].copy()

        list_of_years = ['2016', '2017', '2018', '2019', '2020']
        df = df[df[list_of_years].sum(axis=1) != '_' * len(list_of_years)].copy()

        for col in list_of_years:
            df[col] = df[col].astype(float)

        df = df.drop(columns='ITEM')

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

        df = df.rename(
            columns={
                'CODE': 'COUNTRY_CODE',
                str(self.year): f'CONS_{str(self.year)}'
            }
        )

        return df.copy()

    def get_table_with_relevant_parents(self):
        df = self.oecd.copy()

        df = df.groupby(
            ['PARENT_COUNTRY_CODE', 'PARENT_COUNTRY_NAME']
        ).nunique(
        )[
            'AFFILIATE_COUNTRY_CODE'
        ].reset_index()

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

        table_methodology['Parent country'] = table_methodology['Parent country'].map(
            lambda country_name: 'China' if country_name == "China (People's Republic of)" else country_name
        )

        return table_methodology.reset_index(drop=True)

    def get_table_1(self, formatted=True, sales_type='unrelated'):

        sales_type_correspondence = {
            'unrelated': 'UNRELATED_PARTY_REVENUES',
            'related': 'RELATED_PARTY_REVENUES',
            'total': 'TOTAL_REVENUES'
        }

        column_name = sales_type_correspondence[sales_type.lower()]

        domestic_totals = {}
        foreign_totals = {}

        preprocessor = CbCRPreprocessor(
            year=2016,
            breakdown_threshold=self.breakdown_threshold,
            load_data_online=self.load_data_online
        )
        df = preprocessor.get_preprocessed_revenue_data()

        domestic_totals[2016] = df[df['PARENT_COUNTRY_CODE'] == df['AFFILIATE_COUNTRY_CODE']][column_name].sum()

        df = df[df['PARENT_COUNTRY_CODE'] != df['AFFILIATE_COUNTRY_CODE']].copy()

        foreign_totals[2016] = df[column_name].sum()

        df = df.groupby('CONTINENT_CODE').sum()[[column_name]]
        df[column_name] /= (foreign_totals[2016] / 100)
        df.rename(columns={column_name: 2016}, inplace=True)

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

        dict_df = df.to_dict()

        indices = ['Domestic sales (billion USD)', 'Foreign sales (billion USD)']

        for year in [2016, 2017]:
            dict_df[year][indices[0]] = domestic_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

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

        if formatted:

            for year in [2016, 2017]:
                df[year] = df[year].map('{:,.1f}'.format)

        return df.iloc[:-1, ].copy()  # We exclude the row corresponding to "Other Groups" in the table we output

    def get_intermediary_dataframe_1(self, include_macro_indicator, exclude_US_from_parents):

        oecd = self.oecd.copy()

        columns_of_interest = ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']

        oecd = oecd[
            oecd['PARENT_COUNTRY_CODE'] != oecd['AFFILIATE_COUNTRY_CODE']
        ].copy()

        if exclude_US_from_parents:
            oecd = oecd[oecd['PARENT_COUNTRY_CODE'] != 'USA'].copy()

        # TEMP
        # oecd = oecd[oecd['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

        oecd = oecd.groupby(
            [
                'AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME'
            ]
        ).sum()[
            columns_of_interest
        ].reset_index()

        if include_macro_indicator:

            merged_df = oecd.merge(
                self.macro_indicator[['COUNTRY_CODE', f'{self.macro_indicator_prefix}_{self.year}']].copy(),
                how='left',
                left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
            )

            merged_df = merged_df[~merged_df[f'{self.macro_indicator_prefix}_{self.year}'].isnull()]

            columns_of_interest.append(f'{self.macro_indicator_prefix}_{self.year}')

        else:

            merged_df = oecd.copy()

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

    def get_table_2_a(self, formatted=True):

        merged_df = self.get_intermediary_dataframe_1(
            include_macro_indicator=False, exclude_US_from_parents=False
        )

        output = merged_df[
            ['AFFILIATE_COUNTRY_NAME', 'UNRELATED_PARTY_REVENUES', 'SHARE_OF_UNRELATED_PARTY_REVENUES']
        ].sort_values(
            by='UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        output['UNRELATED_PARTY_REVENUES'] /= 10**9

        output.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME': 'Partner jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (USD billion)',
                'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
            },
            inplace=True
        )

        if formatted:

            for column in ['Unrelated-party revenues (USD billion)', 'Share of foreign unrelated-party revenues (%)']:
                output[column] = output[column].map('{:,.1f}'.format)

        output.reset_index(drop=True, inplace=True)

        return output.copy()

    def get_table_2_b(self, exclude_US_from_parents=False, formatted=True):

        merged_df = self.get_intermediary_dataframe_1(
            include_macro_indicator=True, exclude_US_from_parents=exclude_US_from_parents
        )

        output = merged_df[
            [
                'AFFILIATE_COUNTRY_NAME', 'SHARE_OF_UNRELATED_PARTY_REVENUES',
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ]
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        if formatted:

            for column in ['SHARE_OF_UNRELATED_PARTY_REVENUES', f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}']:
                output[column] = output[column].map('{:,.1f}'.format)

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

    def plot_figure_1(self, kind, exclude_US_from_parents, save_PNG=False, path_to_folder=None):

        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        merged_df = self.get_intermediary_dataframe_1(
            include_macro_indicator=True, exclude_US_from_parents=exclude_US_from_parents
        )

        if kind == 'regplot':

            plot_df = merged_df.dropna().copy()

            plot_df['Category'] = (
                plot_df['AFFILIATE_COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven'})

            plot_df[
                f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            ] = plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'].astype(float)
            plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'] = plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'].astype(float)

            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'],
                plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )[1, 0]

            comment = (
                'Correlation between unrelated-party revenues and '
                + f'{self.macro_indicator_name} in {self.year}: {round(correlation, 2)}'
            )

            plt.rcParams.update(
                {
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18,
                    'legend.fontsize': 18
                }
            )

            plt.figure(figsize=(17, 10))

            col_name_init = f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            col_name_new = f'Share of total {self.year} {self.macro_indicator_name} (%)'

            plot_df.rename(
                columns={
                    col_name_init: col_name_new,
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of foreign unrelated-party revenues (%)'
                },
                inplace=True
            )

            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of foreign unrelated-party revenues (%)',
                data=plot_df,
                ci=None
            )

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

            plt.title(comment)

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

        else:

            merged_df['IS_TAX_HAVEN'] = merged_df['AFFILIATE_COUNTRY_CODE'].isin(self.tax_haven_country_codes)

            plot_df = merged_df.dropna().copy()

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

    def get_table_4_intermediary(self):

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

    def get_table_4(self, sales_type='unrelated', formatted=True):

        domestic_totals = {}
        foreign_totals = {}

        # Determining the revenue variable on which we are focusing
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

        dict_df = df.to_dict()

        indices = ['Domestic sales (billion USD)', 'Sales abroad (billion USD)']

        for year in [2016, 2017]:
            dict_df[year][indices[0]] = domestic_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        df.sort_values(by=[2017, 2016], ascending=False, inplace=True)

        if formatted:

            for year in [2016, 2017]:
                df[year] = df[year].map('{:,.1f}'.format)

        df.index = indices + [f'Of which {continent} (%)' for continent in df.index[2:]]

        return df.copy()

    def get_intermediary_dataframe_2(self, include_macro_indicator, exclude_US_from_parents):

        sales_mapping = self.sales_mapping.copy()

        sales_mapping = sales_mapping[
            sales_mapping['PARENT_COUNTRY_CODE'] != sales_mapping['OTHER_COUNTRY_CODE']
        ].copy()

        if exclude_US_from_parents:
            sales_mapping = sales_mapping[sales_mapping['PARENT_COUNTRY_CODE'] != 'USA'].copy()

        # TEMP
        # sales_mapping = sales_mapping[sales_mapping['OTHER_COUNTRY_CODE'] != 'USA'].copy()

        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        if include_macro_indicator:

            sales_mapping = sales_mapping.merge(
                self.macro_indicator[
                    ['COUNTRY_CODE', 'COUNTRY_NAME', f'{self.macro_indicator_prefix}_{self.year}']
                ].copy(),
                how='left',
                left_on='OTHER_COUNTRY_CODE', right_on='COUNTRY_CODE'
            )

            sales_mapping.drop(columns='OTHER_COUNTRY_CODE', inplace=True)

            sales_mapping[
                f'{self.macro_indicator_prefix}_{self.year}'
            ] = sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            sales_mapping[
                f'{self.macro_indicator_prefix}_{self.year}'
            ] = sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].astype(float)

            sales_mapping = sales_mapping[~sales_mapping[f'{self.macro_indicator_prefix}_{self.year}'].isnull()].copy()

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

    def plot_figure_2(self, kind, exclude_US_from_parents, save_PNG=False, path_to_folder=None):

        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        merged_df = self.get_intermediary_dataframe_2(
            include_macro_indicator=True, exclude_US_from_parents=exclude_US_from_parents
        )

        if kind == 'regplot':

            plot_df = merged_df.dropna().copy()

            plot_df['Category'] = (
                plot_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven'})

            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'].astype(float),
                plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES'].astype(float)
            )[1, 0]

            comment = (
                'Correlation between unrelated-party revenues and '
                + f'{self.macro_indicator_name} in {self.year}: {round(correlation, 2)}'
            )

            plt.rcParams.update(
                {
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18,
                    'legend.fontsize': 18
                }
            )

            plt.figure(figsize=(17, 10))

            col_name_init = f'SHARE_OF_{self.macro_indicator_prefix}_{self.year}'
            col_name_new = f'Share of total {self.year} {self.macro_indicator_name} (%)'

            plot_df.rename(
                columns={
                    col_name_init: col_name_new,
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)'
                },
                inplace=True
            )

            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of total unrelated-party revenues (%)',
                data=plot_df,
                ci=None
            )

            sns.scatterplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of total unrelated-party revenues (%)',
                data=plot_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=100
            )

            plt.title(comment)

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

        else:

            merged_df['IS_TAX_HAVEN'] = merged_df['COUNTRY_CODE'].isin(self.tax_haven_country_codes) * 1

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['COUNTRY_NAME'] != 'United States'].copy()

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

    def get_table_5(self, formatted=True):

        merged_df = self.get_intermediary_dataframe_2(
            include_macro_indicator=True, exclude_US_from_parents=False
        )

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

        output['UNRELATED_PARTY_REVENUES'] /= 10**9

        if formatted:

            for column in output.columns[1:]:
                output[column] = output[column].map('{:.1f}'.format)

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

    def get_comparison_dataframe(self):

        oecd = self.oecd.copy()
        oecd = oecd.groupby(
            ['AFFILIATE_COUNTRY_NAME', 'AFFILIATE_COUNTRY_CODE']
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        sales_mapping = self.sales_mapping.copy()
        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        merged_df = oecd.merge(
            sales_mapping[['OTHER_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES']],
            how='inner',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='OTHER_COUNTRY_CODE'
        )

        merged_df.drop(columns=['OTHER_COUNTRY_CODE'], inplace=True)

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

    def get_focus_on_tax_havens(self):

        merged_df = self.get_comparison_dataframe()

        restricted_df = merged_df[
            merged_df['COUNTRY_CODE'].isin(
                self.tax_haven_country_codes
            )
        ].copy()

        restricted_df.sort_values(by='UPR_OECD', ascending=False, inplace=True)

        restricted_df.reset_index(drop=True, inplace=True)

        dict_df = restricted_df.to_dict()

        dict_df[restricted_df.columns[0]][len(restricted_df)] = 'Total for tax havens'
        dict_df[restricted_df.columns[1]][len(restricted_df)] = '..'
        dict_df[restricted_df.columns[2]][len(restricted_df)] = restricted_df['UPR_OECD'].sum()
        dict_df[restricted_df.columns[3]][len(restricted_df)] = restricted_df['UPR_ADJUSTED'].sum()

        restricted_df = pd.DataFrame.from_dict(dict_df)

        restricted_df['SHARE_OF_UPR_OECD'] = restricted_df['UPR_OECD'] / merged_df['UPR_OECD'].sum() * 100
        restricted_df['SHARE_OF_UPR_ADJUSTED'] = restricted_df['UPR_ADJUSTED'] / merged_df['UPR_ADJUSTED'].sum() * 100

        for column in ['UPR_OECD', 'UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column] / 10**6
            # restricted_df[column] = restricted_df[column].map(round)

        for column in ['SHARE_OF_UPR_OECD', 'SHARE_OF_UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column].map(
                lambda x: round(x, 3)
            )

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

    def plot_focus_on_tax_havens(self, orient='horizontal', save_PNG=False, path_to_folder=None):

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        if orient == 'horizontal':
            figsize = (12, 12)
            ascending = False

        elif orient == 'vertical':
            figsize = (12, 8)
            ascending = True

        else:
            raise Exception('Orientation of the graph can only be "horizontal" or "vertical".')

        df = self.get_focus_on_tax_havens()

        df['Change in unrelated-party revenues (%)'] = (df[df.columns[2]] / df[df.columns[1]] - 1) * 100

        if (np.abs(df[df.columns[-1]]) >= 100).sum() > 0:
            for _, row in df[np.abs(df[df.columns[-1]]) >= 100].iterrows():
                print(
                    row['Country name'], '-', row['Unrelated-party revenues based on OECD ($m)'],
                    '-', row['Adjusted unrelated-party revenues ($m)'], '-', row[df.columns[-1]]
                )

        df = df[np.abs(df[df.columns[-1]]) < 100].copy()

        df_sorted = df.sort_values(
            by=df.columns[-1],
            ascending=ascending
        ).copy()

        plt.rcParams.update(
            {
                'axes.titlesize': 20,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18
            }
        )

        plt.figure(figsize=figsize)

        y_pos = np.arange(len(df_sorted))
        colors = df_sorted[df_sorted.columns[-1]].map(
            lambda x: 'darkred' if x < 0 else 'darkblue'
        )

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

        else:
            plt.bar(
                y_pos,
                df_sorted[df_sorted.columns[-1]],
                color=colors
            )

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

            file_name_suffix = 'v'

        plt.tight_layout()

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

    def get_table_6(self, country_code):

        if country_code == 'BEL':
            country_name = 'Belgium'

        elif country_code == 'LBN':
            country_name = 'Lebanon'

        else:
            country_name = country_code

        sales_mapping = self.sales_mapping.copy()

        focus = sales_mapping[sales_mapping['OTHER_COUNTRY_CODE'] == country_code].copy()

        focus = focus.groupby(
            'AFFILIATE_COUNTRY_CODE'
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        trade_statistics = self.trade_statistics.copy()
        trade_statistics_extract = trade_statistics[trade_statistics['OTHER_COUNTRY_CODE'] == country_code].copy()
        trade_statistics_extract = trade_statistics_extract[
            ['AFFILIATE_COUNTRY_CODE', 'EXPORT_PERC']
        ].drop_duplicates(
        ).copy()

        focus = focus.merge(
            trade_statistics_extract,
            how='left',
            on='AFFILIATE_COUNTRY_CODE'
        )

        focus['UNRELATED_PARTY_REVENUES'] /= 10**6
        focus['UNRELATED_PARTY_REVENUES'] = focus['UNRELATED_PARTY_REVENUES'].map(lambda x: round(x, 1))

        focus['EXPORT_PERC'] = (focus['EXPORT_PERC'] * 100).map('{:.2f}'.format)
        focus['EXPORT_PERC'] = focus['EXPORT_PERC'].map(lambda x: '..' if x == 'nan' else x)

        focus.rename(
            columns={
                'AFFILIATE_COUNTRY_CODE': 'Affiliate jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (million USD)',
                'EXPORT_PERC': f'Share of {country_name} in exports (%)'
            },
            inplace=True
        )

        focus = focus.sort_values(
            by='Unrelated-party revenues (million USD)',
            ascending=False
        ).head(10).reset_index(drop=True)

        return focus.copy()

    def get_country_profile(self, country_code):

        year = self.year

        # ### Collecting the necessary data

        sales_mapping = self.sales_mapping.copy()

        oecd_sales = self.oecd.copy()

        trade_statistics = self.trade_statistics.copy()

        bea_preprocessor = ExtendedBEADataLoader(year=year, load_data_online=self.load_data_online)
        bea = bea_preprocessor.get_extended_sales_percentages()

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

        sales_origin = sales_mapping[
            sales_mapping['OTHER_COUNTRY_CODE'] == country_code
        ].sort_values(
            by='UNRELATED_PARTY_REVENUES',
            ascending=False
        )

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
