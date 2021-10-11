import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.revenue_split import RevenueSplitter
from destination_based_sales.trade_statistics import TradeStatisticsProcessor
from destination_based_sales.per_industry import PerIndustryAnalyser


path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_GNI_data = os.path.join(path_to_dir, 'data', 'gross_national_income.csv')
path_to_tax_haven_list = os.path.join(path_to_dir, 'data', 'tax_havens.csv')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class SalesCalculator:

    def __init__(self, year, include_US=True):

        self.irs_preprocessor = IRSDataPreprocessor(year=year)
        self.irs = self.irs_preprocessor.load_final_data()

        self.splitter = RevenueSplitter(year=year, include_US=include_US)
        self.splitted_revenues = self.splitter.get_splitted_revenues()

        self.trade_stat_processor = TradeStatisticsProcessor(
            year=year,
            winsorize_export_percs=True,
            US_only=True
        )
        self.trade_statistics = self.trade_stat_processor.load_data_with_imputations()

        missing_overlap = (
            ~self.splitted_revenues['CODE'].isin(
                self.trade_statistics['AFFILIATE_COUNTRY_CODE'].unique()
            )
        ).sum()

        if missing_overlap != 0:
            raise Exception(
                'It seems that some countries in the splitted revenues are not covered by trade statistics yet.'
            )

    def get_sales_to_other_foreign_countries(self):

        other_country_sales = self.splitted_revenues[
            [
                'AFFILIATE_COUNTRY_NAME', 'CODE', 'UNRELATED_PARTY_REVENUES_TO_OTHER_COUNTRY',
                'RELATED_PARTY_REVENUES_TO_OTHER_COUNTRY', 'TOTAL_REVENUES_TO_OTHER_COUNTRY'
            ]
        ].copy()

        trade_statistics = self.trade_statistics.copy()

        merged_df = trade_statistics.merge(
            other_country_sales,
            how='inner',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        merged_df.drop(columns='CODE', inplace=True)

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
            columns=existing_columns + ['ALL_EXPORTS'],
            inplace=True
        )

        return merged_df.copy()

    def get_sales_to_affiliate_country(self):

        affiliate_country_sales = self.splitted_revenues[
            [
                'AFFILIATE_COUNTRY_NAME', 'CODE', 'UNRELATED_PARTY_REVENUES_TO_AFFILIATE_COUNTRY',
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

        us_sales = self.splitted_revenues[
            [
                'AFFILIATE_COUNTRY_NAME', 'CODE', 'UNRELATED_PARTY_REVENUES_TO_US',
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

    def get_final_dataframe(self):

        merged_df = self.get_sales_to_other_foreign_countries()
        affiliate_country_sales = self.get_sales_to_affiliate_country()
        us_sales = self.get_sales_to_the_US()

        output_df = pd.concat(
            [merged_df, affiliate_country_sales, us_sales],
            axis=0
        )

        output_df = output_df[output_df[output_df.columns[-3:]].sum(axis=1) > 0].copy()

        return output_df.copy()


class AnalysisProvider:

    def __init__(
        self,
        year,
        include_US,
        path_to_GNI_data=path_to_GNI_data,
        path_to_tax_haven_list=path_to_tax_haven_list,
        path_to_geographies=path_to_geographies
    ):

        self.year = year

        self.path_to_GNI_data = path_to_GNI_data
        self.gross_national_income = pd.read_csv(self.path_to_GNI_data, delimiter=';')

        self.path_to_tax_haven_list = path_to_tax_haven_list
        self.tax_havens = pd.read_csv(self.path_to_tax_haven_list)

        self.path_to_geographies = path_to_geographies

        print('Running computations - This may take around 10 seconds or slightly more.')

        irs_preprocessor = IRSDataPreprocessor(year=year)
        self.irs = irs_preprocessor.load_final_data()

        calculator = SalesCalculator(year=year, include_US=include_US)
        self.sales_mapping = calculator.get_final_dataframe()

        print('Computations finalized - Results are stored as attributes.')

    def get_table_1(self, formatted=True, sales_type='unrelated'):

        sales_type_correspondence = {
            'unrelated': 'UNRELATED_PARTY_REVENUES',
            'related': 'RELATED_PARTY_REVENUES',
            'total': 'TOTAL_REVENUES'
        }

        column_name = sales_type_correspondence[sales_type.lower()]

        us_totals = {}
        foreign_totals = {}

        preprocessor = IRSDataPreprocessor(year=2016)
        df = preprocessor.load_final_data()

        us_totals[2016] = df[df['CODE'] == 'USA'][column_name].iloc[0]

        df = df[df['CODE'] != 'USA'].copy()

        foreign_totals[2016] = df[column_name].sum()

        df = df.groupby('CONTINENT_NAME').sum()[[column_name]]
        df[column_name] /= (foreign_totals[2016] / 100)
        df.rename(columns={column_name: 2016}, inplace=True)

        for year in [2017, 2018]:

            preprocessor = IRSDataPreprocessor(year=year)
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

        for year in [2016, 2017, 2018]:
            dict_df[year][indices[0]] = us_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        if formatted:
            df.sort_values(by=[2018, 2017, 2016], ascending=False, inplace=True)

            for year in [2016, 2017, 2018]:
                df[year] = df[year].map('{:,.1f}'.format)

            df.index = indices + [f'Of which {continent} (%)' for continent in df.index[2:]]

        return df.copy()

    def get_intermediary_dataframe_1(self, include_GNI):

        irs = self.irs.copy()

        columns_of_interest = ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']

        if include_GNI:

            merged_df = irs.merge(
                self.gross_national_income[['COUNTRY_CODE', f'GNI_{self.year}']].copy(),
                how='left',
                left_on='CODE', right_on='COUNTRY_CODE'
            )

            merged_df[f'GNI_{self.year}'] = merged_df[f'GNI_{self.year}'].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            merged_df = merged_df[~merged_df[f'GNI_{self.year}'].isnull()]

            merged_df[f'GNI_{self.year}'] = merged_df[f'GNI_{self.year}'].astype(float)

            columns_of_interest.append(f'GNI_{self.year}')

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

        merged_df = self.get_intermediary_dataframe_1(include_GNI=False)

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
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)'
                },
                inplace=True
            )

        output.reset_index(drop=True, inplace=True)

        return output.copy()

    def get_table_2_b(self, formatted=True):

        merged_df = self.get_intermediary_dataframe_1(include_GNI=True)

        output = merged_df[
            ['AFFILIATE_COUNTRY_NAME', 'SHARE_OF_UNRELATED_PARTY_REVENUES', f'SHARE_OF_GNI_{self.year}']
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        if formatted:

            for column in ['SHARE_OF_UNRELATED_PARTY_REVENUES', f'SHARE_OF_GNI_{self.year}']:
                output[column] = output[column].map('{:.1f}'.format)

            output.rename(
                columns={
                    'AFFILIATE_COUNTRY_NAME': 'Partner jurisdiction',
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)',
                    f'SHARE_OF_GNI_{self.year}': 'Share of Gross National Income (%)'
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

        merged_df = self.get_intermediary_dataframe_1(include_GNI=True)

        if kind == 'regplot':

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            plot_df['Category'] = (
                plot_df['CODE'].isin(self.tax_havens['CODE'].unique()) * 1
                + plot_df['CODE'].isin(['CAN', 'MEX']) * 2
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven', 2: 'NAFTA member'})

            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_GNI_{self.year}'], plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )[1, 0]

            print(f'Correlation between unrelated-party revenues and GNI in 2018: {round(correlation, 5)}')

            plt.figure(figsize=(17, 10))

            plot_df.rename(
                columns={
                    f'SHARE_OF_GNI_{self.year}': f'Share of total {self.year} GNI (%)',
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)'
                },
                inplace=True
            )

            sns.regplot(
                x=f'Share of total {self.year} GNI (%)',
                y='Share of total unrelated-party revenues (%)',
                data=plot_df,
                ci=None
            )

            sns.scatterplot(
                x=f'Share of total {self.year} GNI (%)',
                y='Share of total unrelated-party revenues (%)',
                data=plot_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=100
            )

            if save_PNG:
                plt.savefig(
                    os.path.join(path_to_folder, f'figure_1_{self.year}.png')
                )

            plt.show()

        else:

            merged_df = merged_df.merge(
                self.tax_havens,
                how='left',
                on='CODE'
            )

            merged_df['IS_TAX_HAVEN'] = merged_df['IS_TAX_HAVEN'].fillna(0)

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            if kind == 'scatter':

                plt.figure(figsize=(12, 7))

                sns.scatterplot(
                    x=f'SHARE_OF_GNI_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    hue='IS_TAX_HAVEN',
                    data=plot_df
                )

                plt.show()

                if save_PNG:
                    raise Exception('The option to save the figure as a PNG is only available for the regplot.')

            else:

                colors = plot_df['IS_TAX_HAVEN'].map(
                    lambda x: 'blue' if x == 0 else 'red'
                )

                fig = px.scatter(
                    x=f'SHARE_OF_GNI_{self.year}',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    color=colors,
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
        sales_mapping.drop(columns=['EXPORT_PERC'], inplace=True)

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
        analyser = AnalysisProvider(year=2016, include_US=True)
        df = analyser.get_table_4_intermediary()

        us_totals[2016] = df[df['OTHER_COUNTRY_CODE'] == 'USA'][column_name].iloc[0]

        df = df[df['OTHER_COUNTRY_CODE'] != 'USA'].copy()

        foreign_totals[2016] = df[column_name].sum()

        df = df.groupby('CONTINENT_NAME').sum()[[column_name]]
        df[column_name] /= (foreign_totals[2016] / 100)
        df.rename(columns={column_name: 2016}, inplace=True)

        for year in [2017, 2018]:

            analyser = AnalysisProvider(year=year, include_US=True)
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

        for year in [2016, 2017, 2018]:
            dict_df[year][indices[0]] = us_totals[year] / 10**9
            dict_df[year][indices[1]] = foreign_totals[year] / 10**9

        df = pd.DataFrame.from_dict(dict_df)

        if formatted:
            df.sort_values(by=[2018, 2017, 2016], ascending=False, inplace=True)

            for year in [2016, 2017, 2018]:
                df[year] = df[year].map('{:,.1f}'.format)

            df.index = indices + [f'Of which {continent} (%)' for continent in df.index[2:]]

        return df.copy()

    def get_intermediary_dataframe_2(self, include_GNI):

        sales_mapping = self.sales_mapping.copy()

        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        if include_GNI:

            sales_mapping = sales_mapping.merge(
                self.gross_national_income[['COUNTRY_CODE', 'COUNTRY_NAME', f'GNI_{self.year}']].copy(),
                how='left',
                left_on='OTHER_COUNTRY_CODE', right_on='COUNTRY_CODE'
            )

            sales_mapping.drop(columns='OTHER_COUNTRY_CODE', inplace=True)

            sales_mapping[f'GNI_{self.year}'] = sales_mapping[f'GNI_{self.year}'].map(
                lambda x: x.replace(',', '.') if isinstance(x, str) else x
            )

            sales_mapping[f'GNI_{self.year}'] = sales_mapping[f'GNI_{self.year}'].astype(float)

            sales_mapping = sales_mapping[~sales_mapping[f'GNI_{self.year}'].isnull()].copy()

        sales_mapping = sales_mapping[sales_mapping['COUNTRY_CODE'] != 'USA'].copy()

        new_columns = []

        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES', f'GNI_{self.year}']:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            sales_mapping[new_column] = sales_mapping[column] / sales_mapping[column].sum() * 100

        return sales_mapping.copy()

    def get_table_5(self, formatted=True):

        merged_df = self.get_intermediary_dataframe_2(include_GNI=True)

        output = merged_df[
            [
                'COUNTRY_NAME',
                'UNRELATED_PARTY_REVENUES',
                'SHARE_OF_UNRELATED_PARTY_REVENUES',
                f'SHARE_OF_GNI_{self.year}'
            ]
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        output.reset_index(drop=True, inplace=True)

        if formatted:
            output['UNRELATED_PARTY_REVENUES'] /= 10**9

            for column in output.columns[1:]:
                output[column] = output[column].map('{:.1f}'.format)

            output.rename(
                columns={
                    'COUNTRY_NAME': 'Partner jurisdiction',
                    'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (USD billion)',
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)',
                    f'SHARE_OF_GNI_{self.year}': 'Share of Gross National Income (%)'
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

        merged_df = self.get_intermediary_dataframe_2(include_GNI=True)

        if kind == 'regplot':

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['COUNTRY_NAME'] != 'United States'].copy()

            plot_df['Category'] = (
                plot_df['COUNTRY_CODE'].isin(self.tax_havens['CODE'].unique()) * 1
                + plot_df['COUNTRY_CODE'].isin(['CAN', 'MEX']) * 2
            )
            plot_df['Category'] = plot_df['Category'].map({0: 'Other', 1: 'Tax haven', 2: 'NAFTA member'})

            correlation = np.corrcoef(
                plot_df[f'SHARE_OF_GNI_{self.year}'], plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )[1, 0]

            print(f'Correlation between unrelated-party revenues and GNI in 2018: {round(correlation, 5)}')

            plt.figure(figsize=(17, 10))

            plot_df.rename(
                columns={
                    f'SHARE_OF_GNI_{self.year}': f'Share of total {self.year} GNI (%)',
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)'
                },
                inplace=True
            )

            sns.regplot(
                x=f'Share of total {self.year} GNI (%)',
                y='Share of total unrelated-party revenues (%)',
                data=plot_df,
                ci=None
            )

            sns.scatterplot(
                x=f'Share of total {self.year} GNI (%)',
                y='Share of total unrelated-party revenues (%)',
                data=plot_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=100
            )

            if save_PNG:
                plt.savefig(
                    os.path.join(path_to_folder, f'figure_2_{self.year}.png')
                )

            plt.show()

        else:

            merged_df = merged_df.merge(
                self.tax_havens,
                how='left',
                left_on='COUNTRY_CODE', right_on='CODE'
            )

            merged_df['IS_TAX_HAVEN'] = merged_df['IS_TAX_HAVEN'].fillna(0)

            merged_df.drop(columns=['CODE'], inplace=True)

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['COUNTRY_NAME'] != 'United States'].copy()

            if kind == 'scatter':

                plt.figure(figsize=(12, 7))

                sns.scatterplot(
                    x=f'SHARE_OF_GNI_{self.year}',
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
                    x=f'SHARE_OF_GNI_{self.year}',
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
                list(self.tax_havens['CODE']) + ['UKI']
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
            restricted_df[column] = restricted_df[column].map(round)

        for column in ['SHARE_OF_UPR_IRS', 'SHARE_OF_UPR_ADJUSTED']:
            restricted_df[column] = restricted_df[column].map(
                lambda x: round(x, 3)
            )

        restricted_df.rename(
            columns={
                'COUNTRY_NAME': 'Country name',
                'UPR_IRS': 'Unrelated-party revenues based on IRS ($m)',
                'UPR_ADJUSTED': 'Adjusted unrelated-party revenues ($m)',
                'SHARE_OF_UPR_IRS': 'Share of URP based on IRS (%)',
                'SHARE_OF_UPR_ADJUSTED': 'Share of adjusted URP (%)'
            },
            inplace=True
        )

        restricted_df.drop(columns=['COUNTRY_CODE'], inplace=True)

        return restricted_df.copy()

    def plot_focus_on_tax_havens(self, save_PNG=False, path_to_folder=None):

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        df = self.get_focus_on_tax_havens()

        df['Change in unrelated-party revenues (%)'] = (df[df.columns[2]] / df[df.columns[1]] - 1) * 100

        df_sorted = df.sort_values(
            by=df.columns[-1],
            ascending=False
        ).copy()

        plt.figure(figsize=(12, 12))

        y_pos = np.arange(len(df_sorted))
        colors = df_sorted[df_sorted.columns[-1]].map(
            lambda x: 'darkred' if x < 0 else 'darkblue'
        )

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

        if save_PNG:
            plt.savefig(
                os.path.join(path_to_folder, f'focus_on_tax_havens_{self.year}.png')
            )

        plt.show()

    def get_table_6(self, country_code):

        if country_code == 'BEL':
            country_name = 'Belgium'

        else:
            country_name = country_code

        sales_mapping = self.sales_mapping.copy()

        focus = sales_mapping[sales_mapping['OTHER_COUNTRY_CODE'] == country_code].copy()

        focus = focus[['AFFILIATE_COUNTRY_NAME', 'EXPORT_PERC', 'UNRELATED_PARTY_REVENUES']].copy()

        focus['UNRELATED_PARTY_REVENUES'] /= 10**9
        focus['UNRELATED_PARTY_REVENUES'] = focus['UNRELATED_PARTY_REVENUES'].map(lambda x: round(x, 1))

        focus['EXPORT_PERC'] = (focus['EXPORT_PERC'] * 100).map('{:.1f}'.format)
        focus['EXPORT_PERC'] = focus['EXPORT_PERC'].map(lambda x: '..' if x == 'nan' else x)

        focus.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME': 'Partner jurisdiction',
                'UNRELATED_PARTY_REVENUES': 'Unrelated-party revenues (billion USD)',
                'EXPORT_PERC': f'Share of {country_name} in exports (%)'
            },
            inplace=True
        )

        focus = focus.sort_values(
            by='Unrelated-party revenues (billion USD)',
            ascending=False
        ).head(10).reset_index(drop=True)

        return focus.copy()


if __name__ == '__main__':
    final_output = {}

    path_to_folder = sys.argv[1]

    for year in [2016, 2017, 2018]:

        analyser = AnalysisProvider(year=year, include_US=True)
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
