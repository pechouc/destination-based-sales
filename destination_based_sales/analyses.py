import os

import numpy as np
import pandas as pd

from irs import IRSDataPreprocessor
from revenue_split import RevenueSplitter
from trade_statistics import TradeStatisticsProcessor

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_GNI_data = os.path.join(path_to_dir, 'data', 'gross_national_income.csv')
path_to_tax_haven_list = os.path.join(path_to_dir, 'data', 'tax_havens.csv')


class SalesCalculator:

    def __init__(self):

        self.splitter = RevenueSplitter()
        self.splitted_revenues = self.splitter.get_splitted_revenues()

        self.trade_stat_processor = TradeStatisticsProcessor()
        self.trade_statistics = self.trade_stat_processor.load_data_with_imputations()

        missing_overlap = (
            ~splitted_revenues['CODE'].isin(
                trade_statistics['AFFILIATE_COUNTRY_CODE'].unique()
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
            columns=existing_columns + ['EXPORT_PERC'],
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

    def get_final_dataframe():

        merged_df = self.get_sales_to_other_foreign_countries()
        affiliate_country_sales = self.get_sales_to_affiliate_country()
        us_sales = self.get_sales_to_the_US()

        output_df = pd.concat(
            [merged_df, affiliate_country_sales, us_sales],
            axis=0
        )

        return output_df.copy()


class AnalysisProvider:

    def __init__(
        self,
        path_to_GNI_data=path_to_GNI_data, path_to_tax_haven_list=path_to_tax_haven_list
    ):

        self.path_to_GNI_data = path_to_GNI_data
        self.gross_national_income = pd.read_csv(self.path_to_GNI_data)

        self.path_to_tax_haven_list = path_to_tax_haven_list
        self.tax_havens = pd.read_csv(self.path_to_tax_haven_list)

    def get_intermediary_dataframe_1(self):

        irs_preprocessor = IRSDataPreprocessor()

        irs = irs_preprocessor.load_final_data()

        merged_df = irs.merge(
            self.gross_national_income,
            how='left',
            left_on='CODE', right_on='COUNTRY_CODE'
        )

        merged_df['GNI_2018'] = merged_df['GNI_2018'].map(
            lambda x: x.replace(',', '.') if isinstance(x, str) else x
        )

        merged_df['GNI_2018'] = merged_df['GNI_2018'].astype(float)

        merged_df = merged_df[merged_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

        new_columns = []

        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES', 'GNI_2018']:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            merged_df[new_column] = merged_df[column] / merged_df[column].sum() * 100

        return merged_df.copy()

    def get_table_1(self):

        merged_df = self.get_intermediary_dataframe_1()

        output = merged_df[
            ['AFFILIATE_COUNTRY_NAME', 'SHARE_OF_UNRELATED_PARTY_REVENUES', 'SHARE_OF_GNI_2018']
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        output.reset_index(drop=True, inplace=True)

        return output.copy()

    def plot_figure_1(self, kind):

        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        merged_df = self.get_intermediary_dataframe_1()

        if kind == 'regplot':

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            correlation = np.corrcoef(
                plot_df['SHARE_OF_GNI_2018'], plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )[1, 0]

            print(f'Correlation between unrelated-party revenues and GNI in 2018: {round(correlation, 5)}')

            plt.figure(figsize=(17, 10))

            sns.regplot(
                x=plot_df['SHARE_OF_GNI_2018'], y=plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )

            plt.show()

        else:

            merged_df = merged_df.merge(
                tax_havens,
                how='left',
                on='CODE'
            )

            merged_df['IS_TAX_HAVEN'] = merged_df['IS_TAX_HAVEN'].fillna(0)

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            if kind == 'scatter':

                plt.figure(figsize=(12, 7))

                sns.scatterplot(
                    x='SHARE_OF_GNI_2018',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    hue='IS_TAX_HAVEN',
                    data=plot_df
                )

                plt.show()

            else:

                colors = plot_df['IS_TAX_HAVEN'].map(
                    lambda x: 'blue' if x == 0 else 'red'
                )

                px.scatter(
                    x='SHARE_OF_GNI_2018',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    color=colors,
                    data_frame=plot_df,
                    hover_name='AFFILIATE_COUNTRY_NAME'
                )

    def get_intermediary_dataframe_2(self):

        calculator = SalesCalculator()

        sales_mapping = calculator.get_final_dataframe()

        sales_mapping = sales_mapping.groupby('OTHER_COUNTRY_CODE').sum().reset_index()

        sales_mapping = sales_mapping[sales_mapping['OTHER_COUNTRY_CODE'] != 'USA'].copy()

        sales_mapping = sales_mapping.merge(
            self.gross_national_income,
            how='left',
            left_on='OTHER_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

        sales_mapping.drop(columns='OTHER_COUNTRY_CODE', inplace=True)

        sales_mapping['GNI_2018'] = sales_mapping['GNI_2018'].map(
            lambda x: x.replace(',', '.') if isinstance(x, str) else x
        )

        sales_mapping['GNI_2018'] = sales_mapping['GNI_2018'].astype(float)

        new_columns = []

        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES', 'GNI_2018']:
            new_column = 'SHARE_OF_' + column

            new_columns.append(new_column)

            sales_mapping[new_column] = sales_mapping[column] / sales_mapping[column].sum() * 100

        return sales_mapping.copy()

    def get_table_2(self):

        merged_df = self.get_intermediary_dataframe_2()

        output = merged_df[
            ['AFFILIATE_COUNTRY_NAME', 'SHARE_OF_UNRELATED_PARTY_REVENUES', 'SHARE_OF_GNI_2018']
        ].sort_values(
            by='SHARE_OF_UNRELATED_PARTY_REVENUES',
            ascending=False
        ).head(20)

        output.reset_index(drop=True, inplace=True)

        return output.copy()

    def plot_figure_2(self, kind):

        if kind not in ['regplot', 'scatter', 'interactive']:
            raise Exception(
                'The "kind" argument can only take the following values: "regplot", "scatter" and "interactive".'
            )

        merged_df = self.get_intermediary_dataframe_2()

        if kind == 'regplot':

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            correlation = np.corrcoef(
                plot_df['SHARE_OF_GNI_2018'], plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )[1, 0]

            print(f'Correlation between unrelated-party revenues and GNI in 2018: {round(correlation, 5)}')

            plt.figure(figsize=(17, 10))

            sns.regplot(
                x=plot_df['SHARE_OF_GNI_2018'], y=plot_df['SHARE_OF_UNRELATED_PARTY_REVENUES']
            )

            plt.show()

        else:

            merged_df = merged_df.merge(
                tax_havens,
                how='left',
                left_on='COUNTRY_CODE', right_on='CODE'
            )

            merged_df['IS_TAX_HAVEN'] = merged_df['IS_TAX_HAVEN'].fillna(0)

            merged_df.drop(columns=['CODE'], inplace=True)

            plot_df = merged_df.dropna()
            plot_df = plot_df[plot_df['AFFILIATE_COUNTRY_NAME'] != 'United States'].copy()

            if kind == 'scatter':

                plt.figure(figsize=(12, 7))

                sns.scatterplot(
                    x='SHARE_OF_GNI_2018',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    hue='IS_TAX_HAVEN',
                    data=plot_df
                )

                plt.show()

            else:

                colors = plot_df['IS_TAX_HAVEN'].map(
                    lambda x: 'blue' if x == 0 else 'red'
                )

                px.scatter(
                    x='SHARE_OF_GNI_2018',
                    y='SHARE_OF_UNRELATED_PARTY_REVENUES',
                    color=colors,
                    data_frame=plot_df,
                    hover_name='COUNTRY_NAME'
                )
