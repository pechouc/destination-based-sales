import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from destination_based_sales.utils import CONTINENT_CODES_TO_IMPUTE_TRADE


path_to_dir = os.path.dirname(os.path.abspath(__file__))
path_to_GNI_data = os.path.join(path_to_dir, 'data', 'gross_national_income.csv')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')
path_to_tax_haven_list = os.path.join(path_to_dir, 'data', 'tax_havens.csv')


class PerIndustryAnalyser:

    def __init__(self, year, path_to_tax_haven_list=path_to_tax_haven_list):
        if year not in [2016, 2017, 2018]:
            raise Exception('For now, only the financial years from 2016 to 2018 (included) are covered.')

        self.year = year

        self.path_to_tax_haven_list = path_to_tax_haven_list
        self.tax_havens = pd.read_csv(self.path_to_tax_haven_list)

    def load_clean_data(
        self,
        path_to_dir=path_to_dir, path_to_geographies=path_to_geographies,
        exclude_all_jurisdictions=True
    ):

        # Loading the data from the corresponding Excel file
        path_to_industry_data = os.path.join(
            path_to_dir,
            'data',
            str(self.year),
            f'{self.year - 2000}it02cbc.xlsx'
        )

        data = pd.read_excel(
            path_to_industry_data,
            engine='openpyxl'
        )

        # Eliminating irrelevant columns and rows
        data = data[data.columns[:6]].copy()

        data.columns = [
            'INDUSTRY',
            'AFFILIATE_COUNTRY_NAME',
            'NB_REPORTING_MNEs',
            'UNRELATED_PARTY_REVENUES',
            'RELATED_PARTY_REVENUES',
            'TOTAL_REVENUES'
        ]

        data = data[
            data.isnull().sum(axis=1) != len(data.columns)
        ].copy()
        data = data.iloc[4:-7].copy()

        data.reset_index(drop=True, inplace=True)

        # Adding the right industry name to each observation
        industry_indices = list(data[~data['INDUSTRY'].isnull()].index)

        industries = {}

        for i in range(len(industry_indices)):

            if i < len(industry_indices) - 1:
                restricted_df = data.loc[industry_indices[i]:industry_indices[i + 1] - 1].copy()

            else:
                restricted_df = data.loc[industry_indices[i]:].copy()

            industry = restricted_df['INDUSTRY'].iloc[0]
            restricted_df['INDUSTRY'] = industry
            industries[industry] = restricted_df.copy()

        data = industries[list(industries.keys())[0]].copy()

        for key, value in industries.items():
            if key == list(industries.keys())[0]:
                continue

            data = pd.concat([data, value], axis=0)

        # Eliminating irrelevant observations
        if exclude_all_jurisdictions:
            data = data[
                ~data['AFFILIATE_COUNTRY_NAME'].isin(['All jurisdictions', 'Stateless entities and other country'])
            ].copy()

        else:
            data = data[
                data['AFFILIATE_COUNTRY_NAME'] != 'Stateless entities and other country'
            ].copy()

        data = data[
            ~data['AFFILIATE_COUNTRY_NAME'].map(
                lambda country_name: 'total' in country_name.lower()
            )
        ].copy()

        data = data[
            data.drop(columns=['NB_REPORTING_MNEs']).applymap(
                lambda x: isinstance(x, str) and x == 'd'
            ).sum(axis=1) == 0
        ].copy()

        # Renaming continental aggregates and a few specific partner jurisdictions
        data['AFFILIATE_COUNTRY_NAME'] = data['AFFILIATE_COUNTRY_NAME'].map(
            (
                lambda country_name: f'Other {country_name.split(",")[0].replace("&", "and")}'
                if 'other' in country_name.lower() else country_name
            )
        )

        data['AFFILIATE_COUNTRY_NAME'] = data['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'United Kingdom' if 'United Kingdom' in country_name else country_name
        )
        data['AFFILIATE_COUNTRY_NAME'] = data['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'Korea' if country_name.startswith('Korea') else country_name
        )
        data['AFFILIATE_COUNTRY_NAME'] = data['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'Congo' if country_name.endswith('(Brazzaville)') else country_name
        )

        # Adding alpha-3 country codes
        geographies = pd.read_csv(path_to_geographies)

        data = data.merge(
            geographies[['NAME', 'CODE']],
            how='left',
            left_on='AFFILIATE_COUNTRY_NAME', right_on='NAME'
        )

        data.drop(columns=['NAME'], inplace=True)

        data['CODE'] = data.apply(
            (
                lambda row: 'OASIAOCN' if isinstance(row['CODE'], float) and np.isnan(row['CODE'])
                and row['AFFILIATE_COUNTRY_NAME'] == 'Other Asia and Oceania' else row['CODE']
            ),
            axis=1
        )

        data.rename(
            columns={
                'CODE': 'AFFILIATE_COUNTRY_CODE'
            },
            inplace=True
        )

        data.reset_index(drop=True, inplace=True)

        # Simplifying the sector denominations
        industry_names_mapping = {
            'Agriculture, forestry, fishing and hunting, mining, quarrying, oil and gas extraction, utilities, and construction': 'Agriculture, extractives and construction',
            'Wholesale and retail trade, transportation and warehousing ': 'Wholesale and retail trade',
            'Finance and insurance, real estate and rental and leasing': 'Finance and insurance',
            'Professional, scientific, and technical services': 'Technical services',
            'Management of companies and enterprises, all other services (except public administration)': 'Management (except public administration)'
        }

        data['INDUSTRY'] = data['INDUSTRY'].map(
            lambda industry: industry_names_mapping.get(industry, industry)
        )

        return data.copy()

    def load_data_with_GNI(self, dropna=False, path_to_GNI_data=path_to_GNI_data):

        data = self.load_clean_data()

        gross_national_income = pd.read_csv(path_to_GNI_data, delimiter=';')
        gross_national_income = gross_national_income[['COUNTRY_CODE', f'GNI_{self.year}']].copy()

        gross_national_income[f'GNI_{self.year}'] = gross_national_income[f'GNI_{self.year}'].map(
            lambda x: x.replace(',', '.') if isinstance(x, str) else x
        ).astype(float)

        data = data.merge(
            gross_national_income,
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

        data.drop(columns=['COUNTRY_CODE'], inplace=True)

        if dropna:
            data.dropna(inplace=True)

        return data.copy()

    def get_industry_overview_table(self, output_excel=True):

        final_output = {}

        for year in [2016, 2017, 2018]:
            analyser = PerIndustryAnalyser(year=year)

            data = analyser.load_clean_data(exclude_all_jurisdictions=False)

            data = data[data['AFFILIATE_COUNTRY_NAME'].isin(['All jurisdictions', 'United States'])].copy()

            data.drop(
                columns=[
                    'AFFILIATE_COUNTRY_CODE', 'NB_REPORTING_MNEs',
                    'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES'
                ],
                inplace=True
            )

            df = data.pivot(
                index='INDUSTRY',
                columns='AFFILIATE_COUNTRY_NAME',
                values='UNRELATED_PARTY_REVENUES'
            ).reset_index()

            df['FOREIGN_UPR'] = df['All jurisdictions'] - df['United States']

            df['Share of total unrelated-party revenues (%)'] = (
                df['All jurisdictions'] / df['All jurisdictions'].sum() * 100
            )
            df['Share of foreign unrelated-party revenues (%)'] = df['FOREIGN_UPR'] / df['FOREIGN_UPR'].sum() * 100

            df.drop(columns=['All jurisdictions', 'United States', 'FOREIGN_UPR'], inplace=True)

            df.sort_values(by='Share of foreign unrelated-party revenues (%)', ascending=False, inplace=True)

            final_output[year] = df.copy()

        if output_excel:
            path_to_excel_file = '/Users/Paul-Emmanuel/Desktop/industry_overview_table_PYTHON_OUTPUT.xlsx'

            with pd.ExcelWriter(path_to_excel_file, engine='xlsxwriter') as writer:
                for key, value in final_output.items():
                    value.to_excel(writer, sheet_name=str(key), index=False)

        return final_output.copy()

    def plot_industry_specific_charts(self, save_PNG=False, path_to_folder=None):
        plt.rcParams.update({'font.size': 18})

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        data = self.load_data_with_GNI(dropna=True)

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(25, 40))

        for industry, ax in zip(data['INDUSTRY'].unique(), axes.flatten()):

            restricted_df = data[
                np.logical_and(
                    data['INDUSTRY'] == industry,
                    data['AFFILIATE_COUNTRY_CODE'] != 'USA'
                )
            ].copy()

            restricted_df['SHARE_OF_UNRELATED_PARTY_REVENUES'] = (
                restricted_df['UNRELATED_PARTY_REVENUES'].astype(float) /
                restricted_df['UNRELATED_PARTY_REVENUES'].sum()
            )

            restricted_df[f'SHARE_OF_GNI_{self.year}'] = (
                restricted_df[f'GNI_{self.year}'] / restricted_df[f'GNI_{self.year}'].sum()
            )

            correlation = np.corrcoef(
                restricted_df['SHARE_OF_UNRELATED_PARTY_REVENUES'],
                restricted_df[f'SHARE_OF_GNI_{self.year}']
            )[1, 0]

            restricted_df['Category'] = (
                restricted_df['AFFILIATE_COUNTRY_CODE'].isin(self.tax_havens['CODE'].unique()) * 1
                + restricted_df['AFFILIATE_COUNTRY_CODE'].isin(['CAN', 'MEX']) * 2
            )
            restricted_df['Category'] = restricted_df['Category'].map({0: 'Other', 1: 'Tax haven', 2: 'NAFTA member'})

            restricted_df.rename(
                columns={
                    f'SHARE_OF_GNI_{self.year}': f'Share of total {self.year} GNI (%)',
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)'
                },
                inplace=True
            )

            sns.regplot(
                x=f'Share of total {self.year} GNI (%)',
                y='Share of total unrelated-party revenues (%)',
                data=restricted_df,
                ci=None,
                ax=ax
            )


            sns.scatterplot(
                x=f'Share of total {self.year} GNI (%)',
                y='Share of total unrelated-party revenues (%)',
                data=restricted_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=80,
                ax=ax
            )

            ax.set_title(f'{industry} - Correlation of {round(correlation, 2)}')

        plt.show()

        if save_PNG:
            fig.savefig(
                os.path.join(
                    path_to_folder,
                    f'industry_specific_charts_{self.year}.png'
                )
            )
