"""
This module is dedicated to the industry-specific analyses described in Section 4.a. of the PDF report of August 2021.
Indeed, the US Internal Revenue Service (IRS) provides a breakdown of its country-by-country statistics based on the
main sector of activity of the parent company of the multinational group. We use it to highlight the differences, from
an industry to another, in the concentration of foreign sales in tax havens.
"""


########################################################################################################################
# --- Imports

import os
import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from destination_based_sales.utils import CONTINENT_CODES_TO_IMPUTE_TRADE


########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))
path_to_GNI_data = os.path.join(path_to_dir, 'data', 'gross_national_income.csv')
path_to_CONS_data = os.path.join(path_to_dir, 'data', 'us_gdpcomponent_98866982281181.csv')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')
path_to_tax_haven_list = os.path.join(path_to_dir, 'data', 'tax_havens.csv')
path_to_industry_names_mapping = os.path.join(path_to_dir, 'data', 'industry_names_mapping.json')

with open(path_to_industry_names_mapping) as file:
    industry_names_mapping = json.loads(file.read())


########################################################################################################################
# --- Content

class PerIndustryAnalyser:

    def __init__(
        self,
        year,
        macro_indicator,
        path_to_dir=path_to_dir,
        path_to_tax_haven_list=path_to_tax_haven_list,
        path_to_geographies=path_to_geographies
    ):
        """
        The logic for loading, preprocessing and analysing the industry-specific country-by-country statistics of the
        IRS is encapsulated in a Python class, PerIndustryAnalyser. This is the instantiation function of this class,
        which takes as arguments:

        - the year to consider;
        - the string path to the directory where this Python file is located;
        - the string path to the list of tax havens;
        - the string path to the "geographies.csv" file.
        """
        if year not in [2016, 2017, 2018, 2019]:
            raise Exception('For now, only the financial years from 2016 to 2019 (included) are covered.')

        self.year = year

        self.macro_indicator = macro_indicator

        if macro_indicator == 'CONS':
            self.macro_indicator_name = 'consumption expenditures'

        elif macro_indicator == 'GNI':
            self.macro_indicator_name = 'GNI'

        else:
            raise Exception(
                'Macroeconomic indicator can either be Gross National Income (pass "macro_indicator=GNI" as argument) '
                + 'or the UNCTAD consumption expenditure indicator (pass "macro_indicator=CONS" as argument).'
            )

        # We load the list of tax havens in a dedicated attribute
        self.path_to_tax_haven_list = path_to_tax_haven_list
        self.tax_havens = pd.read_csv(self.path_to_tax_haven_list)

        self.path_to_dir = path_to_dir
        self.path_to_geographies = path_to_geographies

    def load_clean_data(
        self,
        exclude_all_jurisdictions=True
    ):
        """
        This function allows to load and preprocess the industry-specific country-by-country data of the IRS.

        It takes as argument a boolean, "exclude_all_jurisdictions", that determines whether or not to exclude the in-
        dustry-level totals from the dataset. These are characterised by "All jurisdictions" as a partner country.
        """

        # Loading the data from the corresponding Excel file
        path_to_industry_data = os.path.join(
            self.path_to_dir,
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
        # industry_indices = list(data[~data['INDUSTRY'].isnull()].index)

        # industries = {}

        # for i in range(len(industry_indices)):

        #     if i < len(industry_indices) - 1:
        #         restricted_df = data.loc[industry_indices[i]:industry_indices[i + 1] - 1].copy()

        #     else:
        #         restricted_df = data.loc[industry_indices[i]:].copy()

        #     industry = restricted_df['INDUSTRY'].iloc[0]
        #     restricted_df['INDUSTRY'] = industry
        #     industries[industry] = restricted_df.copy()

        # data = industries[list(industries.keys())[0]].copy()

        # for key, value in industries.items():
        #     if key == list(industries.keys())[0]:
        #         continue

        #     data = pd.concat([data, value], axis=0)
        data['INDUSTRY'] = data['INDUSTRY'].ffill()

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
        geographies = pd.read_csv(self.path_to_geographies)

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

        # Renaming industries for convenience
        data['INDUSTRY'] = data['INDUSTRY'].map(
            lambda industry: industry_names_mapping.get(industry, industry)
        )

        return data.copy()

    def load_data_with_GNI(self, dropna=False, path_to_GNI_data=path_to_GNI_data):
        """
        Building upon the previous method, "load_clean_data", this method allows to load and preprocess the industry-
        specific country-by-country data while adding the Gross National Income (GNI) of each partner country, for the
        corresponding year. It takes two arguments:

        - a boolean, "dropna", indicating whether or not to exclude the partner countries for which we lack the GNI;
        - the string path to the file containing GNI data.
        """

        # Loading and cleaning industry-specific country-by-country data
        data = self.load_clean_data()

        # Loading and preprocessing Gross National Income (GNI) data
        gross_national_income = pd.read_csv(path_to_GNI_data, delimiter=';')
        gross_national_income = gross_national_income[['COUNTRY_CODE', f'GNI_{self.year}']].copy()

        gross_national_income[f'GNI_{self.year}'] = gross_national_income[f'GNI_{self.year}'].map(
            lambda x: x.replace(',', '.') if isinstance(x, str) else x
        ).astype(float)

        # Merging the two datasets on partner country codes
        data = data.merge(
            gross_national_income,
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

        data.drop(columns=['COUNTRY_CODE'], inplace=True)

        if dropna:
            data.dropna(inplace=True)

        return data.copy()

    def load_data_with_CONS(self, dropna=False, path_to_CONS_data=path_to_CONS_data):
        """
        Building upon the previous method, "load_clean_data", this method allows to load and preprocess the industry-
        specific country-by-country data while adding the final consumption expenditures of each partner country, for
        the corresponding year. It takes two arguments:

        - a boolean, "dropna", indicating whether or not to exclude the partner countries for which we lack the GNI;
        - the string path to the file containing consumption expenditure data.
        """

        # Loading and cleaning industry-specific country-by-country data
        data = self.load_clean_data()

        df = pd.read_csv(path_to_CONS_data, encoding='latin')

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

        df = df.rename(
            columns={
                'CODE': 'COUNTRY_CODE',
                str(self.year): f'CONS_{str(self.year)}'
            }
        )

        df = df[['COUNTRY_CODE', f'CONS_{str(self.year)}']].copy()

        # Merging the two datasets on partner country codes
        data = data.merge(
            df,
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='COUNTRY_CODE'
        )

        data.drop(columns=['COUNTRY_CODE'], inplace=True)

        if dropna:
            data.dropna(inplace=True)

        return data.copy()

    def get_industry_overview_table(self, output_excel=True, path_to_folder='/Users/Paul-Emmanuel/Desktop/'):
        """
        This method allows to output the industry overview table that shows, for each year in the sample period, the
        distribution of US total unrelated-party revenues and foreign unrelated-party revenues between industries. It
        corresponds to Table 3 in the PDF report of August 2021. The boolean argument, "output_excel", determines
        whether to save the table in an Excel file (change the target file path before using this method with
        "output_excel=True").
        """
        final_output = {}

        for year in [2016, 2017, 2018, 2019]:
            # We instantiate an industry-specific analyser for each year
            analyser = PerIndustryAnalyser(year=year, macro_indicator=self.macro_indicator)

            # Loading the data
            data = analyser.load_clean_data(exclude_all_jurisdictions=False)

            # Focusing on industry totals and US-US rows
            data = data[data['AFFILIATE_COUNTRY_NAME'].isin(['All jurisdictions', 'United States'])].copy()

            # Eliminating irrelevant columns
            data.drop(
                columns=[
                    'AFFILIATE_COUNTRY_CODE', 'NB_REPORTING_MNEs',
                    'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES'
                ],
                inplace=True
            )

            # We pivot the DataFrame to show the revenues of each industry in all jurisdictions and in the US
            df = data.pivot(
                index='INDUSTRY',
                columns='AFFILIATE_COUNTRY_NAME',
                values='UNRELATED_PARTY_REVENUES'
            ).reset_index()

            # Foreign unrelated-party revenues simply correspond to the total minus the US-US revenues
            df['FOREIGN_UPR'] = df['All jurisdictions'] - df['United States']

            # We move from absolute amounts to shares / a distribution
            df['Share of total unrelated-party revenues (%)'] = (
                df['All jurisdictions'] / df['All jurisdictions'].sum() * 100
            )
            df['Share of foreign unrelated-party revenues (%)'] = df['FOREIGN_UPR'] / df['FOREIGN_UPR'].sum() * 100

            df.drop(columns=['All jurisdictions', 'United States', 'FOREIGN_UPR'], inplace=True)

            # Ranking industries based on decreasing importance in the distribution
            df.sort_values(by='Share of foreign unrelated-party revenues (%)', ascending=False, inplace=True)

            final_output[year] = df.copy()

        # Outputting the Excel file if relevant
        if output_excel:
            path_to_excel_file = path_to_folder + 'industry_overview_table_PYTHON_OUTPUT.xlsx'

            with pd.ExcelWriter(path_to_excel_file, engine='xlsxwriter') as writer:
                for key, value in final_output.items():
                    value.to_excel(writer, sheet_name=str(key), index=False)

        return final_output.copy()

    def get_industry_overview_table_simplified(self):
        # Loading the data
        data = self.load_clean_data(exclude_all_jurisdictions=False)

        # Focusing on industry totals and US-US rows
        data = data[data['AFFILIATE_COUNTRY_NAME'].isin(['All jurisdictions', 'United States'])].copy()

        # Eliminating irrelevant columns
        data.drop(
            columns=[
                'AFFILIATE_COUNTRY_CODE', 'NB_REPORTING_MNEs',
                'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES'
            ],
            inplace=True
        )

        # We pivot the DataFrame to show the revenues of each industry in all jurisdictions and in the US
        df = data.pivot(
            index='INDUSTRY',
            columns='AFFILIATE_COUNTRY_NAME',
            values='UNRELATED_PARTY_REVENUES'
        ).reset_index()

        # Foreign unrelated-party revenues simply correspond to the total minus the US-US revenues
        df['FOREIGN_UPR'] = df['All jurisdictions'] - df['United States']

        # We move from absolute amounts to shares / a distribution
        df['Share of total unrelated-party revenues (%)'] = (
            df['All jurisdictions'] / df['All jurisdictions'].sum() * 100
        )
        df['Share of foreign unrelated-party revenues (%)'] = df['FOREIGN_UPR'] / df['FOREIGN_UPR'].sum() * 100

        df.drop(columns=['All jurisdictions', 'United States', 'FOREIGN_UPR'], inplace=True)

        # Ranking industries based on decreasing importance in the distribution
        df.sort_values(by='Share of foreign unrelated-party revenues (%)', ascending=False, inplace=True)

        return df.rename(columns={'INDUSTRY': 'Industry'}).reset_index(drop=True)


    def plot_industry_specific_charts(self, verbose=False, save_PNG=False, path_to_folder=None):
        """
        This method allows to output the graphs that show the relationship between partner jurisdictions’ share of US
        multinational companies’ foreign unrelated-party revenues and their share of Gross National Income (GNI),
        broken down by industry group. These correspond to Figure E.1 of the PDF report of August 2021.

        The method takes two arguments used to save the output charts in a PNG file. This requires to set the boolean
        "save_PNG" to True and to pass, in "path_to_folder", the string path to the target folder.
        """

        # Setting Matplotlib parameters
        plt.rcParams.update({'font.size': 18})

        if save_PNG and path_to_folder is None:
            raise Exception('To save the figure as a PNG, you must indicate the target folder as an argument.')

        if self.macro_indicator == 'GNI':
            # Loading cleaned data with GNI data (eliminating rows for which we have no GNI data)
            data = self.load_data_with_GNI(dropna=True)
        else:
            # Loading cleaned data with consumption expenditure data (eliminating rows for which we have no data)
            data = self.load_data_with_CONS(dropna=True)

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(25, 40))

        # Figure displays one graph per industry group
        for industry, ax in zip(data['INDUSTRY'].unique(), axes.flatten()):

            # Restricting the dataset to the industry group under consideration and excluding the US-US row
            restricted_df = data[
                np.logical_and(
                    data['INDUSTRY'] == industry,
                    data['AFFILIATE_COUNTRY_CODE'] != 'USA'
                )
            ].copy()

            # Computing each partner country's share of US, industry-specific foreign unrelated-party revenues
            restricted_df['SHARE_OF_UNRELATED_PARTY_REVENUES'] = (
                restricted_df['UNRELATED_PARTY_REVENUES'].astype(float) /
                restricted_df['UNRELATED_PARTY_REVENUES'].sum()
            ) * 100

            # Computing each partner country's share of the macroeconomic indicator
            restricted_df[f'SHARE_OF_{self.macro_indicator}_{self.year}'] = (
                restricted_df[f'{self.macro_indicator}_{self.year}']
                / restricted_df[f'{self.macro_indicator}_{self.year}'].sum()
            ) * 100

            if industry == 'Information' and verbose:
                temp = restricted_df[restricted_df['AFFILIATE_COUNTRY_CODE'] == 'IRL'].copy()
                idx = temp.index[0]
                print(
                    'Share of Ireland in INFORMATION unrelated-party revenues:',
                    temp.loc[idx, 'SHARE_OF_UNRELATED_PARTY_REVENUES']
                )
                print(
                    'Corresponding share of the macroeconomic indicator:',
                    temp.loc[idx, f'SHARE_OF_{self.macro_indicator}_{self.year}']
                )

            # Computing the correlation between the two shares for the industry under consideration
            correlation = np.corrcoef(
                restricted_df['SHARE_OF_UNRELATED_PARTY_REVENUES'],
                restricted_df[f'SHARE_OF_{self.macro_indicator}_{self.year}']
            )[1, 0]

            # Distinguishing non-havens, tax havens and NAFTA members
            restricted_df['Category'] = (
                restricted_df['AFFILIATE_COUNTRY_CODE'].isin(self.tax_havens['CODE'].unique()) * 1
                + restricted_df['AFFILIATE_COUNTRY_CODE'].isin(['CAN', 'MEX']) * 2
            )
            restricted_df['Category'] = restricted_df['Category'].map({0: 'Other', 1: 'Tax haven', 2: 'NAFTA member'})

            restricted_df.rename(
                columns={
                    f'SHARE_OF_{self.macro_indicator}_{self.year}': f'Share of total {self.year} {self.macro_indicator_name} (%)',
                    'SHARE_OF_UNRELATED_PARTY_REVENUES': 'Share of total unrelated-party revenues (%)'
                },
                inplace=True
            )

            # Building the graph with the indicative regression line and the scattered plot
            sns.regplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of total unrelated-party revenues (%)',
                data=restricted_df,
                ci=None,
                ax=ax
            )

            sns.scatterplot(
                x=f'Share of total {self.year} {self.macro_indicator_name} (%)',
                y='Share of total unrelated-party revenues (%)',
                data=restricted_df,
                hue='Category',
                palette={
                    'Other': 'darkblue', 'Tax haven': 'darkred', 'NAFTA member': 'darkgreen'
                },
                s=80,
                ax=ax
            )

            # Title indicating the industry being considered and the correlation between the share of foreign unrelated-
            # party revenues and the share of GNI
            ax.set_title(f'{industry} - Correlation of {round(correlation, 2)}')

        axes.flatten()[-1].set_axis_off()

        plt.show()

        # Saving the figure into a PNG file if relevant
        if save_PNG:
            fig.savefig(
                os.path.join(
                    path_to_folder,
                    f'industry_specific_charts_{self.year}.png'
                )
            )
