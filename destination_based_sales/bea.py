"""
This module is used to load and preprocess data from the Bureau of Economic Analysis (BEA). These allow to split revenue
variables between sales directed to the host (or affiliate) country, to the US and to any third country. Data are loaded
from Excel files saved in the "data" folder.
"""


########################################################################################################################
# --- Imports

# General imports
import os

import numpy as np
import pandas as pd

# Imports for the BEADataPreprocessor class
from destination_based_sales.utils import CODES_TO_IMPUTE_BEA, impute_missing_codes

# Imports used through the ExtendedBEADataLoader class
from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor

from destination_based_sales.utils import eliminate_irrelevant_percentages, impute_missing_values

########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


########################################################################################################################
# --- Content

class BEADataPreprocessor:

    def __init__(
        self,
        year,
        path_to_dir=path_to_dir,
        path_to_geo_file=path_to_geographies
    ):
        """
        The instructions allowing to load and preprocess BEA data are organised in a Python class, BEADataPreprocessor.

        This is the instantiation function for this class. It requires several arguments:

        - the year to consider (for now, one of 2016, 2017 or 2018);
        - the path to the directory where this Python file is located, to retrieve the appropriate data file;
        - the path to the "geographies.csv" file, used for instance to complement BEA data with country codes.
        """
        self.year = year

        # We construct the path to the relevant data file, which depends on the year considered
        self.path_to_bea = os.path.join(
            path_to_dir,
            'data',
            str(year),
            'Part-II-E1-E17.xls'
        )

        self.path_to_geo_file = path_to_geo_file

        self.CODES_TO_IMPUTE = CODES_TO_IMPUTE_BEA.copy()

    def load_data(self):
        """
        This class method is used to load and clean the data from the BEA. It relies on the data file paths, saved as
        class attributes when the instantiation function is called. Preprocessing steps are detailed in comments below.
        """

        # We load the data from the appropriate Excel file
        bea = pd.read_excel(self.path_to_bea, sheet_name='Table II.E 2')

        # We rename columns, the following column names being used throughout the code
        bea.columns = [
            'AFFILIATE_COUNTRY_NAME', 'TOTAL', 'TOTAL_US', 'TOTAL_US_RELATED', 'TOTAL_US_UNRELATED', 'TOTAL_FOREIGN',
            'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_AFFILIATE_COUNTRY_RELATED', 'TOTAL_AFFILIATE_COUNTRY_UNRELATED',
            'TOTAL_OTHER_COUNTRY', 'TOTAL_OTHER_COUNTRY_RELATED', 'TOTAL_OTHER_COUNTRY_UNRELATED'
        ]

        # We only keep relevant rows
        bea = bea.loc[8:].copy()

        bea = bea[~(bea.isnull().sum(axis=1) >= 11)].copy()

        bea = bea.iloc[:-2].copy()

        bea = bea[bea['AFFILIATE_COUNTRY_NAME'] != 'Latin America and Other Western Hemisphere'].copy()

        # We re-index the DataFrame after having filtered out inappropriate rows
        bea.reset_index(inplace=True, drop=True)

        # Due to the organisation of the Excel file, the DataFrame contains rows that only display the name of the con-
        # tinent associated with countries below; we want to eliminate these rows and reconstitute a "one-block" dataset
        continent_names = [
            'Europe',
            'South America',
            'Central America',
            'Other Western Hemisphere',
            'Africa',
            'Middle East',
            'Asia and Pacific',
        ]

        # We fetch the list of the indices of these rows
        total_indices = list(
            bea[
                bea['AFFILIATE_COUNTRY_NAME'].isin(continent_names)
            ].index
        )

        # We will store the sub-DataFrames associated with each continent in a dedicated dictionary
        continent_extracts = {}

        for i, continent_name in enumerate(continent_names):
            if i + 1 < len(total_indices):
                continent_df = bea.loc[total_indices[i]:total_indices[i + 1] - 1].copy()

            else:
                continent_df = bea.loc[total_indices[i]:bea.index[-1]].copy()

            # In each sub-DataFrame, we rename the "Other" row as "Other [+ CONTINENT NAME]"
            continent_df['AFFILIATE_COUNTRY_NAME'] = continent_df['AFFILIATE_COUNTRY_NAME'].map(
                lambda country_name: country_name if country_name != 'Other' else 'Other ' + continent_name
            )

            continent_df = continent_df[continent_df['AFFILIATE_COUNTRY_NAME'] != continent_name].copy()

            continent_extracts[continent_name] = continent_df.copy()

        # The Canada row is outside any continent block
        bea_cleaned = bea[bea['AFFILIATE_COUNTRY_NAME'] == 'Canada'].copy()

        # Upon it, we stack the different continent blocks to obtain one "continuous" dataset
        for continent_extract in continent_extracts.values():
            bea_cleaned = pd.concat([bea_cleaned, continent_extract], axis=0)

        # We eventually reformat missing values
        for column in bea_cleaned.columns[1:]:
            bea_cleaned[column] = bea_cleaned[column].map(
                lambda x: np.nan if x == '(D)' else x
            )

        return bea_cleaned.reset_index(drop=True)

    def load_data_with_geo_codes(self):
        """
        This class method is used to add geographical ISO codes to the raw dataset, loaded with the "load_data" method.
        It relies on the "impute_missing_codes" function defined in utils.py.
        """
        bea = self.load_data()

        geographies = pd.read_csv(self.path_to_geo_file)

        # We merge the DataFrame containing raw BEA data with the one containing the ISO code correspondences
        merged_df = bea.merge(
            geographies,
            how='left',
            left_on='AFFILIATE_COUNTRY_NAME', right_on='NAME'
        )

        # We add missing codes
        for column in ['NAME', 'CODE', 'CONTINENT_NAME', 'CONTINENT_CODE']:
            merged_df[column] = merged_df.apply(
                lambda row: impute_missing_codes(
                    row=row,
                    column=column,
                    codes_to_impute=self.CODES_TO_IMPUTE
                ),
                axis=1
            )

        # We don't consider "Other" aggregates as they are not the same as in the IRS data
        merged_df = merged_df[~merged_df['CODE'].isnull()].copy()
        merged_df = merged_df[merged_df['CODE'].map(len) <= 3].copy()

        return merged_df.copy()

    def load_final_data(self):
        """
        This class method allows to load the fully preprocessed BEA data. Relying on the "load_data_with_geo_codes" me-
        thod, continent names and codes are limited to 4 pairs, corresponding respectively to Europe, Africa, America
        (North and South) and Asia-Pacific (gathering Asia and Oceania).
        """
        bea = self.load_data_with_geo_codes()

        bea['CONTINENT_CODE'] = bea['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        bea['CONTINENT_NAME'] = bea['CONTINENT_NAME'].map(
            lambda x: 'America' if x in ['South America', 'North America'] else x
        )

        bea['CONTINENT_CODE'] = bea['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        bea['CONTINENT_NAME'] = bea['CONTINENT_NAME'].map(
            lambda x: 'Asia-Pacific' if x in ['Asia', 'Oceania'] or x is None else x
        )

        return bea.copy()


class ExtendedBEADataLoader:

    def __init__(
        self,
        year,
        path_to_dir=path_to_dir
    ):

        self.year = year

        self.path_to_dir = path_to_dir

        # Instantiating an object of the class BEADataPreprocessor defined above
        self.bea_preprocessor = BEADataPreprocessor(year=year)

        # Defining the complete set of affiliate countries to cover eventually
        if year in [2016, 2017]:
            oecd_preprocessor = CbCRPreprocessor(year=year)
            oecd = oecd_preprocessor.get_preprocessed_revenue_data()

            self.target_countries = oecd[
                ['AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME', 'CONTINENT_CODE']
            ].drop_duplicates()

        else:
            irs_preprocessor = IRSDataPreprocessor(year=year)
            irs = irs_preprocessor.load_final_data()

            self.target_countries = irs[
                ['CODE', 'AFFILIATE_COUNTRY_NAME', 'CONTINENT_CODE']
            ].drop_duplicates(
            ).rename(
                columns={
                    'CODE': 'AFFILIATE_COUNTRY_CODE'
                }
            )

    def load_data_with_US_US_row(self):
        df = self.bea_preprocessor.load_final_data()

        path_to_BEA_KR_tables = os.path.join(
            self.path_to_dir,
            'data',
            str(self.year),
            'Part-I-K1-R2.xls'
        )

        temp = pd.read_excel(path_to_BEA_KR_tables, sheet_name='Table I.O 1')

        # Cleaning and reorganising the table
        column_names = temp.loc[4].to_dict().copy()
        column_names[list(column_names.keys())[0]] = 'Industry'
        column_names['Unnamed: 1'] = 'Total'

        temp.rename(columns=column_names, inplace=True)

        # Saving the relevant information in a dictionary
        us_sales = temp.loc[6].to_dict()

        us_imputation = {}

        # Imputing the BEA-like distribution of US-US sales into the merged DataFrame
        # Sales to the affiliate country and to the headquarter country are directed to the same final destination
        for column in df.columns[1:12]:

            if column == 'TOTAL':
                us_imputation[column] = [us_sales['Total']]

            elif 'US' in column:
                us_imputation[column] = [us_sales['To U.S. persons'] * 0]

            elif 'AFFILIATE_COUNTRY' in column:
                us_imputation[column] = [us_sales['To U.S. persons'] * 1]

            else:
                us_imputation[column] = [us_sales['To foreign affiliates'] + us_sales['To other foreign persons']]

        us_imputation['AFFILIATE_COUNTRY_NAME'] = ['United States']
        us_imputation['NAME'] = ['United States']
        us_imputation['CODE'] = ['USA']
        us_imputation['CONTINENT_NAME'] = ['America']
        us_imputation['CONTINENT_CODE'] = ['AMR']

        us_imputation = pd.DataFrame(us_imputation)

        # Concatenating the two DataFrames to get the BEA data with the US-US row
        df = pd.concat([df, us_imputation], axis=0)

        return df.copy()

    def get_merged_dataframe(self):
        df = self.load_data_with_US_US_row()

        merged_df = self.target_countries.merge(
            df,
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        merged_df.drop(
            columns=['AFFILIATE_COUNTRY_NAME_y', 'NAME', 'CODE', 'CONTINENT_CODE_y', 'CONTINENT_NAME'],
            inplace=True
        )

        merged_df.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME_x': 'AFFILIATE_COUNTRY_NAME',
                'CONTINENT_CODE_x': 'CONTINENT_CODE'
            },
            inplace=True
        )

        return merged_df.copy()

    def get_data_with_indicator_variables(self):
        merged_df = self.get_merged_dataframe()

        mask_US = (merged_df['AFFILIATE_COUNTRY_CODE'] == 'USA')

        # Is the split of related-party sales complete?
        related = ['TOTAL_US_RELATED', 'TOTAL_AFFILIATE_COUNTRY_RELATED', 'TOTAL_OTHER_COUNTRY_RELATED']
        self.related = related.copy()

        mask_0 = ~merged_df[related[0]].isnull()
        mask_1 = ~merged_df[related[1]].isnull()
        mask_2 = ~merged_df[related[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        # Takes value 0 if the split is incomplete, 1 if the split is complete and 2 in the US-US case
        merged_df['IS_RELATED_COMPLETE'] = mask * 1 + mask_US * 1

        # Is the split of unrelated-party sales complete?
        unrelated = ['TOTAL_US_UNRELATED', 'TOTAL_AFFILIATE_COUNTRY_UNRELATED', 'TOTAL_OTHER_COUNTRY_UNRELATED']
        self.unrelated = unrelated.copy()

        mask_0 = ~merged_df[unrelated[0]].isnull()
        mask_1 = ~merged_df[unrelated[1]].isnull()
        mask_2 = ~merged_df[unrelated[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        # Takes value 0 if the split is incomplete, 1 if the split is complete and 2 in the US-US case
        merged_df['IS_UNRELATED_COMPLETE'] = mask * 1 + mask_US * 1

        # Is the split of total sales complete?
        total = ['TOTAL_US', 'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_OTHER_COUNTRY']
        self.total = total.copy()

        mask_0 = ~merged_df[total[0]].isnull()
        mask_1 = ~merged_df[total[1]].isnull()
        mask_2 = ~merged_df[total[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        # Takes value 0 if the split is incomplete, 1 if the split is complete and 2 in the US-US case
        merged_df['IS_TOTAL_COMPLETE'] = mask * 1 + mask_US * 1

        return merged_df.copy()

    def get_data_with_sales_percentages(self):
        merged_df = self.get_data_with_indicator_variables()

        bases = ['TOTAL_US', 'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_OTHER_COUNTRY']

        percentage_columns = []

        for sales_type in ['RELATED', 'UNRELATED', 'TOTAL']:
            if sales_type in ['RELATED', 'UNRELATED']:
                existing_columns = [column + '_' + sales_type for column in bases]

                total_column = 'TOTAL_' + sales_type

            else:
                existing_columns = bases.copy()

                total_column = 'TOTAL_COMPUTED'

            merged_df[total_column] = merged_df[existing_columns].sum(axis=1)

            for i, destination in enumerate(['US', 'AFFILIATE_COUNTRY', 'OTHER_COUNTRY']):
                new_column = '_'.join(['PERC', sales_type, destination])

                percentage_columns.append(new_column)

                merged_df[new_column] = merged_df[existing_columns[i]] / merged_df[total_column]

        # We eliminate the revenue percentages computed while some data are missing
        for column in percentage_columns:
            merged_df[column] = merged_df.apply(
                lambda row: eliminate_irrelevant_percentages(row, column),
                axis=1
            )

        self.percentage_columns = percentage_columns.copy()

        return merged_df.copy()

    def build_imputations_dict(self):
        merged_df = self.get_data_with_sales_percentages()

        imputations = {}

        # We iterate over continents
        for continent_code in merged_df['CONTINENT_CODE'].unique():
            # We first exclude the case of OTHER_GROUPS
            if continent_code == 'OTHER_GROUPS':
                continue

            # We restrict the dataset to the continent under consideration
            restricted_df = merged_df[merged_df['CONTINENT_CODE'] == continent_code].copy()

            # We eliminate the US from the continental aggregation as it is a specific case
            if continent_code == 'AMR':
                restricted_df = restricted_df[restricted_df['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

            # We build a dictionary with continent codes as keys and dictionaries (to be filled) as values
            imputations[continent_code] = {}

            # We iterate over sales categories
            for sales_type in ['UNRELATED', 'RELATED', 'TOTAL']:

                # We restrict the dataset to jurisdictions of the continent under consideration for which BEA data on
                # the given type of sales are complete
                indicator_column = '_'.join(['IS', sales_type, 'COMPLETE'])
                restricted_df = restricted_df[restricted_df[indicator_column] == 1].copy()

                # We aggregate total sales, sales to the host country, sales to the headquarter country and sales to any
                # other country over the restricted dataset, for a given type of sales
                sums = restricted_df.sum()

                if sales_type in ['UNRELATED', 'RELATED']:
                    suffix = '_' + sales_type
                    total_column = 'TOTAL' + suffix

                else:
                    suffix = ''
                    total_column = 'TOTAL_COMPUTED'

                for destination in ['US', 'AFFILIATE_COUNTRY', 'OTHER_COUNTRY']:
                    column = 'TOTAL_' + destination + suffix

                    # key corresponds to the name of the column in which we want to impute the missing value
                    key = '_'.join(['PERC', sales_type, destination])

                    # We compute the sales percentage for a given continent, type of sales and destination
                    imputations[continent_code][key] = sums.loc[column] / sums.loc[total_column]

        # We now deal with the OTHER_GROUPS case
        # We again exclude the US from the aggregation as it is too specific a case
        restricted_df = restricted_df[restricted_df['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

        # We build a dictionary with continent codes as keys and dictionaries (to be filled) as values
        imputations['OTHER_GROUPS'] = {}

        # We iterate over sales categories
        for sales_type in ['UNRELATED', 'RELATED', 'TOTAL']:

            # We restrict the dataset to jurisdictions of the continent under consideration for which BEA data on
            # the given type of sales are complete
            indicator_column = '_'.join(['IS', sales_type, 'COMPLETE'])
            restricted_df = restricted_df[restricted_df[indicator_column] == 1].copy()

            # We aggregate total sales, sales to the host country, sales to the headquarter country and sales to any
            # other country over the restricted dataset, for a given type of sales
            sums = restricted_df.sum()

            if sales_type in ['UNRELATED', 'RELATED']:
                suffix = '_' + sales_type
                total_column = 'TOTAL' + suffix

            else:
                suffix = ''
                total_column = 'TOTAL_COMPUTED'

            for destination in ['US', 'AFFILIATE_COUNTRY', 'OTHER_COUNTRY']:
                column = 'TOTAL_' + destination + suffix

                # key corresponds to the name of the column in which we want to impute the missing value
                key = '_'.join(['PERC', sales_type, destination])

                # We compute the sales percentage for a given continent, type of sales and destination
                imputations['OTHER_GROUPS'][key] = sums.loc[column] / sums.loc[total_column]

        return imputations.copy()

    def get_extended_sales_percentages(self):
        merged_df = self.get_data_with_sales_percentages()
        imputations = self.build_imputations_dict()

        # We impute missing values thanks to the pre-constructed dictionary
        for column in self.percentage_columns:
            merged_df[column] = merged_df.apply(
                lambda row: impute_missing_values(row, column, imputations),
                axis=1
            )

        # We drop absolute amounts from the BEA data
        # We are only interested in sales percentages to distribute the IRS revenue variables
        totals = ['TOTAL_FOREIGN', 'TOTAL', 'TOTAL_RELATED', 'TOTAL_UNRELATED', 'TOTAL_COMPUTED']
        indicators = ['IS_RELATED_COMPLETE', 'IS_UNRELATED_COMPLETE', 'IS_TOTAL_COMPLETE']

        merged_df.drop(
            columns=self.related + self.unrelated + self.total + totals + indicators,
            inplace=True
        )

        return merged_df.copy()
