"""
This module is used to load and preprocess data from the Bureau of Economic Analysis (BEA). These allow to split revenue
variables between sales directed to the host (or affiliate) country, to the US and to any third country. Data are loaded
from Excel files saved in the "data" folder.
"""


########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

from destination_based_sales.utils import CODES_TO_IMPUTE_BEA, impute_missing_codes


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
