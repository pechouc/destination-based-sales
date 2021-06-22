import os

import numpy as np
import pandas as pd

from utils import CODES_TO_IMPUTE_BEA, impute_missing_codes

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_bea_data = os.path.join(path_to_dir, 'data', 'bea_data.csv')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class BEADataPreprocessor:

    def __init__(
        self,
        path_to_bea=path_to_bea_data, path_to_geo_file=path_to_geographies
    ):

        self.path_to_bea = path_to_bea
        self.path_to_geo_file = path_to_geo_file

        self.CODES_TO_IMPUTE = CODES_TO_IMPUTE_BEA.copy()

    def load_data(self):

        bea = pd.read_csv(self.path_to_bea, delimiter=';')

        return bea.copy()

    def load_data_with_geo_codes(self):

        bea = self.load_data()

        geographies = pd.read_csv(self.path_to_geo_file)

        merged_df = bea.merge(
            geographies,
            how='left',
            left_on='AFFILIATE_COUNTRY_NAME', right_on='NAME'
        )

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
