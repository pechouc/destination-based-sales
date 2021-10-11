import os

import numpy as np
import pandas as pd

from destination_based_sales.utils import CODES_TO_IMPUTE_BEA, impute_missing_codes

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class BEADataPreprocessor:

    def __init__(
        self,
        year,
        path_to_dir=path_to_dir,
        path_to_geo_file=path_to_geographies
    ):
        self.year = year

        self.path_to_bea = os.path.join(
            path_to_dir,
            'data',
            str(year),
            'Part-II-E1-E17.xls'
        )

        self.path_to_geo_file = path_to_geo_file

        self.CODES_TO_IMPUTE = CODES_TO_IMPUTE_BEA.copy()

    def load_data(self):

        bea = pd.read_excel(self.path_to_bea, sheet_name='Table II.E 2')

        bea.columns = [
            'AFFILIATE_COUNTRY_NAME', 'TOTAL', 'TOTAL_US', 'TOTAL_US_RELATED', 'TOTAL_US_UNRELATED', 'TOTAL_FOREIGN',
            'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_AFFILIATE_COUNTRY_RELATED', 'TOTAL_AFFILIATE_COUNTRY_UNRELATED',
            'TOTAL_OTHER_COUNTRY', 'TOTAL_OTHER_COUNTRY_RELATED', 'TOTAL_OTHER_COUNTRY_UNRELATED'
        ]

        bea = bea.loc[8:].copy()

        bea = bea[~(bea.isnull().sum(axis=1) >= 11)].copy()

        bea = bea.iloc[:-2].copy()

        bea = bea[bea['AFFILIATE_COUNTRY_NAME'] != 'Latin America and Other Western Hemisphere'].copy()

        bea.reset_index(inplace=True, drop=True)

        continent_names = [
            'Europe',
            'South America',
            'Central America',
            'Other Western Hemisphere',
            'Africa',
            'Middle East',
            'Asia and Pacific',
        ]

        total_indices = list(
            bea[
                bea['AFFILIATE_COUNTRY_NAME'].isin(continent_names)
            ].index
        )

        continent_extracts = {}

        for i, continent_name in enumerate(continent_names):
            if i + 1 < len(total_indices):
                continent_df = bea.loc[total_indices[i]:total_indices[i + 1] - 1].copy()

            else:
                continent_df = bea.loc[total_indices[i]:bea.index[-1]].copy()

            continent_df['AFFILIATE_COUNTRY_NAME'] = continent_df['AFFILIATE_COUNTRY_NAME'].map(
                lambda country_name: country_name if country_name != 'Other' else 'Other ' + continent_name
            )

            continent_df = continent_df[continent_df['AFFILIATE_COUNTRY_NAME'] != continent_name].copy()

            continent_extracts[continent_name] = continent_df.copy()

        bea_cleaned = bea[bea['AFFILIATE_COUNTRY_NAME'] == 'Canada'].copy()

        for continent_extract in continent_extracts.values():
            bea_cleaned = pd.concat([bea_cleaned, continent_extract], axis=0)

        for column in bea_cleaned.columns[1:]:
            bea_cleaned[column] = bea_cleaned[column].map(
                lambda x: np.nan if x == '(D)' else x
            )

        return bea_cleaned.reset_index(drop=True)

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
