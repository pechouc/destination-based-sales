import os

import numpy as np
import pandas as pd

from destination_based_sales.utils import CODES_TO_IMPUTE_IRS, impute_missing_codes, UK_CARIBBEAN_ISLANDS

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_irs_data = os.path.join(path_to_dir, 'data', '18it01acbc.xlsx')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class IRSDataPreprocessor:

    def __init__(
        self,
        year,
        path_to_dir=path_to_dir,
        path_to_geo_file=path_to_geographies
    ):
        self.year = year

        self.path_to_cbcr = os.path.join(
            path_to_dir,
            'data',
            str(year),
            f'{year - 2000}it01acbc.xlsx'
        )

        self.path_to_geo_file = path_to_geographies

        self.CODES_TO_IMPUTE = CODES_TO_IMPUTE_IRS.copy()
        self.UK_CARIBBEAN_ISLANDS = UK_CARIBBEAN_ISLANDS.copy()

    def load_data(self):

        irs = pd.read_excel(
            self.path_to_cbcr,
            engine='openpyxl'
        )

        irs.drop(
            columns=['Unnamed: 1'] + list(irs.columns[5:]),
            inplace=True
        )

        irs.columns = [
            'AFFILIATE_COUNTRY_NAME', 'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES'
        ]

        irs = irs.loc[5:].copy()

        mask_stateless = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'stateless' in country_name.lower()
        )
        mask_total = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'total' in country_name.lower()
        )
        mask = ~np.logical_or(
            mask_stateless, mask_total
        )

        irs = irs[mask].copy()

        irs = irs.iloc[:-4].copy()

        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            (
                lambda country_name: ('Other ' + country_name.split(',')[0]).replace('&', 'and')
                if 'other' in country_name.lower() else country_name
            )
        )
        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'United Kingdom' if 'United Kingdom' in country_name else country_name
        )
        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'Korea' if country_name.startswith('Korea') else country_name
        )
        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'Congo' if country_name.endswith('(Brazzaville)') else country_name
        )

        irs.reset_index(drop=True, inplace=True)

        return irs.copy()

    def load_data_with_geo_codes(self):

        irs = self.load_data()

        geographies = pd.read_csv(self.path_to_geo_file)

        merged_df = irs.merge(
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

        merged_df.drop(columns=['NAME'], inplace=True)

        return merged_df.copy()

    def load_data_with_UKI(self):

        irs = self.load_data_with_geo_codes()

        extract = irs[irs['CODE'].isin(self.UK_CARIBBEAN_ISLANDS)].copy()

        irs = irs[~irs['CODE'].isin(self.UK_CARIBBEAN_ISLANDS)].copy()

        irs.reset_index(inplace=True, drop=True)

        dict_df = irs.to_dict()

        dict_df[irs.columns[0]][len(irs)] = 'United Kingdom Islands, Caribbean'
        dict_df[irs.columns[1]][len(irs)] = extract['UNRELATED_PARTY_REVENUES'].sum()
        dict_df[irs.columns[2]][len(irs)] = extract['RELATED_PARTY_REVENUES'].sum()
        dict_df[irs.columns[3]][len(irs)] = extract['TOTAL_REVENUES'].sum()
        dict_df[irs.columns[4]][len(irs)] = 'UKI'
        dict_df[irs.columns[5]][len(irs)] = 'America'
        dict_df[irs.columns[6]][len(irs)] = 'AMR'

        irs = pd.DataFrame.from_dict(dict_df)

        return irs.copy()

    def load_final_data(self):

        irs = self.load_data_with_UKI()

        irs['CONTINENT_CODE'] = irs['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['SAMR', 'NAMR'] else x
        )

        irs['CONTINENT_NAME'] = irs['CONTINENT_NAME'].map(
            lambda x: 'America' if x in ['South America', 'North America'] else x
        )

        irs['CONTINENT_CODE'] = irs['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] or x is None else x
        )

        irs['CONTINENT_NAME'] = irs['CONTINENT_NAME'].map(
            lambda x: 'Asia-Pacific' if x in ['Asia', 'Oceania'] or x is None else x
        )

        return irs.copy()
