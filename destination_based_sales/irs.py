import numpy as np
import pandas as pd

from utils import CODES_TO_IMPUTE_IRS, impute_missing_codes, UK_CARIBBEAN_ISLANDS

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_irs_data = os.path.join(path_to_dir, 'data', 'irs_2018_revenues.csv')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')

class IRSDataPreprocessor:

    def __init__(
        self,
        path_to_cbcr=path_to_irs_data, path_to_geo_file=path_to_geographies
    ):

        self.path_to_cbcr = path_to_cbcr
        self.path_to_geo_file = path_to_geographies

        self.CODES_TO_IMPUTE = CODES_TO_IMPUTE_IRS.copy()
        self.UK_CARIBBEAN_ISLANDS = UK_CARIBBEAN_ISLANDS

    def load_data(self):

        irs = pd.read_csv(self.path_to_cbcr, delimiter=';')

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

        irs = pd.DataFrame.from_dict(irs)

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
