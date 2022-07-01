"""
This module is used to load and preprocess the country-by-country data of the Internal Revenue Service (IRS). These pro-
vide the three revenue variables that we aim at distributing based on their approximative ultimate destination. Data are
loaded from Excel files saved in the "data" folder.
"""


########################################################################################################################
# --- Imports

import os

import numpy as np
import pandas as pd

from destination_based_sales.utils import CODES_TO_IMPUTE_IRS, impute_missing_codes, UK_CARIBBEAN_ISLANDS,\
    online_path_to_geo_file


########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


########################################################################################################################
# --- Content

class IRSDataPreprocessor:

    def __init__(
        self,
        year,
        path_to_dir=path_to_dir,
        path_to_geo_file=path_to_geographies,
        load_data_online=False,
    ):
        """
        The instructions allowing to load and preprocess IRS data are organised in a Python class, IRSDataPreprocessor.

        This is the instantiation function for this class. It requires several arguments:

        - the year to consider (for now, one of 2016, 2017 or 2018);
        - the path to the directory where this Python file is located, to retrieve the appropriate data file;
        - the path to the "geographies.csv" file, used for instance to complement IRS data with country codes.
        """
        self.year = year

        if not load_data_online:
            # We reconstruct the path to the relevant Excel file, which depends on the year considered
            self.path_to_cbcr = os.path.join(
                path_to_dir,
                'data',
                str(year),
                f'{year - 2000}it01acbc.xlsx'
            )

            self.path_to_geo_file = path_to_geographies

        else:
            self.path_to_cbcr = f'https://www.irs.gov/pub/irs-soi/{year - 2000}it01acbc.xlsx'
            self.path_to_geo_file = online_path_to_geo_file

        self.CODES_TO_IMPUTE = CODES_TO_IMPUTE_IRS.copy()
        self.UK_CARIBBEAN_ISLANDS = UK_CARIBBEAN_ISLANDS.copy()

    def load_data(self):
        """
        This class method is used to load and clean the data from the IRS. It relies on the data file paths, saved as
        class attributes when the instantiation function is called. Preprocessing steps are detailed in comments below.
        """

        # We load the data from the appropriate Excel file
        irs = pd.read_excel(
            self.path_to_cbcr,
            engine='openpyxl'
        )

        # We eliminate irrelevant columns and rename the appropriate ones
        irs.drop(
            columns=['Unnamed: 1'] + list(irs.columns[5:]),
            inplace=True
        )

        irs.columns = [
            'AFFILIATE_COUNTRY_NAME', 'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES'
        ]

        # We filter out irrelevant rows
        # In particular, we eliminate rows corresponding to stateless entities and continental totals
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

        # We rename fields of the form "Other [+ CONTINENT_NAME]"
        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            (
                lambda country_name: ('Other ' + country_name.split(',')[0]).replace('&', 'and')
                if 'other' in country_name.lower() else country_name
            )
        )

        # We deal with a few specific country names
        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'United Kingdom' if 'United Kingdom' in country_name else country_name
        )
        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'Korea' if country_name.startswith('Korea') else country_name
        )
        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'Congo' if country_name.endswith('(Brazzaville)') else country_name
        )
        irs['AFFILIATE_COUNTRY_NAME'] = irs['AFFILIATE_COUNTRY_NAME'].map(
            lambda country_name: 'US Virgin Islands' if country_name == 'U.S. Virgin Islands' else country_name
        )

        irs.reset_index(drop=True, inplace=True)

        return irs.copy()

    def load_data_with_geo_codes(self):
        """
        This class method is used to add geographical ISO codes to the raw dataset, loaded with the "load_data" method.
        It relies on the "impute_missing_codes" function defined in utils.py.
        """
        irs = self.load_data()

        geographies = pd.read_csv(self.path_to_geo_file)

        # We merge the DataFrame containing raw IRS data with the one containing the ISO code correspondences
        merged_df = irs.merge(
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

        merged_df.drop(columns=['NAME'], inplace=True)

        return merged_df.copy()

    def load_data_with_UKI(self):
        """
        To match the BEA data, that show a "United Kingdom Islands, Caribbean" aggregate for all the related jurisdi-
        ctions, we do the same aggregation in the IRS data. Basically, we sum unrelated-party, related-party and total
        revenues for all the countries concerned and associate the totals to a new "United Kingdom Islands, Caribbean"
        affiliate country name. We then eliminate fields included in the aggregation.
        """
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
        """
        This class method allows to load the fully preprocessed IRS data. Relying on the "load_data_with_UKI" method,
        continent names and codes are limited to 4 pairs, corresponding respectively to Europe, Africa, America (North
        and South) and Asia-Pacific (gathering Asia and Oceania).
        """
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
