import os

import numpy as np
import pandas as pd

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_comtrade_data = os.path.join(path_to_dir, 'data', 'selected_comtrade_data.csv')
path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')


class UNComtradeProcessor():

    def __init__(
        self,
        year,
        path_to_comtrade_data=path_to_comtrade_data,
        path_to_geographies=path_to_geographies
    ):

        self.year = year

        self.path_to_comtrade_data = path_to_comtrade_data

        self.path_to_geographies = path_to_geographies
        self.geographies = pd.read_csv(self.path_to_geographies)

    def load_raw_data(self):
        data = pd.read_csv(self.path_to_comtrade_data)

        # Focusing on the year of interest
        data = data[data['Year'] == self.year].copy()
        data = data.drop(columns='Year')

        # Eliminating certain reporting entities in which we are not interested
        data = data[~data['Reporter'].isin(['EU-28', 'EU', 'ASEAN', 'Other Asia, nes'])].copy()

        # Eliminating certain partner entities in which we are not interested
        data = data[
            ~data['Partner'].isin(
                [
                    'Other Asia, nes', 'Special Categories', 'Other Africa, nes', 'Areas, nes',
                    'Other Europe, nes', 'Bunkers', 'LAIA, nes', 'Oceania, nes',
                    'North America and Central America, nes', 'Free Zones'
                ]
            )
        ].copy()

        data = data.drop(columns=['Reporter', 'Partner'])

        return data.copy()

    def load_net_imports_data(self):
        data = self.load_raw_data()

        # We net re-imports out of the imports when they are available
        data_reshaped = data.pivot(
            index=['Reporter ISO', 'Partner ISO'],
            columns='Trade Flow',
            values='Trade Value (US$)'
        ).reset_index()

        data_reshaped['Re-Import'] = data_reshaped['Re-Import'].fillna(0)
        data_reshaped['Re-Export'] = data_reshaped['Re-Export'].fillna(0)

        data_reshaped['NET_IMPORTS'] = data_reshaped['Import'] - data_reshaped['Re-Import']
        data_reshaped['NET_EXPORTS'] = data_reshaped['Export'] - data_reshaped['Re-Export']

        data_reshaped = data_reshaped.drop(columns=['Import', 'Re-Import', 'Export', 'Re-Export'])

        # We flip the dataset from a mapping of imports into a mapping of exports
        # We simply rename the reporting country as the destination and the partner as the exporter
        base_df = data_reshaped.rename(
            columns={
                'Reporter ISO': 'OTHER_COUNTRY_CODE',
                'Partner ISO': 'AFFILIATE_COUNTRY_CODE',
                'NET_IMPORTS': 'MERCHANDISE_EXPORTS'
            }
        )
        base_df = base_df.drop(columns='NET_EXPORTS').dropna().copy()

        # We add exports whenever the destination country does not report any trade data
        data_reshaped_exports = data_reshaped.drop(columns='NET_IMPORTS').dropna().copy()
        data_reshaped_exports = data_reshaped_exports[
            ~data_reshaped_exports['Partner ISO'].isin(base_df['OTHER_COUNTRY_CODE'].unique())
        ]
        df_to_append = data_reshaped_exports.rename(
            columns={
                'Partner ISO': 'OTHER_COUNTRY_CODE',
                'Reporter ISO': 'AFFILIATE_COUNTRY_CODE',
                'NET_EXPORTS': 'MERCHANDISE_EXPORTS'
            }
        )

        output_df = pd.concat([base_df, df_to_append], axis=0)

        return output_df.copy()

    def load_data_with_geographies(self):
        data = self.load_net_imports_data()

        # Adding the OTHER_COUNTRY_CONTINENT_CODE column
        data = data.merge(
            self.geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='OTHER_COUNTRY_CODE', right_on='CODE'
        )

        data['CONTINENT_CODE'] = data['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR', 'ATC'] else x  # Attributing Antarctica to Americas (arbitrary)
        )
        data['CONTINENT_CODE'] = data['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        data = data.drop(columns=['CODE'])
        data = data.rename(columns={'CONTINENT_CODE': 'OTHER_COUNTRY_CONTINENT_CODE'})

        # Adding the AFFILIATE_COUNTRY_CONTINENT_CODE column
        data = data.merge(
            self.geographies[['CODE', 'CONTINENT_CODE']].drop_duplicates(),
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        data['CONTINENT_CODE'] = data['CONTINENT_CODE'].map(
            lambda x: 'AMR' if x in ['NAMR', 'SAMR'] else x
        )
        data['CONTINENT_CODE'] = data['CONTINENT_CODE'].map(
            lambda x: 'APAC' if x in ['ASIA', 'OCN'] else x
        )

        data = data.drop(columns=['CODE'])
        data = data.rename(columns={'CONTINENT_CODE': 'AFFILIATE_COUNTRY_CONTINENT_CODE'})

        # Imputing the missing continent codes not found in geographies.csv
        imputation = {
            'BLM': 'AMR',   # Saint Barthélémy
            'ATB': 'AMR'   # British Antarctic Territory arbitrarily rattached to America
        }

        data['AFFILIATE_COUNTRY_CONTINENT_CODE'] = data.apply(
            (
                lambda row: imputation.get(row['AFFILIATE_COUNTRY_CODE'], row['AFFILIATE_COUNTRY_CONTINENT_CODE'])
                if isinstance(row['AFFILIATE_COUNTRY_CONTINENT_CODE'], float) and
                np.isnan(row['AFFILIATE_COUNTRY_CONTINENT_CODE']) else row['AFFILIATE_COUNTRY_CONTINENT_CODE']
            ),
            axis=1
        )

        return data.copy()
