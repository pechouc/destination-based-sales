"""
This module defines several useful functions, mobilised throughout the other Python files.
"""

########################################################################################################################
# --- Imports

import os
import json

import numpy as np


########################################################################################################################
# --- For IRS data

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_codes_to_impute_IRS = os.path.join(path_to_dir, 'data', 'codes_to_impute_IRS.json')
path_to_codes_to_impute_BEA = os.path.join(path_to_dir, 'data', 'codes_to_impute_BEA.json')

with open(path_to_codes_to_impute_IRS) as file:
    CODES_TO_IMPUTE_IRS = json.loads(file.read())

with open(path_to_codes_to_impute_BEA) as file:
    CODES_TO_IMPUTE_BEA = json.loads(file.read())

CONTINENT_CODES_TO_IMPUTE_TRADE = {
    'OASIAOCN': 'APAC',
    'UKI': 'AMR'
}

CONTINENT_CODES_TO_IMPUTE_OECD_CBCR = {
    'OAF': 'AFR',
    'OAM': 'AMR',
    'OAS': 'APAC',
    'OTE': 'EUR',
    'AFRIC': 'AFR',
    'AMER': 'AMR',
    'ASIAT': 'APAC',
    'EUROP': 'EUR',
    'GRPS': 'OTHER_GROUPS',
    'UKI': 'AMR'
}

UK_CARIBBEAN_ISLANDS = [
    'CYM',
    'VGB',
    'AIA',
    'MSR',
    'TCA'
]


def impute_missing_codes(row, column, codes_to_impute):
    if row['AFFILIATE_COUNTRY_NAME'] in codes_to_impute.keys():
        return codes_to_impute[row['AFFILIATE_COUNTRY_NAME']][column]

    else:
        return row[column]


########################################################################################################################
# --- For splitting revenue variables

def eliminate_irrelevant_percentages(row, column):
    sales_type = column.split('_')[1]

    indicator_column = '_'.join(['IS', sales_type, 'COMPLETE'])

    if row[indicator_column] != 0:
        return row[column]

    else:
        return np.nan


def impute_missing_values(row, column, imputations):
    if np.isnan(row[column]):
        return imputations[row['CONTINENT_CODE']][column]

    else:
        return row[column]


# FOR TRADE STATISTICS (AND OECD CBCR FOR THE FIRST FUNCTION)

def impute_missing_continent_codes(row, mapping):
    if not isinstance(row['CONTINENT_CODE'], str) and np.isnan(row['CONTINENT_CODE']):
        if 'CODE' in row.index:
            if isinstance(row['CODE'], float) and np.isnan(row['CODE']):
                print(row)
            return mapping[row['CODE']]
        else:
            return mapping[row['AFFILIATE_COUNTRY_CODE']]

    else:
        return row['CONTINENT_CODE']


def ensure_country_overlap_with_IRS(row, unique_IRS_country_codes, UK_caribbean_islands):
    mapping = {
        'EUR': 'OEUR',
        'AFR': 'OAFR',
        'APAC': 'OASIAOCN',
        'AMR': 'OAMR'
    }

    if row['OTHER_COUNTRY_CODE'] in UK_caribbean_islands:
        return 'UKI'

    else:
        if row['OTHER_COUNTRY_CODE'] in unique_IRS_country_codes:
            return row['OTHER_COUNTRY_CODE']

        else:
            return mapping[row['OTHER_COUNTRY_CONTINENT_CODE']]


def ensure_country_overlap_with_OECD_CbCR(row, unique_OECD_country_codes, UK_caribbean_islands):
    mapping = {
        'EUR': 'OEUR',
        'AFR': 'OAFR',
        'APAC': 'OASIAOCN',
        'AMR': 'OAMR'
    }

    if row['OTHER_COUNTRY_CODE'] in UK_caribbean_islands:
        return 'UKI'

    else:
        if row['OTHER_COUNTRY_CODE'] in unique_OECD_country_codes:
            return row['OTHER_COUNTRY_CODE']

        else:
            return mapping[row['OTHER_COUNTRY_CONTINENT_CODE']]


# class ServicesDataTransformer:

#     def __init__(self):
#         self.amounts_to_distribute = {}

#         self.allocations = {}
#         self.list_of_OTHER_codes = ['OAFR', 'OAMR', 'OASIAOCN', 'OEUR']

#     def fit(self, data):
#         for country in data['AFFILIATE_COUNTRY_CODE'].unique():
#             mask_affiliate_country = data['AFFILIATE_COUNTRY_CODE'] == country

#             mask_RWD = data['OTHER_COUNTRY_CODE'] == 'RWD'
#             mask_OTHER = data['OTHER_COUNTRY_CODE'].isin(self.list_of_OTHER_codes)

#             mask = np.logical_and(mask_affiliate_country, mask_RWD)

#             if not data[mask].empty:
#                 self.amounts_to_distribute[country] = data[mask]['SERVICES_EXPORTS'].iloc[0]

#             else:
#                 self.amounts_to_distribute[country] = 0

#             mask = np.logical_and(mask_affiliate_country, mask_OTHER)

#             if not data[mask].empty:
#                 restricted_df = data[mask].copy()

#                 restricted_df['ALLOCABLE_SHARE'] = (
#                     restricted_df['SERVICES_EXPORTS'] / restricted_df['SERVICES_EXPORTS'].sum()
#                 )

#                 self.allocations[country] = {}

#                 for code in self.list_of_OTHER_codes:
#                     if code not in restricted_df['OTHER_COUNTRY_CODE'].unique():
#                         self.allocations[country][code] = 0

#                     else:
#                         self.allocations[country][code] = restricted_df[
#                             restricted_df['OTHER_COUNTRY_CODE'] == code
#                         ]['ALLOCABLE_SHARE'].iloc[0]

#             else:
#                 self.allocations[country] = {
#                     code: 0.25 for code in self.list_of_OTHER_codes
#                 }

#     def transform(self, data):
#         data = data[data['OTHER_COUNTRY_CODE'] != 'RWD'].copy()

#         data['SERVICES_EXPORTS'] = data.apply(
#             (
#                 lambda row: row['SERVICES_EXPORTS'] + self.amounts_to_distribute[row['AFFILIATE_COUNTRY_CODE']]
#                 * self.allocations[row['AFFILIATE_COUNTRY_CODE']][row['OTHER_COUNTRY_CODE']]
#                 if row['OTHER_COUNTRY_CODE'] in self.list_of_OTHER_codes else row['SERVICES_EXPORTS']
#             ),
#             axis=1
#         )

#         return data.reset_index(drop=True)


########################################################################################################################
# --- For Analytical AMNE data

def compute_foreign_owned_gross_output(row, include_US):
    foreign_owned_gross_output = 0

    for column in row.index:
        if column in ['cou', 'GROSS_OUTPUT_INCL_US']:
            continue

        elif column == row['cou']:
            continue

        else:
            foreign_owned_gross_output += row[column]

    if include_US:
        return foreign_owned_gross_output

    else:
        return foreign_owned_gross_output - row['USA']


########################################################################################################################
# --- For tax_reforms.py

def define_category(row):
    if row['IS_EU'] == 1:
        return 3

    elif row['REPORTS_CBCR'] == 1:
        return 2

    elif row['IS_LARGE_TAX_HAVEN'] == 1:
        return 1

    else:
        return 0
