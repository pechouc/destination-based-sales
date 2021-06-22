import json

# FOR IRS DATA

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_codes_to_impute_IRS = os.path.join(path_to_dir, 'data', 'codes_to_impute_IRS.json')
path_to_codes_to_impute_BEA = os.path.join(path_to_dir, 'data', 'codes_to_impute_BEA.json')

with open(path_to_codes_to_impute_IRS) as file:
    CODES_TO_IMPUTE_IRS = json.loads(file)

with open(path_to_codes_to_impute_BEA) as file:
    CODES_TO_IMPUTE_BEA = json.loads(file)

CONTINENT_CODES_TO_IMPUTE_TRADE = {
    'OASIAOCN': 'APAC',
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


# FOR REVENUE SPLIT

def eliminate_irrelevant_percentages(row, column):
    sales_type = column.split('_')[1]

    indicator_column = '_'.join(['IS', sales_type, 'COMPLETE'])

    if bool(row[indicator_column]):
        return row[column]

    else:
        return np.nan


def impute_missing_values(row, column, imputations):
    if np.isnan(row[column]):
        return imputations[row['CONTINENT_CODE']][column]

    else:
        return row[column]


# FOR TRADE STATISTICS

def impute_missing_continent_codes(row, mapping):
    if not isinstance(row['CONTINENT_CODE'], str) and np.isnan(row['CONTINENT_CODE']):
        return mapping[row['CODE']]

    else:
        return row['CONTINENT_CODE']