import json

### FOR IRS DATA

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_codes_to_impute_IRS = os.path.join(path_to_dir, 'data', 'codes_to_impute_IRS.json')
path_to_codes_to_impute_BEA = os.path.join(path_to_dir, 'data', 'codes_to_impute_BEA.json')

with open(path_to_codes_to_impute_IRS) as file:
    CODES_TO_IMPUTE_IRS = json.loads(file)

with open(path_to_codes_to_impute_BEA) as file:
    CODES_TO_IMPUTE_BEA = json.loads(file)

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

### FOR BEA DATA
