"""
This module is used to load and preprocess data from the Bureau of Economic Analysis (BEA). These allow to split revenue
variables between sales directed to the host (or affiliate) country, to the US and to any third country.
"""


########################################################################################################################
# --- Imports

# General imports
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Imports for the BEADataPreprocessor class
from destination_based_sales.utils import CODES_TO_IMPUTE_BEA, impute_missing_codes

# Imports used through the ExtendedBEADataLoader class
from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor

from destination_based_sales.utils import eliminate_irrelevant_percentages, impute_missing_values, \
    online_path_to_geo_file, online_path_to_TH_list

########################################################################################################################
# --- Diverse

path_to_dir = os.path.dirname(os.path.abspath(__file__))

path_to_geographies = os.path.join(path_to_dir, 'data', 'geographies.csv')
path_to_tax_havens = os.path.join(path_to_dir, 'data', 'tax_havens.csv')


########################################################################################################################
# --- Content

class BEADataPreprocessor:

    def __init__(
        self,
        year: int,
        load_data_online: bool = False
    ):
        """Encapsulates the logic allowing to load and preprocess BEA data.

        :param year: the year to consider (for now, one of 2016, 2017, 2018 or 2019)
        :type year: int
        :param load_data_online: whether to load the data online or locally, defaults to False
        :type load_data_online: bool, optional

        :rtype: destination_based_sales.bea.BEADataPreprocessor
        :returns: object of the class BEADataPreprocessor, used to load and preprocess BEA data
        """
        self.year = year

        # We construct the path to the relevant data file, which depends on the year considered
        if not load_data_online:
            self.path_to_bea = os.path.join(
                path_to_dir,
                'data',
                str(year),
                'Part-II-E1-E17.xls'
            )

            self.path_to_geo_file = path_to_geographies
            self.path_to_tax_havens = path_to_tax_havens

        else:
            if year in [2016, 2017, 2018]:
                self.path_to_bea = f'https://apps.bea.gov/international/xls/usdia{year}r/Part-II-E1-E17.xls'

            else:
                self.path_to_bea = 'https://apps.bea.gov/international/xls/usdia2019p/Part-II-E1-E17.xls'

            self.path_to_geo_file = online_path_to_geo_file
            self.path_to_tax_havens = online_path_to_TH_list

        self.CODES_TO_IMPUTE = CODES_TO_IMPUTE_BEA.copy()

    def load_data(self) -> pd.DataFrame:
        """Loads data from the BEA and applies the basic cleaning steps.

        :rtype: pandas.DataFrame
        :return: BEA data on the sales of goods and services of US multinationals, after a basic cleaning
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

    def load_data_with_geo_codes(self) -> pd.DataFrame:
        """Adds geographical ISO codes to the raw dataset, loaded with the "load_data" method.

        :rtype: pandas.DataFrame
        :return: BEA data with country and continent codes and names
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

    def load_final_data(self) -> pd.DataFrame:
        """Loads the fully preprocessed BEA data, relying on the "load_data_with_geo_codes" method.

        :rtype: pandas.DataFrame
        :return: fully preprocessed BEA data
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

    def plot_shares_of_foreign_sales(self, distinguish_THs=False, save_PNG=False, path_to_output_folder=None):

        if save_PNG and path_to_output_folder is None:
            raise Exception('To save the figure as a PNG file, you must indicate the path to an output folder.')

        bea = self.load_final_data()

        # Computing the shares of foreign sales for each type of transactions
        for sales_type_suffix in ['_RELATED', '_UNRELATED', '']:
            sales_to_us_column = 'TOTAL_US' + sales_type_suffix
            local_sales_column = 'TOTAL_AFFILIATE_COUNTRY' + sales_type_suffix
            sales_to_other_country_column = 'TOTAL_OTHER_COUNTRY' + sales_type_suffix

            new_column = (
                'PERC_FOREIGN_SALES' + sales_type_suffix if sales_type_suffix != '' else 'PERC_FOREIGN_SALES_TOTAL'
            )

            bea[new_column] = (
                (bea[sales_to_us_column] + bea[sales_to_other_country_column]) /
                (bea[sales_to_us_column] + bea[sales_to_other_country_column] + bea[local_sales_column])
            ) * 100

        # Adding the "Tax haven" indicator
        if distinguish_THs:
            tax_havens = pd.read_csv(self.path_to_tax_havens)
            bea['Is tax haven?'] = bea['CODE'].isin(tax_havens['CODE'].unique()) * 1
            bea['Is tax haven?'] = bea['Is tax haven?'].map({0: 'No', 1: 'Yes'})

        # Matplotlib parameters determining the look of the graph
        plt.rcParams.update(
            {
                'axes.titlesize': 20,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18
            }
        )

        # Plotting the boxplot graphs
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))

        i = 0

        for ax, column_suffix in zip(axes.flatten(), ['_UNRELATED', '_RELATED', '_TOTAL']):
            if not distinguish_THs:
                sns.boxplot(
                    bea['PERC_FOREIGN_SALES' + column_suffix],
                    orient='v',
                    ax=ax,

                )

            else:
                sns.boxplot(
                    orient='v',
                    x='Is tax haven?',
                    y='PERC_FOREIGN_SALES' + column_suffix,
                    ax=ax,
                    data=bea
                )

            ax.set_ylim(0, 105)

            if i == 0:
                ax.set_ylabel('Share of foreign sales (%)')
            else:
                ax.set_ylabel(None)

            i += 1

            if column_suffix == '_UNRELATED':
                ax.set_title('Panel A: Unaffiliated sales')
            elif column_suffix == '_RELATED':
                ax.set_title('Panel B: Affiliated sales')
            else:
                ax.set_title('Panel C: Total sales')

        if save_PNG:
            if not distinguish_THs:
                path = os.path.join(path_to_output_folder, f'BEA_basic_boxplots_{self.year}.png')

            else:
                path = os.path.join(path_to_output_folder, f'BEA_boxplots_with_THs_{self.year}.png')

            plt.savefig(path, bbox_inches='tight')

        plt.show()


class ExtendedBEADataLoader:

    def __init__(
        self,
        year: int,
        load_data_online: bool = False
    ):
        """Encapsulates the logic behind the extension of BEA sales percentages to all partners in CbCR data.

        :param year: the year to consider (for now, one of 2016, 2017, 2018 or 2019)
        :type year: int
        :param load_data_online: whether to load the data online or locally, defaults to False
        :type load_data_online: bool, optional

        :rtype: destination_based_sales.bea.ExtendedBEADataLoader
        :return: object of the class ExtendedBEADataLoader, used to load and preprocess extended BEA data
        """
        self.year = year

        self.path_to_dir = path_to_dir
        self.load_data_online = load_data_online

        # Instantiating an object of the class BEADataPreprocessor defined above
        self.bea_preprocessor = BEADataPreprocessor(year=year, load_data_online=load_data_online)

        # Defining the complete set of affiliate countries to cover eventually
        if year in [2016, 2017]:
            oecd_preprocessor = CbCRPreprocessor(
                year=year,
                breakdown_threshold=0,
                load_data_online=load_data_online
            )
            oecd = oecd_preprocessor.get_preprocessed_revenue_data()

            self.target_countries = oecd[
                ['AFFILIATE_COUNTRY_CODE', 'AFFILIATE_COUNTRY_NAME', 'CONTINENT_CODE']
            ].drop_duplicates()

        else:
            irs_preprocessor = IRSDataPreprocessor(year=year, load_data_online=load_data_online)
            irs = irs_preprocessor.load_final_data()

            self.target_countries = irs[
                ['CODE', 'AFFILIATE_COUNTRY_NAME', 'CONTINENT_CODE']
            ].drop_duplicates(
            ).rename(
                columns={
                    'CODE': 'AFFILIATE_COUNTRY_CODE'
                }
            )

    def load_data_with_US_US_row(self) -> pd.DataFrame:
        """Loads cleaned BEA data from the "load_final_data" method defined above and adds the US-US values.

        :rtype: pandas.DataFrame
        :return: BEA data loaded from Table II.E2, complemented with the US-US row from Table I.O1
        """
        df = self.bea_preprocessor.load_final_data()

        if not self.load_data_online:
            path_to_BEA_KR_tables = os.path.join(
                self.path_to_dir,
                'data',
                str(self.year),
                'Part-I-K1-R2.xls'
            )

        else:
            if self.year in [2016, 2017, 2018]:
                path_to_BEA_KR_tables = f'https://apps.bea.gov/international/xls/usdia{self.year}r/Part-I-K1-R2.xls'

            else:
                path_to_BEA_KR_tables = 'https://apps.bea.gov/international/xls/usdia2019p/Part-I-K1-R2.xls'

        temp = pd.read_excel(path_to_BEA_KR_tables, sheet_name='Table I.O 1')

        # Cleaning and reorganising the table
        column_names = temp.loc[4].to_dict().copy()
        column_names[list(column_names.keys())[0]] = 'Industry'
        column_names['Unnamed: 1'] = 'Total'

        temp.rename(columns=column_names, inplace=True)

        # Saving the relevant information in a dictionary
        us_sales = temp.loc[6].to_dict()

        us_imputation = {}

        # Imputing the BEA-like distribution of US-US sales into the merged DataFrame
        # Sales to the affiliate country and to the headquarter country are directed to the same final destination
        for column in df.columns[1:12]:

            if column == 'TOTAL':
                us_imputation[column] = [us_sales['Total']]

            elif 'US' in column:
                us_imputation[column] = [us_sales['To U.S. persons'] * 0]

            elif 'AFFILIATE_COUNTRY' in column:
                us_imputation[column] = [us_sales['To U.S. persons'] * 1]

            else:
                us_imputation[column] = [us_sales['To foreign affiliates'] + us_sales['To other foreign persons']]

        us_imputation['AFFILIATE_COUNTRY_NAME'] = ['United States']
        us_imputation['NAME'] = ['United States']
        us_imputation['CODE'] = ['USA']
        us_imputation['CONTINENT_NAME'] = ['America']
        us_imputation['CONTINENT_CODE'] = ['AMR']

        us_imputation = pd.DataFrame(us_imputation)

        # Concatenating the two DataFrames to get the BEA data with the US-US row
        df = pd.concat([df, us_imputation], axis=0)

        return df.copy()

    def get_merged_dataframe(self) -> pd.DataFrame:
        """Merges the BEA data with the US-US row (obtained from previous method) onto the target set of partners.

        :rtype: pandas.DataFrame
        :return:
        """
        # Loading BEA data with the US-US row
        df = self.load_data_with_US_US_row()

        # Merging these data onto the target set of partner countries, to which we want to extend them
        merged_df = self.target_countries.merge(
            df,
            how='left',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        # Dropping unnecessary columns and renaming some others
        merged_df.drop(
            columns=['AFFILIATE_COUNTRY_NAME_y', 'NAME', 'CODE', 'CONTINENT_CODE_y', 'CONTINENT_NAME'],
            inplace=True
        )

        merged_df.rename(
            columns={
                'AFFILIATE_COUNTRY_NAME_x': 'AFFILIATE_COUNTRY_NAME',
                'CONTINENT_CODE_x': 'CONTINENT_CODE'
            },
            inplace=True
        )

        return merged_df.copy()

    def get_data_with_indicator_variables(self) -> pd.DataFrame:
        """Adds indicator variables to the dataset, that provide information about the availability of sales amounts.

        :rtype: pandas.DataFrame
        :return: target set of partners with BEA data and indicator variables showing the availability of the latter
        """
        merged_df = self.get_merged_dataframe()

        mask_US = (merged_df['AFFILIATE_COUNTRY_CODE'] == 'USA')

        # Is the split of related-party sales complete?
        related = ['TOTAL_US_RELATED', 'TOTAL_AFFILIATE_COUNTRY_RELATED', 'TOTAL_OTHER_COUNTRY_RELATED']
        self.related = related.copy()

        mask_0 = ~merged_df[related[0]].isnull()
        mask_1 = ~merged_df[related[1]].isnull()
        mask_2 = ~merged_df[related[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        # Takes value 0 if the split is incomplete, 1 if the split is complete and 2 in the US-US case
        merged_df['IS_RELATED_COMPLETE'] = mask * 1 + mask_US * 1

        # Is the split of unrelated-party sales complete?
        unrelated = ['TOTAL_US_UNRELATED', 'TOTAL_AFFILIATE_COUNTRY_UNRELATED', 'TOTAL_OTHER_COUNTRY_UNRELATED']
        self.unrelated = unrelated.copy()

        mask_0 = ~merged_df[unrelated[0]].isnull()
        mask_1 = ~merged_df[unrelated[1]].isnull()
        mask_2 = ~merged_df[unrelated[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        # Takes value 0 if the split is incomplete, 1 if the split is complete and 2 in the US-US case
        merged_df['IS_UNRELATED_COMPLETE'] = mask * 1 + mask_US * 1

        # Is the split of total sales complete?
        total = ['TOTAL_US', 'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_OTHER_COUNTRY']
        self.total = total.copy()

        mask_0 = ~merged_df[total[0]].isnull()
        mask_1 = ~merged_df[total[1]].isnull()
        mask_2 = ~merged_df[total[2]].isnull()

        mask = np.logical_and(
            mask_0,
            np.logical_and(
                mask_1,
                mask_2
            )
        )

        # Takes value 0 if the split is incomplete, 1 if the split is complete and 2 in the US-US case
        merged_df['IS_TOTAL_COMPLETE'] = mask * 1 + mask_US * 1

        return merged_df.copy()

    def get_data_with_sales_percentages(self) -> pd.DataFrame:
        """Moves from the absolute amounts covered in the dataset to sales percentages.

        :rtype: pandas.DataFrame
        :return:

        .. note::

            Relies on the "eliminate_irrelevant_percentages" function defined in the "utils.py" module.
        """
        # --- Preliminary steps

        # We load the dataset with the indicator variables showing the availability of sales amounts
        merged_df = self.get_data_with_indicator_variables()

        # We therefore have one column per destination of the sales ("US", "OTHER_COUNTRY", "AFFILIATE_COUNTRY") x per
        # type of transaction ("RELATED", 'UNRELATED", "TOTAL"), as well as various sub-totals

        # Relevant columns start with one of these character strings depending on the destination of the sales
        bases = ['TOTAL_US', 'TOTAL_AFFILIATE_COUNTRY', 'TOTAL_OTHER_COUNTRY']

        # This list will store the names of the columns with sales percentages
        percentage_columns = []

        # --- Main loop and computations

        # We iterate over types of transaction
        for sales_type in ['RELATED', 'UNRELATED', 'TOTAL']:
            # We define (i) the list of existing columns that correspond to the type considered
            # and (ii) the name of the total column (e.g., showing the total related-party and unrelated-party sales)

            # First, for related and unrelated transactions
            if sales_type in ['RELATED', 'UNRELATED']:
                # We have three existing columns depending on the destination
                existing_columns = [column + '_' + sales_type for column in bases]

                # We name the column as "TOTAL_RELATED" or "TOTAL_UNRELATED"
                total_column = 'TOTAL_' + sales_type

            else:
                # We have three existing columns depending on the destination
                existing_columns = bases.copy()

                # We will basically re-compute the "TOTAL" column, that sums all sales
                total_column = 'TOTAL_COMPUTED'

            # We compute the total column by summing sales over the set of destinations
            merged_df[total_column] = merged_df[existing_columns].sum(axis=1)

            # We deduce sales percentages
            for i, destination in enumerate(['US', 'AFFILIATE_COUNTRY', 'OTHER_COUNTRY']):
                new_column = '_'.join(['PERC', sales_type, destination])

                percentage_columns.append(new_column)

                merged_df[new_column] = merged_df[existing_columns[i]] / merged_df[total_column]

        # --- Filtering out irrelevant sales percentages

        # We eliminate the revenue percentages computed while some data are missing (cf. "utils.py")
        for column in percentage_columns:
            merged_df[column] = merged_df.apply(
                lambda row: eliminate_irrelevant_percentages(row, column),
                axis=1
            )

        # --- Final steps

        # We save the names of the columns with sales percentages as an attribute
        self.percentage_columns = percentage_columns.copy()

        return merged_df.copy()

    def build_imputations_dict(self) -> dict:
        """Builds the dictionary used to impute missing sales percentages.

        :rtype: dict
        :return: builds the dictionary with each continent's mean sales percentages, used for imputation

        .. note::

            Dictionary with continent codes as keys and dictionaries as values.
            Sub-dictionaries show the continent's mean sales percentage for each destination x sales type combination.
        """
        # Loading BEA data with sales percentages
        merged_df = self.get_data_with_sales_percentages()

        imputations = {}

        # We iterate over continents
        for continent_code in merged_df['CONTINENT_CODE'].unique():
            # We first exclude the case of OTHER_GROUPS
            if continent_code == 'OTHER_GROUPS':
                continue

            # We restrict the dataset to the continent under consideration
            restricted_df = merged_df[merged_df['CONTINENT_CODE'] == continent_code].copy()

            # We eliminate the US from the continental aggregation as it is a specific case
            if continent_code == 'AMR':
                restricted_df = restricted_df[restricted_df['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

            # We build a dictionary with continent codes as keys and dictionaries (to be filled) as values
            imputations[continent_code] = {}

            # We iterate over sales categories
            for sales_type in ['UNRELATED', 'RELATED', 'TOTAL']:

                # We restrict the dataset to jurisdictions of the continent under consideration for which BEA data on
                # the given type of sales are complete
                indicator_column = '_'.join(['IS', sales_type, 'COMPLETE'])
                restricted_df = restricted_df[restricted_df[indicator_column] == 1].copy()

                # We aggregate total sales, sales to the host country, sales to the headquarter country and sales to any
                # other country over the restricted dataset, for a given type of sales
                sums = restricted_df.sum()

                if sales_type in ['UNRELATED', 'RELATED']:
                    suffix = '_' + sales_type
                    total_column = 'TOTAL' + suffix

                else:
                    suffix = ''
                    total_column = 'TOTAL_COMPUTED'

                for destination in ['US', 'AFFILIATE_COUNTRY', 'OTHER_COUNTRY']:
                    column = 'TOTAL_' + destination + suffix

                    # key corresponds to the name of the column in which we want to impute the missing value
                    key = '_'.join(['PERC', sales_type, destination])

                    # We compute the sales percentage for a given continent, type of sales and destination
                    imputations[continent_code][key] = sums.loc[column] / sums.loc[total_column]

        # We now deal with the OTHER_GROUPS case
        # We again exclude the US from the aggregation as it is too specific a case
        restricted_df = restricted_df[restricted_df['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

        # We build a dictionary with continent codes as keys and dictionaries (to be filled) as values
        imputations['OTHER_GROUPS'] = {}

        # We iterate over sales categories
        for sales_type in ['UNRELATED', 'RELATED', 'TOTAL']:

            # We restrict the dataset to jurisdictions of the continent under consideration for which BEA data on
            # the given type of sales are complete
            indicator_column = '_'.join(['IS', sales_type, 'COMPLETE'])
            restricted_df = restricted_df[restricted_df[indicator_column] == 1].copy()

            # We aggregate total sales, sales to the host country, sales to the headquarter country and sales to any
            # other country over the restricted dataset, for a given type of sales
            sums = restricted_df.sum()

            if sales_type in ['UNRELATED', 'RELATED']:
                suffix = '_' + sales_type
                total_column = 'TOTAL' + suffix

            else:
                suffix = ''
                total_column = 'TOTAL_COMPUTED'

            for destination in ['US', 'AFFILIATE_COUNTRY', 'OTHER_COUNTRY']:
                column = 'TOTAL_' + destination + suffix

                # key corresponds to the name of the column in which we want to impute the missing value
                key = '_'.join(['PERC', sales_type, destination])

                # We compute the sales percentage for a given continent, type of sales and destination
                imputations['OTHER_GROUPS'][key] = sums.loc[column] / sums.loc[total_column]

        return imputations.copy()

    def get_extended_sales_percentages(self) -> pd.DataFrame:
        """Deduces the extended sales percentages, covering all partners in country-by-country data.

        :rtype: pandas.DataFrame
        :return: extends sales percentages to the target set of countries (partners in country-by-country data)
        """
        merged_df = self.get_data_with_sales_percentages()
        imputations = self.build_imputations_dict()

        # We impute missing values thanks to the pre-constructed dictionary
        for column in self.percentage_columns:
            merged_df[column] = merged_df.apply(
                lambda row: impute_missing_values(row, column, imputations),
                axis=1
            )

        # We drop absolute amounts from the BEA data
        # We are only interested in sales percentages to distribute the IRS revenue variables
        totals = ['TOTAL_FOREIGN', 'TOTAL', 'TOTAL_RELATED', 'TOTAL_UNRELATED', 'TOTAL_COMPUTED']
        indicators = ['IS_RELATED_COMPLETE', 'IS_UNRELATED_COMPLETE', 'IS_TOTAL_COMPLETE']

        merged_df.drop(
            columns=self.related + self.unrelated + self.total + totals + indicators,
            inplace=True
        )

        return merged_df.copy()
