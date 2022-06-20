from numpy.testing import assert_almost_equal

from destination_based_sales.trade_statistics import TradeStatisticsProcessor

from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor

########################################################################################################################
# We directly instantiate the processor objects and load the final tables as computations can be quite long

first_outputs = {}
second_outputs = {}

for year in [2016, 2017, 2018, 2019]:
    US_only = False if year in [2016, 2017] else True

    # We will test a first combination of data sources
    first_processor = TradeStatisticsProcessor(
        year=year,
        US_only=US_only,
        US_merchandise_exports_source='BoP',
        US_services_exports_source='BaTIS',
        non_US_merchandise_exports_source='Comtrade',
        non_US_services_exports_source='BaTIS',
        service_flows_to_exclude=[],
        winsorize_export_percs=True,
        US_winsorizing_threshold=0.5,
        non_US_winsorizing_threshold=0.5
    )

    first_output_df = first_processor.get_final_exports_distributions()

    first_outputs[year] = first_output_df.copy()

    # As well as a second combination of data sources
    second_processor = TradeStatisticsProcessor(
        year=year,
        US_only=US_only,
        US_merchandise_exports_source='Comtrade',
        US_services_exports_source='BaTIS',
        non_US_merchandise_exports_source='Comtrade',
        non_US_services_exports_source='BaTIS',
        service_flows_to_exclude=[],
        winsorize_export_percs=True,
        US_winsorizing_threshold=0.5,
        non_US_winsorizing_threshold=0.5
    )

    second_output_df = second_processor.get_final_exports_distributions()

    second_outputs[year] = second_output_df.copy()

# We fetch the winsorizing thresholds
winsorizing_threshold = second_processor.winsorizing_threshold
winsorizing_threshold_US = second_processor.winsorizing_threshold_US


def test_first_combination_affiliate_vs_other():
    for _, df in first_outputs.items():
        assert(
            (df['AFFILIATE_COUNTRY_CODE'] == df['OTHER_COUNTRY_CODE']).sum() == 0
        )


def test_second_combination_affiliate_vs_other():
    for _, df in second_outputs.items():
        assert(
            (df['AFFILIATE_COUNTRY_CODE'] == df['OTHER_COUNTRY_CODE']).sum() == 0
        )


def test_first_combination_missing_values():
    for _, df in first_outputs.items():
        ser = df.isnull().sum()
        assert (ser > 0).sum() == 0


def test_second_combination_missing_values():
    for _, df in second_outputs.items():
        ser = df.isnull().sum()
        assert (ser > 0).sum() == 0


def test_first_combination_export_percs():
    for _, df in first_outputs.items():
        assert_almost_equal(
            list(df.groupby('AFFILIATE_COUNTRY_CODE').sum()['EXPORT_PERC']),
            1
        )


def test_second_combination_export_percs():
    for _, df in second_outputs.items():
        assert_almost_equal(
            list(df.groupby('AFFILIATE_COUNTRY_CODE').sum()['EXPORT_PERC']),
            1
        )


def test_first_combination_winsorizing():
    for _, df in first_outputs.items():
        us_extract = df[df['AFFILIATE_COUNTRY_CODE'] == 'USA'].copy()
        non_us_extract = df[df['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

        assert(
            (us_extract['EXPORT_PERC'] < winsorizing_threshold_US).sum() == 0
        )

        assert(
            (non_us_extract['EXPORT_PERC'] < winsorizing_threshold).sum() == 0
        )


def test_second_combination_winsorizing():
    for _, df in second_outputs.items():
        us_extract = df[df['AFFILIATE_COUNTRY_CODE'] == 'USA'].copy()
        non_us_extract = df[df['AFFILIATE_COUNTRY_CODE'] != 'USA'].copy()

        assert(
            (us_extract['EXPORT_PERC'] < winsorizing_threshold_US).sum() == 0
        )

        assert(
            (non_us_extract['EXPORT_PERC'] < winsorizing_threshold).sum() == 0
        )


def test_first_combination_overlap():
    for year, df in first_outputs.items():
        if year in [2016, 2017]:
            processor = CbCRPreprocessor(year=year, breakdown_threshold=0)
            unique_country_codes = processor.get_preprocessed_revenue_data()
            unique_country_codes = list(unique_country_codes['AFFILIATE_COUNTRY_CODE'].unique())

        else:
            processor = IRSDataPreprocessor(year=year)
            unique_country_codes = processor.load_final_data()
            unique_country_codes = unique_country_codes['CODE'].unique()

        assert(
            df[~df['OTHER_COUNTRY_CODE'].isin(unique_country_codes)].shape[0] == 0
        )


def test_second_combination_overlap():
    for year, df in second_outputs.items():
        if year in [2016, 2017]:
            processor = CbCRPreprocessor(year=year, breakdown_threshold=0)
            unique_country_codes = processor.get_preprocessed_revenue_data()
            unique_country_codes = list(unique_country_codes['AFFILIATE_COUNTRY_CODE'].unique())

        else:
            processor = IRSDataPreprocessor(year=year)
            unique_country_codes = processor.load_final_data()
            unique_country_codes = unique_country_codes['CODE'].unique()

        assert(
            df[~df['OTHER_COUNTRY_CODE'].isin(unique_country_codes)].shape[0] == 0
        )


if __name__ == '__main__':
    test_first_combination_overlap()
    test_second_combination_overlap()
