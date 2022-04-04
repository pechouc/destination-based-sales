import numpy as np
from numpy.testing import assert_almost_equal

from destination_based_sales.bea import BEADataPreprocessor, ExtendedBEADataLoader


def test_missing_values():
    for year in [2016, 2017, 2018]:
        processor = BEADataPreprocessor(year=year)

        df = processor.load_final_data()

        ser = df[
            ['AFFILIATE_COUNTRY_NAME', 'NAME', 'CODE', 'CONTINENT_NAME', 'CONTINENT_CODE']
        ].isnull().sum()

        assert (ser > 0).sum() == 0


def test_grand_total():
    for year in [2016, 2017, 2018]:
        processor = BEADataPreprocessor(year=year)

        df = processor.load_final_data()

        df['DIFF'] = np.abs(
            df['TOTAL'] - (df['TOTAL_US'] + df['TOTAL_AFFILIATE_COUNTRY'] + df['TOTAL_OTHER_COUNTRY'])
        )

        assert (df['DIFF'] > 1).sum() == 0


def test_total_foreign():
    for year in [2016, 2017, 2018]:
        processor = BEADataPreprocessor(year=year)

        df = processor.load_final_data()

        df['DIFF'] = np.abs(
            df['TOTAL_FOREIGN'] - (df['TOTAL_AFFILIATE_COUNTRY'] + df['TOTAL_OTHER_COUNTRY'])
        )

        assert (df['DIFF'] > 1).sum() == 0


def test_total_us():
    for year in [2016, 2017, 2018]:
        processor = BEADataPreprocessor(year=year)

        df = processor.load_final_data()

        df['DIFF'] = np.abs(
            df['TOTAL_US'] - (df['TOTAL_US_RELATED'] + df['TOTAL_US_UNRELATED'])
        )

        assert (df['DIFF'] > 1).sum() == 0


def test_total_affiliate_country():
    for year in [2016, 2017, 2018]:
        processor = BEADataPreprocessor(year=year)

        df = processor.load_final_data()

        df['DIFF'] = np.abs(
            df['TOTAL_AFFILIATE_COUNTRY'] - (
                df['TOTAL_AFFILIATE_COUNTRY_RELATED'] + df['TOTAL_AFFILIATE_COUNTRY_UNRELATED']
            )
        )

        assert (df['DIFF'] > 1).sum() == 0


def test_total_other_country():
    for year in [2016, 2017, 2018]:
        processor = BEADataPreprocessor(year=year)

        df = processor.load_final_data()

        df['DIFF'] = np.abs(
            df['TOTAL_OTHER_COUNTRY'] - (
                df['TOTAL_OTHER_COUNTRY_RELATED'] + df['TOTAL_OTHER_COUNTRY_UNRELATED']
            )
        )

        assert (df['DIFF'] > 1).sum() == 0


def test_missing_values_extended_BEA():
    for year in [2016, 2017, 2018]:
        loader = ExtendedBEADataLoader(year=year)

        df = loader.get_extended_sales_percentages()

        ser = df.isnull().sum()

        assert (ser > 0).sum() == 0


def test_summed_percentages_extended_BEA():
    for year in [2016, 2017, 2018]:
        loader = ExtendedBEADataLoader(year=year)

        df = loader.get_extended_sales_percentages()

        related = ['PERC_RELATED_US', 'PERC_RELATED_AFFILIATE_COUNTRY', 'PERC_RELATED_OTHER_COUNTRY']
        unrelated = ['PERC_UNRELATED_US', 'PERC_UNRELATED_AFFILIATE_COUNTRY', 'PERC_UNRELATED_OTHER_COUNTRY']
        total = ['PERC_TOTAL_US', 'PERC_TOTAL_AFFILIATE_COUNTRY', 'PERC_TOTAL_OTHER_COUNTRY']

        for columns in [related, unrelated, total]:
            restricted_df = df[columns].copy()

            assert_almost_equal(
                list(restricted_df.sum(axis=1)),
                1
            )
