from numpy.testing import assert_almost_equal

from destination_based_sales.sales_calculator import USSalesCalculator, SimplifiedGlobalSalesCalculator
from destination_based_sales.irs import IRSDataPreprocessor
from destination_based_sales.oecd_cbcr import CbCRPreprocessor


def test_us_sales_calculator_missing_values():

    for year in [2016, 2017, 2018, 2019]:
        calculator = USSalesCalculator(
            year=year,
            US_only=False,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            service_flows_to_exclude=[],
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5
        )

        df = calculator.get_final_sales_mapping()

        ser = df.isnull().sum()

        assert (ser > 0).sum() == 0


def test_us_sales_calculator_matching_totals():

    for year in [2016, 2017, 2018, 2019]:
        # Fetching the adjusted sales mapping and grouping by affiliate country
        calculator = USSalesCalculator(
            year=year,
            US_only=False,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            service_flows_to_exclude=[],
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5
        )

        df = calculator.get_final_sales_mapping()

        grouped_df = df.groupby('AFFILIATE_COUNTRY_CODE').sum().reset_index()

        # Fetching the IRS revenue data for comparison
        preprocessor = IRSDataPreprocessor(year=year)
        irs = preprocessor.load_final_data()
        irs = irs[['CODE', 'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']].copy()

        # Operating the comparison
        merged_df = grouped_df.merge(
            irs,
            how='outer',
            left_on='AFFILIATE_COUNTRY_CODE', right_on='CODE'
        )

        ser = merged_df.isnull().sum()
        assert (ser > 0).sum() == 0

        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']:
            assert_almost_equal(
                list(merged_df[column + '_x']),
                list(merged_df[column + '_y']),
                decimal=1
            )


def test_us_sales_calculator_duplicate_country_pairs():

    for year in [2016, 2017, 2018, 2019]:
        calculator = USSalesCalculator(
            year=year,
            US_only=False,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            service_flows_to_exclude=[],
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5
        )

        df = calculator.get_final_sales_mapping()

        nrows = df.shape[0]
        nrows_bis = df[['AFFILIATE_COUNTRY_CODE', 'OTHER_COUNTRY_CODE']].drop_duplicates().shape[0]

        assert nrows == nrows_bis


def test_simplified_global_sales_calculator_missing_values():

    for year in [2016, 2017]:
        calculator = SimplifiedGlobalSalesCalculator(
            year=year,
            aamne_domestic_sales_perc=False,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            service_flows_to_exclude=[],
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5
        )

        df = calculator.get_final_sales_mapping()

        ser = df.isnull().sum()

        assert (ser > 0).sum() == 0


def test_simplified_global_sales_calculator_duplicate_country_pairs():

    for year in [2016, 2017]:
        calculator = SimplifiedGlobalSalesCalculator(
            year=year,
            aamne_domestic_sales_perc=False,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            service_flows_to_exclude=[],
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5
        )

        df = calculator.get_final_sales_mapping()

        nrows = df.shape[0]
        nrows_bis = df[
            ['PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE', 'OTHER_COUNTRY_CODE']
        ].drop_duplicates().shape[0]

        assert nrows == nrows_bis


def test_simplified_global_sales_calculator_matching_totals():

    for year in [2016, 2017]:
        calculator = SimplifiedGlobalSalesCalculator(
            year=year,
            aamne_domestic_sales_perc=False,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            service_flows_to_exclude=[],
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5
        )

        df = calculator.get_final_sales_mapping()

        # ### At the level of parent country / affiliate country pairs
        grouped_df = df.groupby(
            ['PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE']
        ).sum().reset_index()

        # Fetching the OECD revenue data for comparison
        preprocessor = CbCRPreprocessor(year=year)
        oecd = preprocessor.get_preprocessed_revenue_data()
        oecd = oecd[
            [
                'PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE',
                'UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES'
            ]
        ].copy()

        # Operating the comparison
        merged_df = grouped_df.merge(
            oecd,
            how='outer',
            on=['PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE']
        )

        ser = merged_df.isnull().sum()
        assert (ser > 0).sum() == 0

        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']:
            assert_almost_equal(
                list(merged_df[column + '_x']),
                list(merged_df[column + '_y']),
                decimal=2
            )

        # ### At the parent country level
        grouped_df = df.groupby('PARENT_COUNTRY_CODE').sum().reset_index()

        # Fetching the OECD revenue data for comparison
        oecd = preprocessor.get_preprocessed_revenue_data()
        oecd = oecd.groupby('PARENT_COUNTRY_CODE').sum().reset_index()

        # Operating the comparison
        merged_df = grouped_df.merge(
            oecd,
            how='outer',
            on='PARENT_COUNTRY_CODE'
        )

        ser = merged_df.isnull().sum()
        assert (ser > 0).sum() == 0

        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']:
            assert_almost_equal(
                list(merged_df[column + '_x']),
                list(merged_df[column + '_y']),
                decimal=1
            )

        # ### At the affiliate country level
        grouped_df = df.groupby('AFFILIATE_COUNTRY_CODE').sum().reset_index()

        # Fetching the OECD revenue data for comparison
        oecd = preprocessor.get_preprocessed_revenue_data()
        oecd = oecd.groupby('AFFILIATE_COUNTRY_CODE').sum().reset_index()

        # Operating the comparison
        merged_df = grouped_df.merge(
            oecd,
            how='outer',
            on='AFFILIATE_COUNTRY_CODE'
        )

        ser = merged_df.isnull().sum()
        assert (ser > 0).sum() == 0

        for column in ['UNRELATED_PARTY_REVENUES', 'RELATED_PARTY_REVENUES', 'TOTAL_REVENUES']:
            assert_almost_equal(
                list(merged_df[column + '_x']),
                list(merged_df[column + '_y']),
                decimal=1
            )
