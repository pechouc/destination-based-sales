from destination_based_sales.comtrade import UNComtradeProcessor
from destination_based_sales.balanced_trade import BalancedTradeStatsProcessor
from destination_based_sales.bop import USBalanceOfPaymentsProcessor


def test_missing_values_balanced_trade_merchandise():

    for year in [2016, 2017, 2018, 2019]:
        processor = BalancedTradeStatsProcessor(year=year, service_flows_to_exclude=[])

        merchandise = processor.load_clean_merchandise_data()

        ser = merchandise.isnull().sum()

        assert (ser > 0).sum() == 0


def test_missing_values_balanced_trade_services():

    for year in [2016, 2017, 2018, 2019]:
        processor = BalancedTradeStatsProcessor(year=year, service_flows_to_exclude=[])

        services = processor.load_clean_services_data()

        ser = services.isnull().sum()

        assert (ser > 0).sum() == 0


def test_missing_values_us_bop_merchandise():

    for year in [2016, 2017, 2018, 2019]:
        processor = USBalanceOfPaymentsProcessor(year=year)

        merchandise = processor.load_final_merchandise_data()

        ser = merchandise.isnull().sum()

        assert (ser > 0).sum() == 0


def test_missing_values_us_bop_services():

    for year in [2016, 2017, 2018, 2019]:
        processor = USBalanceOfPaymentsProcessor(year=year)

        services = processor.load_final_services_data()

        ser = services.isnull().sum()

        assert (ser > 0).sum() == 0


def test_missing_values_un_comtrade():

    for year in [2016, 2017, 2018, 2019]:
        processor = UNComtradeProcessor(year=year)

        merchandise = processor.load_data_with_geographies()

        ser = merchandise.isnull().sum()

        assert (ser > 0).sum() == 0


def test_nb_observations_balanced_trade_merchandise():

    for year in [2016, 2017, 2018, 2019]:
        processor = BalancedTradeStatsProcessor(year=year, service_flows_to_exclude=[])

        merchandise = processor.load_clean_merchandise_data()

        merchandise['KEY'] = merchandise['OTHER_COUNTRY_CODE'] + merchandise['AFFILIATE_COUNTRY_CODE']

        assert merchandise.shape[0] == merchandise['KEY'].nunique()


def test_nb_observations_balanced_trade_services():

    for year in [2016, 2017, 2018, 2019]:
        processor = BalancedTradeStatsProcessor(year=year, service_flows_to_exclude=[])

        services = processor.load_clean_services_data()

        services['KEY'] = services['OTHER_COUNTRY_CODE'] + services['AFFILIATE_COUNTRY_CODE']

        assert services.shape[0] == services['KEY'].nunique()


def test_nb_observations_us_bop_merchandise():

    for year in [2016, 2017, 2018, 2019]:
        processor = USBalanceOfPaymentsProcessor(year=year)

        merchandise = processor.load_final_merchandise_data()

        merchandise['KEY'] = merchandise['OTHER_COUNTRY_CODE'] + merchandise['AFFILIATE_COUNTRY_CODE']

        assert merchandise.shape[0] == merchandise['KEY'].nunique()


def test_nb_observations_us_bop_services():

    for year in [2016, 2017, 2018, 2019]:
        processor = USBalanceOfPaymentsProcessor(year=year)

        services = processor.load_final_services_data()

        services['KEY'] = services['OTHER_COUNTRY_CODE'] + services['AFFILIATE_COUNTRY_CODE']

        assert services.shape[0] == services['KEY'].nunique()


def test_nb_observations_un_comtrade():

    for year in [2016, 2017, 2018, 2019]:
        processor = UNComtradeProcessor(year=year)

        merchandise = processor.load_data_with_geographies()

        merchandise['KEY'] = merchandise['OTHER_COUNTRY_CODE'] + merchandise['AFFILIATE_COUNTRY_CODE']

        assert merchandise.shape[0] == merchandise['KEY'].nunique()
