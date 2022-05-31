########################################################################################################################
# --- Imports

from sales_calculator import USSalesCalculator, SimplifiedGlobalSalesCalculator

from flask import Flask
from flask import request


########################################################################################################################
# --- App instantiation and landing page

app = Flask(__name__)


@app.route(
    '/',
    methods=['GET']
)
def get_landing_page():
    return 'Hello! Welcome to the adjusted sales mapping API. Documentation is here:'


########################################################################################################################
# --- Main routes

@app.route(
    '/US_sales_mapping',
    methods=['GET']
)
def get_adjusted_US_sales_mapping():

    year = request.args.get('year', default=2018, type=int)

    US_merchandise_exports_source = request.args.get(
        'US_merchandise_exports_source',
        default='Comtrade',
        type=str
    )
    US_services_exports_source = request.args.get(
        'US_services_exports_source',
        default='BaTIS',
        type=str
    )
    non_US_merchandise_exports_source = request.args.get(
        'non_US_merchandise_exports_source',
        default='Comtrade',
        type=str
    )
    non_US_services_exports_source = request.args.get(
        'non_US_services_exports_source',
        default='BaTIS',
        type=str
    )

    winsorize_export_percs = request.args.get('winsorize_export_percs', default=True, type=bool)
    US_winsorizing_threshold = request.args.get('US_winsorizing_threshold', default=0.5, type=float)
    non_US_winsorizing_threshold = request.args.get('non_US_winsorizing_threshold', default=0.5, type=float)

    service_flows_to_exclude = request.args.get('service_flows_to_exclude', default=[], type=list)

    calculator = USSalesCalculator(
        year=year,
        US_only=True,
        US_merchandise_exports_source=US_merchandise_exports_source,
        US_services_exports_source=US_services_exports_source,
        non_US_merchandise_exports_source=non_US_merchandise_exports_source,
        non_US_services_exports_source=non_US_services_exports_source,
        winsorize_export_percs=winsorize_export_percs,
        US_winsorizing_threshold=US_winsorizing_threshold,
        non_US_winsorizing_threshold=non_US_winsorizing_threshold,
        service_flows_to_exclude=service_flows_to_exclude
    )

    sales_mapping = calculator.get_final_sales_mapping()

    return sales_mapping.to_json()


@app.route('/global_sales_mapping', methods=['GET'])
def get_adjusted_global_sales_mapping():

    year = request.args.get('year', default=2018, type=int)

    aamne_domestic_sales_perc = request.args.get('aamne_domestic_sales_perc', default=True, type=bool)

    US_merchandise_exports_source = request.args.get(
        'US_merchandise_exports_source',
        default='Comtrade',
        type=str
    )
    US_services_exports_source = request.args.get(
        'US_services_exports_source',
        default='BaTIS',
        type=str
    )
    non_US_merchandise_exports_source = request.args.get(
        'non_US_merchandise_exports_source',
        default='Comtrade',
        type=str
    )
    non_US_services_exports_source = request.args.get(
        'non_US_services_exports_source',
        default='BaTIS',
        type=str
    )

    winsorize_export_percs = request.args.get('winsorize_export_percs', default=True, type=bool)
    US_winsorizing_threshold = request.args.get('US_winsorizing_threshold', default=0.5, type=float)
    non_US_winsorizing_threshold = request.args.get('non_US_winsorizing_threshold', default=0.5, type=float)

    service_flows_to_exclude = request.args.get('service_flows_to_exclude', default=[], type=list)

    calculator = SimplifiedGlobalSalesCalculator(
        year=year,
        aamne_domestic_sales_perc=aamne_domestic_sales_perc,
        US_merchandise_exports_source=US_merchandise_exports_source,
        US_services_exports_source=US_services_exports_source,
        non_US_merchandise_exports_source=non_US_merchandise_exports_source,
        non_US_services_exports_source=non_US_services_exports_source,
        winsorize_export_percs=winsorize_export_percs,
        US_winsorizing_threshold=US_winsorizing_threshold,
        non_US_winsorizing_threshold=non_US_winsorizing_threshold,
        service_flows_to_exclude=service_flows_to_exclude
    )

    sales_mapping = calculator.get_final_sales_mapping()

    return sales_mapping.to_json()
