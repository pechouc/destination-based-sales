from destination_based_sales.sales_calculator import SimplifiedGlobalSalesCalculator

from tax_deficit_simulator.calculator import TaxDeficitCalculator


class TaxReformSimulator:

    def __init__(self, year):

        self.year = year

        # Storing the global sales calculator and the unadjusted / adjusted sales mappings
        self.sales_calculator = SimplifiedGlobalSalesCalculator(
            year=self.year,
            aamne_domestic_sales_perc=False,
            US_merchandise_exports_source='Comtrade',
            US_services_exports_source='BaTIS',
            non_US_merchandise_exports_source='Comtrade',
            non_US_services_exports_source='BaTIS',
            winsorize_export_percs=True,
            US_winsorizing_threshold=0.5,
            non_US_winsorizing_threshold=0.5,
            service_flows_to_exclude=[]
        )

        self.oecd_sales_mapping = self.oecd.copy()

        self.adjusted_sales_mapping = self.get_final_sales_mapping()

        # Storing the tax deficits to redistribute
        self.tax_deficit_calculator = TaxDeficitCalculator(
            year=self.year,
            add_AUT_AUT_row=True,
            sweden_treatment='adjust',
            belgium_treatment='replace',
            use_adjusted_profits=True,
            average_ETRs=True,
            carve_outs=False,
            de_minimis_exclusion=True,
            fetch_data_online=True
        )

        self.tax_deficit_calculator.load_clean_data()

        self.tax_deficits = self.get_total_tax_deficits(minimum_ETR=0.15)

    def compare_Barake_et_al_unilateral_scenarios(self):

        oecd_sales_mapping = self.oecd_sales_mapping.copy()
        adjusted_sales_mapping = self.adjusted_sales_mapping.copy()

        tax_deficits = self.tax_deficits.copy()

        tax_deficits = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(
                adjusted_sales_mapping['PARENT_COUNTRY_CODE'].unique()
            )
        ].copy()

        # Testing GitHub jobs

        return oecd_sales_mapping.copy()
