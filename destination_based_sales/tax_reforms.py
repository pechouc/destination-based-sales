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

    def compare_Barake_et_al_unilateral_scenarios(self, elim_negative_revenues=True):

        oecd_sales_mapping = self.oecd_sales_mapping.copy()
        adjusted_sales_mapping = self.adjusted_sales_mapping.copy()

        tax_deficits = self.tax_deficits.copy()

        tax_deficits = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'] != '..'
        ].copy()

        # ### Grouping the tax deficits of UK Caribbean Islands ---------------------------------------------------- ###
        UKI_extract = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(UK_CARIBBEAN_ISLANDS)
        ].copy()

        ser = UKI_extract.sum()

        ser.loc['Parent jurisdiction (whitespaces cleaned)'] = 'United Kingdom Islands, Caribbean'
        ser.loc['Parent jurisdiction (alpha-3 code)'] = 'UKI'

        UKI_extract = pd.DataFrame(ser).T.copy()

        restr_tax_deficits = tax_deficits[
            ~tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(UK_CARIBBEAN_ISLANDS)
        ].copy()

        restr_tax_deficits = pd.concat([restr_tax_deficits, UKI_extract], axis=0)
        restr_tax_deficits['tax_deficit'] = restr_tax_deficits['tax_deficit'].astype(float)

        # ### Preparing the unadjusted sales mapping --------------------------------------------------------------- ###
        restr_oecd_mapping = oecd_sales_mapping[
            [
                'PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES'
            ]
        ].copy()

        if elim_negative_revenues:
            restr_oecd_mapping['UNRELATED_PARTY_REVENUES'] = restr_oecd_mapping['UNRELATED_PARTY_REVENUES'].map(
                lambda x: max(x, 0)
            )

        parent_totals = restr_oecd_mapping.groupby('PARENT_COUNTRY_CODE').sum().to_dict()['UNRELATED_PARTY_REVENUES']

        restr_oecd_mapping['URP_PERCENTAGE'] = (
            restr_oecd_mapping['UNRELATED_PARTY_REVENUES']
            / restr_oecd_mapping['PARENT_COUNTRY_CODE'].map(parent_totals)
        )

        # We restrict to the affiliate countries for which we have computed a tax deficit already
        # Otherwise, the resulting estimated revenue gains will not be comparable
        restr_oecd_mapping = restr_oecd_mapping[
            restr_oecd_mapping['AFFILIATE_COUNTRY_CODE'].isin(
                restr_tax_deficits['Parent jurisdiction (alpha-3 code)'].unique()
            )
        ].copy()

        # Cases where the parent country and the affiliate country are the same are dealt with when taking the
        # pre-computed tax deficits
        restr_oecd_mapping = restr_oecd_mapping[
            restr_oecd_mapping['PARENT_COUNTRY_CODE'] != restr_oecd_mapping['AFFILIATE_COUNTRY_CODE']
        ].copy()

        restr_oecd_mapping = restr_oecd_mapping.drop(columns=['UNRELATED_PARTY_REVENUES'])

        # ### Preparing the adjusted sales mapping ----------------------------------------------------------------- ###

        restr_adjusted_mapping = adjusted_sales_mapping.groupby(
            ['PARENT_COUNTRY_CODE', 'OTHER_COUNTRY_CODE']
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        if elim_negative_revenues:
            restr_adjusted_mapping['UNRELATED_PARTY_REVENUES'] = restr_adjusted_mapping['UNRELATED_PARTY_REVENUES'].map(
                lambda x: max(x, 0)
            )

        parent_totals = restr_adjusted_mapping.groupby(
            'PARENT_COUNTRY_CODE'
        ).sum().to_dict()['UNRELATED_PARTY_REVENUES']

        restr_adjusted_mapping['URP_PERCENTAGE'] = (
            restr_adjusted_mapping['UNRELATED_PARTY_REVENUES']
            / restr_adjusted_mapping['PARENT_COUNTRY_CODE'].map(parent_totals)
        )

        # We restrict to the affiliate countries for which we have computed a tax deficit already
        # Otherwise, the resulting estimated revenue gains will not be comparable
        restr_adjusted_mapping = restr_adjusted_mapping[
            restr_adjusted_mapping['OTHER_COUNTRY_CODE'].isin(
                restr_tax_deficits['Parent jurisdiction (alpha-3 code)'].unique()
            )
        ].copy()

        # Cases where the parent country and the affiliate country are the same are dealt with when taking the
        # pre-computed tax deficits
        restr_adjusted_mapping = restr_adjusted_mapping[
            restr_adjusted_mapping['PARENT_COUNTRY_CODE'] != restr_adjusted_mapping['OTHER_COUNTRY_CODE']
        ].copy()

        restr_adjusted_mapping = restr_adjusted_mapping.drop(columns=['UNRELATED_PARTY_REVENUES'])

        # ### Deducing revenue gain estimates ---------------------------------------------------------------------- ###

        merged_df = restr_oecd_mapping.merge(
            restr_tax_deficits,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='Parent jurisdiction (alpha-3 code)'
        )
        merged_df = merged_df.drop(
            columns=[
                'Parent jurisdiction (alpha-3 code)', 'Parent jurisdiction (whitespaces cleaned)'
            ]
        )

        merged_df['ATTRIBUTABLE_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        temp_unadjusted = merged_df.groupby('AFFILIATE_COUNTRY_CODE').sum()['ATTRIBUTABLE_REVENUES'].to_dict()

        merged_df = restr_adjusted_mapping.merge(
            restr_tax_deficits,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='Parent jurisdiction (alpha-3 code)'
        )
        merged_df = merged_df.drop(
            columns=[
                'Parent jurisdiction (alpha-3 code)', 'Parent jurisdiction (whitespaces cleaned)'
            ]
        )

        merged_df['ATTRIBUTABLE_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        temp_adjusted = merged_df.groupby('OTHER_COUNTRY_CODE').sum()['ATTRIBUTABLE_REVENUES'].to_dict()

        # ### Gathering all types of revenue gains ----------------------------------------------------------------- ###

        restr_tax_deficits['UNADJ_ADDITIONAL_REVENUES'] = restr_tax_deficits['Parent jurisdiction (alpha-3 code)'].map(
            temp_unadjusted
        )
        restr_tax_deficits['ADJ_ADDITIONAL_REVENUES'] = restr_tax_deficits['Parent jurisdiction (alpha-3 code)'].map(
            temp_adjusted
        )

        restr_tax_deficits['UNADJ_TOTAL_TAX_DEFICITS'] = (
            restr_tax_deficits['tax_deficit'] + restr_tax_deficits['UNADJ_ADDITIONAL_REVENUES']
        )
        restr_tax_deficits['ADJ_TOTAL_TAX_DEFICITS'] = (
            restr_tax_deficits['tax_deficit'] + restr_tax_deficits['ADJ_ADDITIONAL_REVENUES']
        )

        # ### Formatting the table --------------------------------------------------------------------------------- ###


        return restr_tax_deficits.copy()

    def full_sales_apportionment(self, elim_negative_revenues=True):

        oecd_sales_mapping = self.oecd_sales_mapping.copy()
        adjusted_sales_mapping = self.adjusted_sales_mapping.copy()

        tax_deficits = self.tax_deficits.copy()

        tax_deficits = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'] != '..'
        ].copy()


        # ### Grouping the tax deficits of UK Caribbean Islands ---------------------------------------------------- ###

        UKI_extract = tax_deficits[
            tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(UK_CARIBBEAN_ISLANDS)
        ].copy()

        ser = UKI_extract.sum()

        ser.loc['Parent jurisdiction (whitespaces cleaned)'] = 'United Kingdom Islands, Caribbean'
        ser.loc['Parent jurisdiction (alpha-3 code)'] = 'UKI'

        UKI_extract = pd.DataFrame(ser).T.copy()

        restr_tax_deficits = tax_deficits[
            ~tax_deficits['Parent jurisdiction (alpha-3 code)'].isin(UK_CARIBBEAN_ISLANDS)
        ].copy()

        restr_tax_deficits = pd.concat([restr_tax_deficits, UKI_extract], axis=0)
        restr_tax_deficits['tax_deficit'] = restr_tax_deficits['tax_deficit'].astype(float)

        # ### Preparing the unadjusted sales mapping --------------------------------------------------------------- ###

        restr_oecd_mapping = oecd_sales_mapping[
            [
                'PARENT_COUNTRY_CODE', 'AFFILIATE_COUNTRY_CODE', 'UNRELATED_PARTY_REVENUES'
            ]
        ].copy()

        if elim_negative_revenues:
            restr_oecd_mapping['UNRELATED_PARTY_REVENUES'] = restr_oecd_mapping['UNRELATED_PARTY_REVENUES'].map(
                lambda x: max(x, 0)
            )

        parent_totals = restr_oecd_mapping.groupby('PARENT_COUNTRY_CODE').sum().to_dict()['UNRELATED_PARTY_REVENUES']

        restr_oecd_mapping['URP_PERCENTAGE'] = (
            restr_oecd_mapping['UNRELATED_PARTY_REVENUES']
            / restr_oecd_mapping['PARENT_COUNTRY_CODE'].map(parent_totals)
        )

        restr_oecd_mapping = restr_oecd_mapping.drop(columns=['UNRELATED_PARTY_REVENUES'])

        # ### Preparing the adjusted sales mapping ----------------------------------------------------------------- ###

        restr_adjusted_mapping = adjusted_sales_mapping.groupby(
            ['PARENT_COUNTRY_CODE', 'OTHER_COUNTRY_CODE']
        ).agg(
            {
                'UNRELATED_PARTY_REVENUES': 'sum'
            }
        ).reset_index()

        if elim_negative_revenues:
            restr_adjusted_mapping['UNRELATED_PARTY_REVENUES'] = restr_adjusted_mapping['UNRELATED_PARTY_REVENUES'].map(
                lambda x: max(x, 0)
            )

        parent_totals = restr_adjusted_mapping.groupby(
            'PARENT_COUNTRY_CODE'
        ).sum().to_dict()['UNRELATED_PARTY_REVENUES']
        restr_adjusted_mapping['URP_PERCENTAGE'] = (
            restr_adjusted_mapping['UNRELATED_PARTY_REVENUES']
            / restr_adjusted_mapping['PARENT_COUNTRY_CODE'].map(parent_totals)
        )

        restr_adjusted_mapping = restr_adjusted_mapping.drop(columns=['UNRELATED_PARTY_REVENUES'])

        # ### Deducing the unadjusted distribution of revenues ----------------------------------------------------- ###

        merged_df = restr_oecd_mapping.merge(
            restr_tax_deficits,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='Parent jurisdiction (alpha-3 code)'
        ).drop(
            columns=[
                'Parent jurisdiction (alpha-3 code)',
                'Parent jurisdiction (whitespaces cleaned)'
            ]
        )

        merged_df['ATTRIBUTED_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        unadjusted_temp = merged_df.groupby('AFFILIATE_COUNTRY_CODE').sum()['ATTRIBUTED_REVENUES'].to_dict()

        # ### Deducing the adjusted distribution of revenues ------------------------------------------------------- ###

        merged_df = restr_adjusted_mapping.merge(
            restr_tax_deficits,
            how='left',
            left_on='PARENT_COUNTRY_CODE', right_on='Parent jurisdiction (alpha-3 code)'
        ).drop(
            columns=[
                'Parent jurisdiction (alpha-3 code)',
                'Parent jurisdiction (whitespaces cleaned)'
            ]
        )

        merged_df['ATTRIBUTED_REVENUES'] = merged_df['URP_PERCENTAGE'] * merged_df['tax_deficit']
        adjusted_temp = merged_df.groupby('OTHER_COUNTRY_CODE').sum()['ATTRIBUTED_REVENUES']

        return restr_tax_deficits.copy()
