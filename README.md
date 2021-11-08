# Context

This repository accompanies empirical work on the adjustment of revenue variables in the OECD's aggregated and anonymized [country-by-country data](https://stats.oecd.org/Index.aspx?DataSetCode=CBCR_TABLEI). In particular, it applies the methodology described in the [internship report](https://github.com/pechouc/destination-based-sales/blob/main/reports/Internship%20report%20-%20Revised%20version.pdf) of August 2021.

In this study, we explore a key limitation of multinational companies’ country-by-country data: as revenue variables reflect the tax jurisdiction of the affiliates that register the transactions and not the ultimate destination of goods or services, they are heavily distorted by the use of platform jurisdictions, from which sales are operated remotely.

- We first evidence these distortions based on unadjusted country-by-country data and highlight the disproportionate weight of a few small, low-tax countries in the distribution of US multinational firms’ unrelated-party revenues. Most of the code behind these analyses can be found in the `analyses.py` and `per_industry.py` modules.

- Second, we develop a methodology to adjust these revenue variables and approximate a destination-based mapping of sales. This relies on the articulation of the logic encapsulated in `irs.py`, `bea.py`, `trade_statistics.py` and `revenue_split.py`.

- Eventually, we propose a tentative extension of our methodology to non-US headquarter countries and re-estimate the revenue gains from Baraké et al. (2021)’s minimum tax unilateral implementation scenario. This is mostly achieved in `oecd_cbcr.py`, `analytical_amne.py`, `trade_statistics.py` and `global_sales_calculator.py`.

Further work on the topic is being conducted in the framework of a master thesis.

# Data

This research relies on the following data sources:

- the IRS' country-by-country data (cf. `irs.py` and `per_industry.py`, [link](https://www.irs.gov/statistics/soi-tax-stats-country-by-country-report));
- the OECD's aggregated and anonymized country-by-country data (cf. `oecd_cbcr.py`, [link](https://stats.oecd.org/Index.aspx?DataSetCode=CBCR_TABLEI));
- the BEA's data on the activities of US multinational companies (cf. `bea.py`, [link to 2018 statistics](https://www.bea.gov/worldwide-activities-us-multinational-enterprises-preliminary-2018-statistics));
- the Analytical AMNE database of the OECD (cf. `analytical_amne.py`, [link](https://www.bea.gov/worldwide-activities-us-multinational-enterprises-preliminary-2018-statistics));
- and the OECD's balanced trade statistics for trade in merchandise ([BIMTS](https://stats.oecd.org/Index.aspx?DataSetCode=BIMTS_CPA)) and services ([BATIS](https://stats.oecd.org/Index.aspx?DataSetCode=BATIS_EBOPS2010)), prepared in `trade_statistics.py`.

All these data sources, their use and their preprocessing are described in the report of August 2021.

# Documentation

A not-yet-complete documentation of the code is available online, following this link. It was built with [pdoc](https://pdoc3.github.io/pdoc/).

# Contact

I have started working on these topics for the report following my recent internship at the [EU Tax Observatory](https://www.taxobservatory.eu) and having entered the second year of the Master in Economics of Institut Polytechnique de Paris, I now pursue this research for my master thesis. For any remark or question, feel free to write to paul-emmanuel.chouc@ensae.fr.
