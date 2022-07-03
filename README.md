# Context

## Motivation & Overview

This repository accompanies empirical work on the adjustment of revenue variables in the OECD's aggregated and anonymized [country-by-country data](https://stats.oecd.org/Index.aspx?DataSetCode=CBCR_TABLEI). In particular, it applies the methodology described in this [internship report](https://github.com/pechouc/destination-based-sales/blob/main/reports/Internship%20report%20-%20Revised%20version.pdf) of August 2021 and more recently, in the [Online Appendix](https://github.com/pechouc/destination-based-sales) of my master thesis (June 2022).

In this study, we explore a key limitation of multinational companies’ country-by-country data: as revenue variables reflect the tax jurisdiction of the affiliates that register the transactions and not the ultimate destination of goods or services, they are heavily distorted by the use of platform jurisdictions, from which sales are operated remotely.

- We first evidence these distortions based on unadjusted country-by-country data and highlight the disproportionate weight of a few small, low-tax countries in the distribution of US multinational firms’ unrelated-party revenues. Most of the code behind these analyses can be found in the `analyses_provider.py` and `per_industry.py` modules.

- Second, we develop a methodology to adjust these revenue variables and approximate a destination-based mapping of sales. This relies on the articulation of the logic encapsulated in `irs.py`, `bea.py`, `trade_statistics.py` and `sales_calculator.py`. Trade statistics are prepared in `comtrade.py`, `balanced_trade.py` and `bop.py`.

- Eventually, we propose a tentative extension of our methodology to non-US headquarter countries and re-estimate the revenue gains from Baraké et al. (2021)’s minimum tax unilateral implementation scenario. Additional computations are required and they are encapsulated in `oecd_cbcr.py`, `analytical_amne.py` and `sales_calculator.py`.

## Data

This research relies on the following data sources:

- the IRS' country-by-country data (cf. `irs.py` and `per_industry.py`, [link](https://www.irs.gov/statistics/soi-tax-stats-country-by-country-report));
- the OECD's aggregated and anonymized country-by-country data (cf. `oecd_cbcr.py`, [link](https://stats.oecd.org/Index.aspx?DataSetCode=CBCR_TABLEI));
- the BEA's data on the activities of US multinational companies (cf. `bea.py`, [link to 2018 statistics](https://www.bea.gov/worldwide-activities-us-multinational-enterprises-preliminary-2018-statistics));
- the OECD's balanced trade statistics for trade in merchandise ([BIMTS](https://stats.oecd.org/Index.aspx?DataSetCode=BIMTS_CPA)) and services ([BATIS](https://stats.oecd.org/Index.aspx?DataSetCode=BATIS_EBOPS2010)), prepared in `balanced_trade.py`;
- the UN Comtrade data on trade in goods (cf. `comtrade.py`, [data portal](https://comtrade.un.org/data/));
- the trade statistics included in the US balance of payments (cf. `bop.py`, [link to BEA data portal](https://apps.bea.gov/iTable/iTable.cfm?reqid=62&step=6&isuri=1&tablelist=30164&product=1));
- and the Analytical AMNE database of the OECD (cf. `analytical_amne.py`, [link](https://www.bea.gov/worldwide-activities-us-multinational-enterprises-preliminary-2018-statistics)).

All these data sources, their use and their preprocessing are described in the report of August 2021 or in the more recent master thesis and its appendix.

# Use This Work!

## Accessing the Adjusted Database

There are two main possibilities to access the data:

- The adjusted revenue variables can directly be downloaded as `.csv` files from [this folder](https://github.com/pechouc/dbs_api/tree/main/dbs_api/outputs) of the associated repository `dbs_api`. The files are described in a dedicated `README`.

- The dedicated [API](https://dbs-api.herokuapp.com) also allows to obtain the adjusted database in `JSON` format. Note that, for now, the API simply serves the files mentioned just above: the two sources are therefore fully aligned. The API is described in more details in [this repository](https://github.com/pechouc/dbs_api) and the documentation is available here.

## Using the Code

The adjusted data files available online for download or via the API correspond to the benchmark case that I describe in my master thesis and the associated appendix. You may typically be interested in alternative methodologies or in running some robustness checks, in which case you need to re-use the code.

For Python users, the code has been built as a standalone package to facilitate its installation and use. To install it, you can run either this command:

```
pip install git+https://github.com/pechouc/destination-based-sales.git
```

Or you can clone this repository and install the `destination_based_sales` package with the following commands:

```
git clone git@github.com:pechouc/destination-based-sales.git
cd destination_based_sales
pip install -r requirements.txt
pip install .
```

Note that in both cases, the data required to run the code locally and stored in this repository will be downloaded (some files are relatively heavy). Note also that you can run a series of tests with the `make test` command (roughly 20-25 minutes).

## Documentation

A not-yet-complete documentation of the code is available online, following [this link](https://pechouc.github.io/destination-based-sales/index.html). It was built with [pdoc](https://pdoc3.github.io/pdoc/).

# Contact

I have started working on these topics for the report following my recent internship at the [EU Tax Observatory](https://www.taxobservatory.eu). During my second year in the Master in Economics of Institut Polytechnique de Paris, I have pursued this research for my master thesis. The main document can be found here and the associated appendix can be downloaded here. For any remark or question, feel free to write to paul-emmanuel.chouc@ensae.fr or to open issues directly on GitHub. Advice to improve the code or push this project further is also very welcome!
