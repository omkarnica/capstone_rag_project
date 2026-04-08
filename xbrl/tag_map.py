"""
Canonical XBRL tag mappings for normalization.
"""

TAG_MAP = {
    # Revenue
    "Revenues": "Revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "Revenue",
    # Net income
    "NetIncomeLoss": "NetIncome",
    "ProfitLoss": "NetIncome",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "NetIncome",
    # Stockholders' equity
    "StockholdersEquity": "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "StockholdersEquity",
    # Operating income
    "OperatingIncomeLoss": "OperatingIncome",
    # Total assets
    "Assets": "TotalAssets",
}
