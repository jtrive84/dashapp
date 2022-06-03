import time
import datetime
import os
import os.path
import re
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import xlrd




datestamp_    = datetime.datetime.now().strftime("%Y%m%d")
DATA_PATH     = "C:\\Users\\cac9159\\Repos\\LTC\\FWA\\Datasets\\TRAINING.csv"
xlsxpath      = "C:\\Users\\cac9159\\Repos\\LTC\\FWA\\Datasets\\JK_FWA_Variable_Assessment_Preliminary_Ranking_20190722.xlsx"
dfall         = pd.read_csv(DATA_PATH)
dfall.columns = [i.upper().strip() for i in dfall.columns.tolist()]
dfall.columns = [re.sub(r"\s{1,}", "_", i) for i in dfall.columns.tolist()]

RESPONSE_   = "FRAUD_INDICATOR"
dropcols_   = ["INSURED_TO_INCURRED_AGE",
               "YEAR_DLR", "MNTH_DLR", "DT_LOSS", "EXIT_DATE_FROM_STUDY",
               "ISSUE_YEAR", "YEAR_INCURRED", "MNTH_INCURRED", "MNTH_TERM",
               "ISSUE_STATE", "POLICY_FORM_NUMBER", "AGENCY_GROUP_CODE",
               "AGE_AT_CLAIM_BANDED", "AGE_AT_CLAIM", "ISSUE_AGE",
               "CREATE_DATE_TS", "STUDY_DATE",]

dropcols = [i for i in dropcols_ if i in dfall.columns]
dfinit   = dfall.drop(dropcols_, axis=1)

continuous_ = ["CLAIM_DURATION", "DAILY_BENEFIT_INFLATED", "POLICY_YEAR",
               "PAID_AMOUNT", "DLR_AMT", "ANNUAL_PREMIUM", "ATTAINED_AGE",
               "EXPOSURE_DAYS",]

categorical_ = list(set(dfinit.columns.tolist()).difference(set(continuous_)))


# Encode missing values ======================================================]
dfinit["RATE_INCREASE_INDICATOR"] = dfinit["RATE_INCREASE_INDICATOR"].astype(str)
dfinit["RATE_INCREASE_INDICATOR"] = dfinit["RATE_INCREASE_INDICATOR"].map(lambda v: "N" if v=="nan" else v)


# Consolidate levels of target variates ======================================]
dfinit = dfinit.rename({"REPEATED_CALLS":"REPEATED_CALLS_INIT"}, axis=1)
dfinit["REPEATED_CALLS"] = dfinit["REPEATED_CALLS_INIT"].map(lambda v: "N" if v<1 else "Y")
dfinit.drop("REPEATED_CALLS_INIT", axis=1, inplace=True)


# Convert fields to continuous numeric types =================================]
for var_ in continuous_:
    dfinit[var_] = dfinit[var_].astype(np.float_)

for var_ in categorical_:
    dfinit[var_] = dfinit[var_].astype(object)



# Purge singular columns =====================================================]
for var_ in set(dfinit.columns).difference({"FRAUD_INDICATOR"}):
    if dfinit[var_].value_counts().size<=1:
        dfinit = dfinit.drop(var_, axis=1)


# predictors_ = ["DLR_AMT", "SITUS_SVC", "POLICY_TERM_RSN", "COVERAGE_TYPE",
#                "DAILY_BENEFIT_BANDED", "GENDER", "HHC_PERCENT_BANDED",
#                "INFLATION_LIFETIME",
#                "POOL_OF_MONEY_BENEFIT", "PREMIUM_PAYMENT_TYPE",
#                "RATE_INCREASE_INDICATOR", "DUAL_WAIVER", "SURVIVORSHIP",
#                "TAX_QUALIFIED_STATUS",
#                "DIAGNOSIS", "ELIM_PERIOD_BANDED", "ICOS_AMT", "PAID_UP",
#                "PREMIUM_WAIVED",]

# predictors_ = ["REPEATED_CALLS", "ANNUAL_PREMIUM", "ATTAINED_AGE", "POOL_OF_MONEY_BENEFIT",
#                "PAID_AMOUNT", "TAX_QUALIFIED_STATUS", "ELIM_PERIOD_BANDED", "ICOS_AMT",
#                "PREMIUM_WAIVED", "INDEMNITY_VS_EXPENSE_INCURRED", "DUAL_WAIVER",
#                "PREMIUM_PAYMENT_TYPE", "COVERAGE_TYPE", "DAILY_BENEFIT_INFL_BANDED",
#                "RESIDENT_STATE",]

predictors_ = [
    "ATTAINED_AGE_BANDED", "BENEFIT_PERIOD", "BENEFIT_TRIGGER_OPTIONS",
    "COLI", "DAILY_BENEFIT_INFL_BANDED", "DUAL_WAIVER",
    "INDEMNITY_VS_EXPENSE_INCURRED", "LINKED_POLICY_INDICATOR",
    "REPEATED_CALLS", "PREMIUM_PAYMENT_MODE", "PREMIUM_WAIVED",
    "RESTORATION_OF_BENEFITS", "SITUS_CURRENT", "TAX_QUALIFIED_STATUS",
    "UNDERWRITING_CLASS", "ELIM_PERIOD_BANDED", "PAID_AMOUNT",
    "DLR_AMT", "RESIDENT_STATE",
    ]


dfall = dfinit.copy(deep=True)
dfall = dfall[["CLAIM_NUMBER", "POLICY_NUMBER", "RESIDENT_STATE"] + predictors_ + [RESPONSE_]]

dfall["score"] = 2 * np.random.rand(dfall.shape[0]) - 1
min_score_ = dfall["score"].min()
dfall["priority"] = (dfall["score"] / min_score_)
dfall["priority"] = (dfall["priority"] + 1) * 50
dfall["priority"] = dfall["priority"].round(2)
dfall["score"]    = dfall["score"].round(5)

# Encode ELIM_PERIOD_BANDED:
lkp_ = {
    "90":"(90 Days)", ">0 & <=30":"(0-30 Days)", "0":"(0 Days)",
    ">90":"(91+ Days)", "60":"(60 Days)",
    }

dfall["ELIM_PERIOD_BANDED"] = dfall["ELIM_PERIOD_BANDED"].map(lkp_)



xlsx = pd.ExcelFile(xlsxpath)
sheets_ = [i for i in xlsx.sheet_names if i!="Instructions"]

vdict = {}
for sheet_ in sheets_:
    vdict[sheet_] = {}
    # sheet_ = sheets_[1]
    dfinit_ = xlsx.parse(sheet_)
    if "METRIC" in dfinit_.columns:
        vdict[sheet_]["vartype"] = "continuous"
        vdict[sheet_]["data"] = None
    else:
        vdict[sheet_]["vartype"] = "categorical"
        keep_ = [sheet_, "Initial Rank", "Final Rank", "Notes"]
        vdict[sheet_]["vartype"] = "categorical"
        vdict[sheet_]["data"] = dfinit_[keep_]









# lku  = {
#     "BASE"       : xlsx.parse("BASE"),
#     "ZIP_FCT"    : xlsx.parse("ZIP_FCT"),
#     "EXPOS_FCT"  : xlsx.parse("EXPOS_FCT"),
#     "AGE_BLD_FCT": xlsx.parse("AGE_BLD_FCT"),
#     "DED_FCT"    : xlsx.parse("DED_FCT"),
#     "CONSTR_FCT" : xlsx.parse("CONSTR_FCT"),
#     "COINS_FCT"  : xlsx.parse("COINS_FCT"),
#     }















