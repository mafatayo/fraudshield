import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_DIR = os.path.join(ROOT, "system", "model")
RUN_DIR = os.path.join(ROOT, "system", "runs")
RESULTS = os.path.join(ROOT, "results")

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FEATURES = ["amount", "hour", "day", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
            "newbalanceDest", "errBalOrig", "errBalDest", "origZeroedOut", "destIsMerchant",
            "amountToBalanceRatio", "origPriorTxnCount", "origTimeSinceLast",
            "destPriorTxnCount", "destTimeSinceLast", "destPriorAmtMean", "amountOverDestMean",
            "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
RECENT_CHARS = 160  # the recent-identifier list is capped at 160 characters: about sixteen identifiers
