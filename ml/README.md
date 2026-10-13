# ml/ — demand forecasting

| File | Purpose |
|---|---|
| `forecast.csv` | **Committed output** (160 rows). The API only reads this file. |
| `backtest_report.json` | Backtest numbers quoted in DECISIONS.md. |
| `common.py` | Data loading and the weeks-to-Diwali feature. |
| `models.py` | The three baselines, the Poisson GLM, the gradient-boosting challenger and the shipped `ForecastModel`. |
| `backtest.py` | Rolling-origin backtest harness (`ml/backtest.py:20` is the leakage guard). |
| `train_forecast.py` | Re-trains on all history, rewrites `forecast.csv` and `backtest_report.json`. |

Re-run (needs `pip install -r ml/requirements.txt`; the Django app does not need these):

    python -m ml.train_forecast

Deterministic: no random seeds are involved in the shipped model. Takes about 15 seconds.
