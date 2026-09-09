# Adaptive Electricity Forecasting

This project keeps the existing adaptive forecasting architecture but reorients it around the electricity challenge.

## Problem statement

The forecasting target is the electricity `target` column in `data/electricity.csv`. The system must predict future target values from historical timestamp and feature information while preserving a strict no-lookahead rule: the target is hidden during prediction and only revealed after the prediction is made.

## Dataset

The dataset is expected to live at `data/electricity.csv` and must contain the following columns:

- `timestamp`
- `feature_1`
- `feature_2`
- `feature_3`
- `feature_4`
- `feature_5`
- `feature_6`
- `target`

The project treats all `feature_*` columns as numeric inputs and does not assign arbitrary business meaning to them beyond that.

## PySpark pipeline

The implementation uses PySpark for the ingestion and validation pipeline. A SparkSession is created through the `ElectricityDataLoader` in `src/spark_loader.py`, which:

- loads the CSV with a header,
- validates the expected schema,
- parses the date column using `dd-MM-yyyy`,
- converts numeric features and target to numeric types,
- detects null and invalid values,
- sorts the observations chronologically,
- adds temporal features such as year, month, day, day_of_week, and week_of_year,
- rejects empty or malformed data before model fitting.

## Forecasting approach

The adaptive forecasting loop reuses the existing ensemble/monitor/adaptation structure for a numeric target sequence. The forecasting engine is kept generic and works with sequential target values while the feature engineering layer supplies the date and feature context.

## No-data-leakage strategy

Predictions are generated from the current features and historical values only. The target is never used as a feature during prediction. The actual target is only consumed after prediction to compute an error, update monitoring, and trigger adaptation if the drift signal shows persistent behavior.

## Adaptive loop

The project follows the required cycle:

1. current features are supplied,
2. target remains hidden,
3. prediction is generated,
4. actual target is revealed,
5. prediction error is computed,
6. drift is monitored,
7. persistent change triggers adaptation,
8. the system continues forecasting with the updated model state.

## Evaluation and SC1/SC2 support

The existing evaluation client and mock session logic remain in place and continue to support the sequential hidden-target workflow without automatically starting the real external evaluation server.

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

Run the dashboard:

```bash
streamlit run dashboard\dashboard.py
```

## Notes

The workspace currently contains an empty `data/electricity.csv` file. The project code validates the dataset and will not fabricate or invent rows. The loader is built to process the actual dataset once it is populated with the expected electricity fields.



The monitoring system identifies that this is not just a single abnormal observation.



After confirming a persistent change, the forecasting system adapts to the new demand pattern and produces forecasts based on the updated behavior.



*9. BUSINESS IMPACT*



An adaptive retail demand forecasting system can help businesses:



Reduce product stockouts

Reduce excess inventory

Improve inventory planning

Reduce product wastage

Improve purchasing decisions

Respond faster to changing customer demand

Improve operational efficiency



*10. SYSTEM ARCHITECTURE*



The system consists of four major components:



1\. Forecasting Module



Generates future demand predictions using historical and currently available data.



2\. Monitoring and Change Detection Module



Continuously evaluates forecasting errors and identifies meaningful changes.



3\. Adaptation Module



Updates or retrains the forecasting strategy when persistent change is confirmed.



4\. Dashboard and API Module



Processes evaluation data in real time and displays the current system status, predictions, errors, change scores, and adaptation events.



*11. DASHBOARD*



The dashboard is designed to display:



Current forecast

Actual observation

Forecast error

Change or uncertainty score

System status

Adaptation status

Adaptation history



The dashboard provides a real-time view of how the forecasting system responds to changing retail demand.



*12. TECHNOLOGY STACK*



Programming Language:

Python



Libraries and Tools:

Pandas

NumPy

Scikit-learn

Streamlit



The system will also integrate with the provided evaluation API for real-time evaluation.



*13. KEY FEATURES*



Real-time demand forecasting

Continuous performance monitoring

Explicit change detection

Adaptive forecasting

Protection against temporary anomalies

No future data leakage

Real-time dashboard

Adaptation event logging

Programmatic predictions



*14. IMPORTANT DESIGN PRINCIPLE*



The system follows a strict no-future-data-leakage principle.



For every prediction:



1\. Use only the information currently available.

2\. Generate the forecast.

3\. Wait for the actual value.

4\. Calculate the forecast error.

5\. Monitor the error and demand pattern.

6\. Adapt the system only when meaningful change is confirmed.

7\. Generate the next prediction.



This ensures that the forecasting system does not use future information while making predictions.



*15. EXPECTED OUTCOME*



The expected outcome is an adaptive retail demand forecasting system capable of:



Predicting future product demand

Monitoring its own forecasting reliability

Detecting meaningful changes in demand patterns

Ignoring short-lived abnormalities

Adapting to persistent changes

Providing real-time visualization through a dashboard



The final system will demonstrate how time-series forecasting can remain useful even when real-world retail demand patterns change over time.





