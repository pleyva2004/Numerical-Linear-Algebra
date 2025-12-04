# EIA Hourly Demand Download Steps

1. **Open the API Dashboard**
   - Go to <https://www.eia.gov/opendata/apps/api/#/electricity>.
   - Click “API Dashboard”.

2. **Choose the data route**
   - In the three “Select route” dropdowns pick:
     1. `Electricity`
     2. `Electric Power Operations (Daily And Hourly)`
     3. `Hourly Demand, Demand Forecast, Generation, and Interchange`

3. **Set frequency and date window**
   - Frequency: `Hourly (UTC)`.
   - Start date: 31 days ago (e.g., `2025-10-27`).
   - End date: today (e.g., `2025-11-26`).

4. **Filter by balancing authority (region)**
   - Expand “Filter by Facet”.
   - Choose `Region` and select the code you want (e.g., `US48`, `NYIS`, `CAISO`).

5. **Sort the results**
   - Expand “Sort / Order”.
   - Add a sort on `Time Period` with direction `Ascending`.

6. **Submit and download**
   - Click `Submit`.
   - When results appear, use the download icon → CSV.
   - Save as `data/electricity_demand.csv` inside this repo.

7. **Refresh the notebook**
   - Re-open `w12.ipynb`.
   - Run cells up through Homework 9-2 to plot the updated data.

Tips:
- If you need more than 31 days, increase the date range (each request returns up to 5,000 rows).
- The same CSV works for other regions; just switch the `Region` filter.

