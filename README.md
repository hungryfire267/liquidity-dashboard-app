# ASX 50 Liquidity Dashboard

A Streamlit app for ASX 50 liquidity and price-volume diagnostics.

The app analyzes two years of ASX 50 OHLCV-style trading history and calculates:

- volume versus absolute price-move correlation
- volume elasticity per 1 percentage point price move
- down-day versus up-day volume lift
- volatile-session versus quiet-session dollar-volume lift
- Amihud illiquidity
- dollar depth per 1% daily range
- impact proxy in bps per $10m traded
- volatility-adjusted dollar volume
- turnover concentration across the selected stock set
- scenario order notional, modeled impact, and participation of ADV

Run it with:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

If you are using the bundled Codex Python runtime:

```powershell
& "C:\Users\Gordon Li\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install -r requirements.txt
$env:PYTHONPATH=(Resolve-Path .packages).Path
& "C:\Users\Gordon Li\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m streamlit run app.py --global.developmentMode false --server.port 8501 --server.headless true
```

The app includes one stock filter. By default it analyzes the full ASX 50; when you filter to one or more stocks, every metric, answer, chart, and table updates to that selected set. The detailed price/volume charts use the highest average dollar-volume stock inside the current filter.
