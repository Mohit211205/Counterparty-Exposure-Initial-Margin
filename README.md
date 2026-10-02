# Counterparty Credit Exposure and Initial Margin Simulator

Simulates how much a counterparty could owe us in the future on two trades,
and how netting and collateral reduce that exposure.

## Trades with one counterparty
- 1-year **USD/INR FX forward** (we buy USD 1 million)
- 5-year **INR interest rate swap**, INR 10 crore notional (we pay fixed, receive floating, annual payments)

## Model
- **USD/INR**: geometric Brownian motion, volatility estimated from real historical data
- **INR short rate**: Vasicek model (parameters are my assumptions: a = 0.20, b = 6.5%, sigma = 1%, r0 = 6.5%)
- 5,000 scenarios, weekly time steps for 5 years
- Swap valued with Vasicek zero-coupon bond prices; fixed rate set so the swap starts at value 0
- Market assumptions: INR rate 6.5%, USD rate 4.5% (used for the forward)

## Exposure measures
- **EE** (expected exposure) = average of max(value, 0)
- **PFE 95% / 99%** (potential future exposure) = percentile of max(value, 0)
- **EPE** = average EE over time
- **CVA** = expected loss from the counterparty defaulting, using an assumed 150 bp credit spread and 60% loss given default
- Four set-ups: no netting, netting, netting + variation margin, netting + variation margin + initial margin

## Collateral
- **Variation margin**: counterparty posts the trade value as it was 2 weeks ago (margin period of risk of 10 business days)
- **Initial margin**: 99% expected shortfall of the 2-week increase in portfolio value
- Check: initial margin for the FX forward from real historical 10-day USD/INR moves vs from the simulation

## How to run
```
pip install -r requirements.txt
python get_real_data.py                 # downloads real data, saves usdinr.csv (needs internet)
python exposure_simulator.py --csv usdinr.csv
python exposure_simulator.py --test     # fake FX history, only checks that the code runs
```
Creates `exposure_report.xlsx` and `exposure_profile.png`.

## Results (USD/INR history from 2015 to 2 Oct 2026: spot 96.30, annual volatility 5.97%)
| Set-up | Peak EE (INR m) | EPE (INR m) | Peak PFE 95% (INR m) | Peak PFE 99% (INR m) | CVA (INR m) |
|---|---|---|---|---|---|
| No netting, no collateral | 3.23 | 1.14 | 11.35 | 15.62 | 0.072 |
| Netting only | 2.49 | 1.05 | 10.74 | 15.32 | 0.066 |
| Netting + variation margin | 0.46 | 0.09 | 2.52 | 4.10 | 0.006 |
| Netting + variation margin + initial margin | 0.01 | 0.00 | 0.00 | 0.48 | 0.000 |

- Initial margin for the portfolio (99% expected shortfall over 2 weeks): INR 3.62 million.
- Check on the FX forward alone: initial margin from real historical 10-day USD/INR moves is INR 3.32m, from the simulation INR 3.45m (about 4% apart).
- Variation margin cut peak PFE 99% from INR 15.3m to 4.1m (-73%). Adding initial margin cut it to 0.48m.
- Netting helped only a little here (15.6m to 15.3m), because the FX forward dominates the exposure in year 1 and the swap's value moves independently.
- The exposure drops sharply when the forward matures after year 1 and at each swap payment date.
- Full tables: `results/exposure_report.xlsx`, chart: `results/exposure_profile.png`. Random seed is fixed, so the numbers repeat.

## Limitations
- Interest rate parameters are assumptions, not calibrated to Indian market data
- FX and interest rates are assumed independent; one-way collateral (counterparty posts to us)
- Credit spread and loss given default are assumptions; constant discount rates for the forward; flat thresholds and zero minimum transfer amount
