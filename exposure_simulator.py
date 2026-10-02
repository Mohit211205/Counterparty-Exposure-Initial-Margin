"""
Project 2: Counterparty credit exposure and initial margin simulator
Trades with one counterparty: a 1-year USD/INR FX forward and a 5-year INR interest rate swap.
We simulate market paths, value the trades along every path, and measure the exposure
(EE, PFE) with and without netting and collateral.

Run:  python exposure_simulator.py                   (USD/INR history from Yahoo Finance)
      python exposure_simulator.py --csv usdinr.csv  (your own file: Date column + a USD/INR close column)
      python exposure_simulator.py --test            (fake FX history, only to test the code)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------ settings
N_PATHS = 5000            # number of simulated scenarios
YEARS = 5                 # length of the longest trade (swap)
DT = 1 / 52               # weekly time steps
STEPS = int(round(YEARS / DT))
MPOR = 2                  # margin period of risk in steps (2 weeks = about 10 business days)

# FX forward: we BUY 1 million USD in 1 year at today's forward price
FX_NOTIONAL_USD = 1_000_000
FX_MATURITY_STEPS = 52
R_DOM = 0.065             # INR interest rate (assumption)
R_FOR = 0.045             # USD interest rate (assumption)

# Interest rate swap: we PAY fixed, RECEIVE floating, annual payments, 5 years
SWAP_NOTIONAL = 100_000_000   # 10 crore INR
PAY_STEPS = [52, 104, 156, 208, 260]

# Counterparty credit quality (assumptions, used for CVA)
CDS_SPREAD = 0.015        # 150 basis points
LGD = 0.60                # loss given default

# Vasicek model for the INR short rate: dr = a (b - r) dt + sigma dW   (assumed parameters)
A, B, SIGMA_R, R0 = 0.20, 0.065, 0.010, 0.065


# ------------------------------------------------------------ 1. historical FX data
def get_usdinr(use_fake_data):
    if "--csv" in sys.argv:
        df = pd.read_csv(sys.argv[sys.argv.index("--csv") + 1], index_col=0, parse_dates=True).sort_index()
        cols = [c for c in df.columns if "USDINR" in str(c).upper().replace("/", "").replace("=X", "")]
        col = cols[0] if cols else df.columns[0]      # otherwise use the first column
        return df[col].dropna().astype(float), False
    if not use_fake_data:
        try:
            import yfinance as yf
            px = yf.download("USDINR=X", start="2015-01-01", auto_adjust=True, progress=False)["Close"]
            px = px.squeeze().dropna()
            if len(px) > 500:
                return px, False
        except Exception as e:
            print("Download failed:", e)
        print("Using fake data instead.")
    fake = 70 * np.exp(np.cumsum(np.random.normal(0.0001, 0.04 / np.sqrt(252), 2500)))
    return pd.Series(fake, index=pd.bdate_range("2015-01-01", periods=2500)), True


# ------------------------------------------------------------ 2. simulate market paths
def simulate_fx(s0, vol):
    """USD/INR as geometric Brownian motion (risk-neutral drift = R_DOM - R_FOR)."""
    z = np.random.standard_normal((N_PATHS, STEPS))
    log_step = (R_DOM - R_FOR - 0.5 * vol ** 2) * DT + vol * np.sqrt(DT) * z
    s = np.zeros((N_PATHS, STEPS + 1))
    s[:, 0] = s0
    s[:, 1:] = s0 * np.exp(np.cumsum(log_step, axis=1))
    return s

def simulate_rates():
    """Vasicek short rate, simulated with its exact discrete-time formula."""
    r = np.zeros((N_PATHS, STEPS + 1))
    r[:, 0] = R0
    decay = np.exp(-A * DT)
    sd = SIGMA_R * np.sqrt((1 - np.exp(-2 * A * DT)) / (2 * A))
    for i in range(STEPS):
        r[:, i + 1] = r[:, i] * decay + B * (1 - decay) + sd * np.random.standard_normal(N_PATHS)
    return r


# ------------------------------------------------------------ 3. valuation
def vasicek_bond(r, tau):
    """Price of a zero-coupon bond paying 1 after tau years, given short rate r."""
    if tau <= 0:
        return np.ones_like(r)
    b_tau = (1 - np.exp(-A * tau)) / A
    a_tau = np.exp((B - SIGMA_R ** 2 / (2 * A ** 2)) * (b_tau - tau) - SIGMA_R ** 2 * b_tau ** 2 / (4 * A))
    return a_tau * np.exp(-b_tau * r)

def value_fx_forward(s):
    """Value (in INR) of the forward we bought, at every step of every path."""
    strike = s[0, 0] * np.exp((R_DOM - R_FOR) * FX_MATURITY_STEPS * DT)   # forward price today
    v = np.zeros_like(s)
    for i in range(FX_MATURITY_STEPS + 1):
        tau = (FX_MATURITY_STEPS - i) * DT
        forward_now = s[:, i] * np.exp((R_DOM - R_FOR) * tau)
        v[:, i] = FX_NOTIONAL_USD * (forward_now - strike) * np.exp(-R_DOM * tau)
    return v          # after maturity the trade is gone, so the value stays 0

def value_swap(r):
    """Value to us of the pay-fixed / receive-floating swap, at every step of every path.
    The floating coupon for each year is fixed at the start of that year (at the short rate seen then),
    so between payment dates:
        floating leg = N * [ (1 + L) * P(next payment) - P(last payment) ]
        fixed leg    = N * K * sum of P(each remaining payment)
    where L is the floating rate set at the last reset date and P(.) are Vasicek bond prices.
    Value shown is just after any payment made on that date."""
    reset_steps = [0] + PAY_STEPS[:-1]
    # fixed rate K is set so the swap is worth 0 today
    annuity0 = sum(vasicek_bond(r[:, 0], p * DT) for p in PAY_STEPS)
    k = (1 - vasicek_bond(r[:, 0], PAY_STEPS[-1] * DT)) / annuity0
    v = np.zeros_like(r)
    for i in range(STEPS + 1):
        remaining = [p for p in PAY_STEPS if p > i]
        if not remaining:
            continue
        last_reset = max(x for x in reset_steps if x <= i)
        accrual = (remaining[0] - last_reset) * DT                     # length of the current period (1 year)
        floating_rate = (1 / vasicek_bond(r[:, last_reset], accrual) - 1) / accrual
        p_next = vasicek_bond(r[:, i], (remaining[0] - i) * DT)
        p_last = vasicek_bond(r[:, i], (remaining[-1] - i) * DT)
        floating = (1 + floating_rate * accrual) * p_next - p_last
        annuity = sum(vasicek_bond(r[:, i], (p - i) * DT) for p in remaining)
        v[:, i] = SWAP_NOTIONAL * (floating - k[0] * annuity)
    return v


# ------------------------------------------------------------ 4. exposure measures
def exposure_profile(exposure):
    """Takes exposure (paths x steps, already >= 0) and returns EE, PFE95, PFE99 per step."""
    return pd.DataFrame({"EE": exposure.mean(axis=0),
                         "PFE95": np.percentile(exposure, 95, axis=0),
                         "PFE99": np.percentile(exposure, 99, axis=0)})

def cva(ee):
    """Credit valuation adjustment = expected loss from the counterparty defaulting.
    Sum over time of: LGD x expected exposure x discount factor x probability of default in that step.
    Default probability comes from a constant hazard rate = spread / LGD."""
    hazard = CDS_SPREAD / LGD
    t = np.arange(STEPS + 1) * DT
    survival = np.exp(-hazard * t)
    default_prob = survival[:-1] - survival[1:]
    avg_ee = (ee[:-1] + ee[1:]) / 2
    discount = np.exp(-R_DOM * (t[:-1] + t[1:]) / 2)
    return LGD * np.sum(avg_ee * discount * default_prob)

def initial_margin(v):
    """IM = 99% expected shortfall of the INCREASE in value over the margin period of risk
    (our exposure grows when the trade value goes up)."""
    change = (v[:, MPOR:] - v[:, :-MPOR]).ravel()
    cutoff = np.percentile(change, 99)
    return change[change >= cutoff].mean()

def collateralised_exposure(v, im=0.0):
    """We hold variation margin equal to the value from MPOR steps ago (counterparty posts when we are in the money),
    plus the initial margin. Exposure is what is left uncovered."""
    old_v = np.hstack([np.zeros((v.shape[0], MPOR)), v[:, :-MPOR]])
    collateral = np.maximum(old_v, 0) + im
    return np.maximum(v - collateral, 0)


# ------------------------------------------------------------ 5. main
def main():
    usdinr, fake = get_usdinr("--test" in sys.argv)
    log_ret = np.log(usdinr / usdinr.shift(1)).dropna()
    vol = log_ret.std() * np.sqrt(252)            # annual volatility from history
    s0 = float(usdinr.iloc[-1])
    print(f"USD/INR spot {s0:.2f}, historical annual volatility {100 * vol:.2f}%")

    s = simulate_fx(s0, vol)
    r = simulate_rates()
    v_fx = value_fx_forward(s)
    v_swap = value_swap(r)
    v_net = v_fx + v_swap                       # one netting agreement covers both trades
    t = np.arange(STEPS + 1) * DT

    # exposure under four set-ups
    im = initial_margin(v_net)
    cases = {
        "No netting, no collateral": np.maximum(v_fx, 0) + np.maximum(v_swap, 0),
        "Netting only": np.maximum(v_net, 0),
        "Netting + variation margin": collateralised_exposure(v_net),
        "Netting + variation margin + initial margin": collateralised_exposure(v_net, im),
    }
    profiles = {name: exposure_profile(e) for name, e in cases.items()}
    summary = pd.DataFrame({
        "Set-up": list(cases),
        "Peak EE (INR m)": [round(p["EE"].max() / 1e6, 2) for p in profiles.values()],
        "EPE (INR m)": [round(p["EE"].mean() / 1e6, 2) for p in profiles.values()],
        "Peak PFE 95% (INR m)": [round(p["PFE95"].max() / 1e6, 2) for p in profiles.values()],
        "Peak PFE 99% (INR m)": [round(p["PFE99"].max() / 1e6, 2) for p in profiles.values()],
        "CVA (INR m)": [round(cva(p["EE"].values) / 1e6, 3) for p in profiles.values()]})
    print("\n", summary.to_string(index=False))
    print(f"\nInitial margin (99% ES over {MPOR} weeks): INR {im / 1e6:.2f} million")

    # historical check of the FX initial margin: use real 10-day USD/INR moves instead of the model
    ten_day = np.log(usdinr.shift(-10) / usdinr).dropna()
    fx_change = FX_NOTIONAL_USD * s0 * (np.exp(ten_day.values) - 1)
    tail = fx_change[fx_change >= np.percentile(fx_change, 99)]
    hist_im_fx = tail.mean()
    model_im_fx = initial_margin(v_fx)
    print(f"FX forward only - initial margin from history: INR {hist_im_fx / 1e6:.2f}m, "
          f"from simulation: INR {model_im_fx / 1e6:.2f}m")

    # charts
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    p = profiles["Netting only"]
    ax[0].plot(t, p["EE"] / 1e6, label="EE")
    ax[0].plot(t, p["PFE95"] / 1e6, label="PFE 95%")
    ax[0].plot(t, p["PFE99"] / 1e6, label="PFE 99%")
    ax[0].set_title("Exposure profile (netted portfolio)" + (" - FAKE FX DATA" if fake else ""))
    ax[0].set_xlabel("Years"); ax[0].set_ylabel("INR million"); ax[0].legend()
    for name, p in profiles.items():
        ax[1].plot(t, p["PFE99"] / 1e6, label=name)
    ax[1].set_title("PFE 99%: effect of netting and collateral")
    ax[1].set_xlabel("Years"); ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("exposure_profile.png", dpi=150)

    # excel report
    with pd.ExcelWriter("exposure_report.xlsx") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        sheet_names = ["No netting", "Netting", "Netting + VM", "Netting + VM + IM"]
        for sheet, p in zip(sheet_names, profiles.values()):
            out = (p / 1e6).round(3)
            out.insert(0, "Year", t.round(3))
            out.to_excel(xl, sheet_name=sheet, index=False)
    print("\nSaved exposure_report.xlsx and exposure_profile.png")
    if fake:
        print("WARNING: fake FX data used. Do not quote these numbers anywhere.")


if __name__ == "__main__":
    main()
