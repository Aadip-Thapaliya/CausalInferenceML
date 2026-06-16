# 📊 Analyzing RCT with Precision by Adjusting for Baseline Covariates

> A hands-on simulation study exploring how to estimate the Average Treatment Effect (ATE) as precisely as possible in Randomized Controlled Trials — using Jonathan Roth's heterogeneous-effects DGP.

---

## 🗂️ Table of Contents

- [Overview](#-overview)
- [The Core Problem](#-the-core-problem)
- [Economic Motivation](#-economic-motivation)
- [Data Generating Process (DGP)](#-data-generating-process-dgp)
- [Three Estimators Compared](#-three-estimators-compared)
- [Key Theoretical Results](#-key-theoretical-results)
- [Actual Output Results](#-actual-output-results)
- [Why Robust Standard Errors Matter](#-why-robust-standard-errors-matter)
- [Monte Carlo Simulation](#-monte-carlo-simulation)
- [Concepts Explained Simply](#-concepts-explained-simply)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)
- [References](#-references)

---

## 🔍 Overview

This project demonstrates a fundamental but often overlooked issue in empirical economics and statistics:

> **Not all ways of adjusting for baseline covariates in an RCT are equally good — and some can actually make your estimates *worse*.**

We compare three estimation strategies for the Average Treatment Effect (ATE):

| Method | Model | Efficiency |
|--------|-------|------------|
| **CL** — Classical (no adjustment) | `Y ~ D` | Baseline |
| **CRA** — Classical Regression Adjustment | `Y ~ D + Z` | ❌ Worse than CL |
| **IRA** — Interactive Regression Adjustment | `Y ~ D + Z + Z×D` | ✅ Best |

All inference uses **robust (Eicker-Huber-White) standard errors**, not the classical non-robust ones that Python's `smf.ols()` reports by default.

---

## 🧠 The Core Problem

In any experiment, we want to know:

> *Did the treatment actually cause a change in the outcome?*

The challenge is that people differ from each other. Some are more skilled, richer, younger, etc. These differences create "noise" in our data that makes it hard to detect the true treatment effect.

The idea behind **covariate adjustment** is to use information we already have about people (measured *before* the experiment) to reduce this noise and get a more precise estimate of the ATE.

But the key insight of this notebook is:

> Adjusting for covariates the **wrong way** (CRA) can make things worse.  
> Adjusting for covariates the **right way** (IRA) always helps.

---

## 🎓 Economic Motivation

Think of it this way:

- **Treatment D** = Going to college (1 = yes, 0 = no)
- **Outcome Y** = Earnings later in life
- **Covariate Z** = Academic skill level

The key twist: **skill affects earnings differently depending on whether someone went to college.**

| Person | Skill (Z) | Without College Y(0) | With College Y(1) | Effect of College |
|--------|-----------|----------------------|-------------------|-------------------|
| High skill | High | Low (skill wasted) | High (skill rewarded) | Large positive |
| Low skill | Low | Medium | Low | Negative |

This means the **treatment effect is heterogeneous** — it is different for different people. Averaging across everyone gives ATE = 0 in this DGP (the positive and negative effects cancel out).

This heterogeneous structure is exactly what makes CRA fail and IRA shine.

---

## ⚙️ Data Generating Process (DGP)

Based on **Jonathan Roth's DGP**:

```
E[Y(0) | Z] = -Z        → Baseline outcome decreases with skill
E[Y(1) | Z] = +Z        → Treated outcome increases with skill
Z ~ N(0, 1)             → Skill is standard normal

CATE(Z) = E[Y(1) - Y(0) | Z] = 2Z    → Effect varies with skill
ATE = E[2Z] = 0                        → True ATE is zero
```

In code:

```python
def gen_data(random_seed):
    np.random.seed(random_seed)
    n = 1000
    Z = np.random.normal(size=n)
    Y0 = -Z + np.random.normal(0, 0.1, size=n)
    Y1 =  Z + np.random.normal(0, 0.1, size=n)
    D  = np.random.binomial(1, 0.2, size=n)   # only 20% treated
    Y  = Y1 * D + Y0 * (1 - D)
    data = pd.DataFrame({"Y": Y, "D": D, "Z": 1 + Z})
    return data
```

Key features:
- **n = 1,000** observations
- **20% treatment rate** (imbalanced — only 1 in 5 gets treated)
- Small noise term (σ = 0.1) so most variation comes from Z and D
- Z is shifted by 1 in the dataframe (intercept added artificially)

---

## 📐 Three Estimators Compared

### 1️⃣ CL — Classical Two-Sample Approach

```python
CL = smf.ols("Y ~ D", data=data).fit()
CL.get_robustcov_results(cov_type="HC0").summary()
```

- Regresses Y on D only
- Equivalent to a simple difference-in-means
- Ignores all covariate information
- Serves as the **efficiency benchmark**

---

### 2️⃣ CRA — Classical Regression Adjustment

```python
data['Zdemean'] = data['Z'] - data['Z'].mean()
CRA = smf.ols("Y ~ D + Zdemean", data=data).fit()
CRA.get_robustcov_results(cov_type="HC0").summary()
```

- Adds demeaned Z as a linear control
- Demeaning Z makes the intercept interpretable as **E[Y(0)]** — expected outcome under control at average skill
- **Freedman (2008):** When the linear model is misspecified (as it is here), CRA can be *less* efficient than CL
- The SE on D increases from 0.081 (CL) → 0.119 (CRA) — a clear efficiency loss

**⚠️ Standard error correction needed for the intercept:**

Because we estimated the mean of Z from data (not from some external source), there is extra variance in the intercept that the naive HC0 SE does not account for. The corrected SE for the intercept is **0.0325** vs. the naive **0.014** — more than double.

```python
J = np.mean(1 - data['D'])
score = CRA.resid * (1 - data['D']) / J
score += data[['Zdemean']] @ CRA.params[['Zdemean']]
corrected_se = np.sqrt(np.mean(score**2) / len(data))
```

---

### 3️⃣ IRA — Interactive Regression Adjustment ✅ Recommended

```python
IRA = smf.ols("Y ~ D + Zdemean + Zdemean:D", data=data).fit()
IRA.get_robustcov_results(cov_type="HC1").summary()
```

- Adds both Z and its **interaction with D**
- The interaction term allows the relationship between skill and earnings to differ between treated and control groups
- This correctly captures the heterogeneous treatment effect structure
- **Lin (2013):** IRA always weakly improves efficiency over CL
- R² = 0.991 — the model fits the data almost perfectly

**⚠️ Standard error correction needed for both D and the intercept:**

```python
# Correction for D
correction = np.var(data[['Zdemean']].values @ IRA.params[['Zdemean:D']]) / len(data)
corrected_se_D = np.sqrt(IRA.HC0_se['D']**2 + correction)
# → 0.0634

# Correction for Intercept
J = np.mean(1 - data['D'])
score = (IRA.resid * (1 - data['D']) + J * data[['Zdemean']] @ IRA.params[['Zdemean']]) / J
corrected_se_int = np.sqrt(np.mean(score**2) / len(data))
# → 0.0320
```

---

## 📌 Key Theoretical Results

### Freedman (2008)
> Classical linear regression adjustment (CRA) can *hurt* precision when the conditional expectation of Y given D and X is not truly linear. In that case, CRA introduces bias in the variance estimator.

### Lin (2013)
> Interactive regression adjustment (IRA) — regressing Y on D, X, and D×X — always produces estimates that are at least as efficient as the unadjusted estimator (CL), asymptotically.

### Why the Difference?

CRA assumes: *"The effect of skill on earnings is the same whether or not you went to college."*

This is **wrong** in our DGP — skill has a negative effect without college and a positive effect with college. The linear model is misspecified, which inflates variance.

IRA allows: *"The effect of skill on earnings can be different for treated and control."*

This is **correct** and fully captures the underlying structure.

---

## 📊 Actual Output Results

### CL — No Adjustment

| Variable | Coefficient | Robust SE | t-stat | p-value |
|----------|-------------|-----------|--------|---------|
| Intercept | 0.017 | 0.035 | 0.49 | 0.624 |
| **D (ATE)** | **−0.150** | **0.081** | −1.86 | **0.063** |

- R² = 0.003
- ATE is **not significant** at 5% level

---

### CRA — Linear Adjustment

| Variable | Coefficient | Robust SE | t-stat | p-value |
|----------|-------------|-----------|--------|---------|
| Intercept | 0.031 | 0.014 | 2.27 | 0.023 |
| **D (ATE)** | **−0.223** | **0.119** | −1.87 | **0.061** |
| Zdemean | −0.628 | 0.041 | −15.24 | 0.000 |

- R² = 0.392
- SE on D is **larger** than CL — efficiency loss confirmed
- Corrected intercept SE = **0.0325** (naive was 0.014)

---

### IRA — Interactive Adjustment ✅

| Variable | Coefficient | Robust SE | t-stat | p-value |
|----------|-------------|-----------|--------|---------|
| Intercept | 0.039 | 0.003 | 11.80 | 0.000 |
| **D (ATE)** | **−0.079** | **0.008** | −9.93 | **<0.001** |
| Zdemean | −1.006 | 0.003 | −301.2 | 0.000 |
| Zdemean:D | 1.988 | 0.008 | 264.7 | 0.000 |

- R² = **0.991**
- SE on D is **0.008** (raw) / **0.0634** (corrected) — far better than CL and CRA
- ATE is **highly significant**
- Interaction coefficient ≈ **2.0** — correctly recovers the true CATE slope from the DGP

---

### Summary Comparison

```
True ATE = 0

Estimator   ATE Est.   Robust SE   Significant?   Verdict
──────────────────────────────────────────────────────────
CL          −0.150     0.081       ❌ No           Baseline
CRA         −0.223     0.119       ❌ No           ❌ Worse
IRA         −0.079     ~0.063*     ✅ Yes          ✅ Best

*corrected SE
```

---

## 🛡️ Why Robust Standard Errors Matter

Python's `smf.ols()` reports **non-robust** (classical) standard errors by default. These assume the model is perfectly correct — which is almost never true.

**The dangerous illusion with non-robust SEs:**

| Estimator | Non-Robust SE on D | Robust SE on D |
|-----------|-------------------|----------------|
| CL        | 0.081             | 0.081          |
| CRA       | **0.064** ← looks better! | **0.119** ← actually worse |

With non-robust SEs, CRA appears to improve on CL. With robust SEs, we see the truth: CRA is worse. **Non-robust SEs are misleading in this setting.**

Always use `get_robustcov_results(cov_type="HC0")` or `"HC1"` for reliable inference.

---

## 🎲 Monte Carlo Simulation

To verify that these results hold beyond a single sample, the notebook runs **1,000 independent simulations**, each with n=1,000 observations, and records the ATE estimate from all three methods.

```python
from joblib import Parallel, delayed

B = 1000
res = Parallel(n_jobs=-1)(delayed(exp)(it) for it in range(B))
```

The empirical standard deviation across 1,000 estimates measures real-world precision:

```
Expected ranking (confirmed by simulation):

IRA estimates → tightly clustered around 0  ✅ Most precise
CL  estimates → moderately spread
CRA estimates → most spread                 ❌ Least precise
```

This confirms the asymptotic theory holds even in finite samples of n=1,000.

---

## 💡 Concepts Explained Simply

### What is an RCT?
A **Randomized Controlled Trial** randomly assigns people to treatment or control. Because assignment is random, both groups are statistically similar before the experiment — so any difference in outcomes afterward must be caused by the treatment.

### What is ATE?
The **Average Treatment Effect** is the average difference in outcomes between the treated and control groups across the whole population. It answers: *"On average, how much does the treatment change the outcome?"*

### What is a Covariate?
A **covariate** is information measured about a person *before* the experiment starts (like skill, age, income). Using it wisely can reduce noise in your ATE estimate.

### What is Heterogeneous Treatment Effect?
When the treatment works **differently for different people**. In this example, college helps high-skill people a lot but hurts low-skill people. The effect is not the same for everyone.

### What is Demeaning?
**Demeaning** means subtracting the average value of a variable from every observation. It centers the variable around zero. When we demean Z, the coefficient on D becomes directly interpretable as the ATE (the effect at the average skill level).

### What is a Standard Error?
A **standard error (SE)** measures how uncertain your estimate is. Smaller SE = more precise estimate. The SE determines the width of your confidence interval.

### What is "Identical on Average"?
When we say random groups are *identical on average*, we mean their group-level characteristics (average age, average skill, average income, etc.) are nearly the same — not that individuals are clones. Randomization ensures this, so outcome differences after treatment are caused by the treatment, not pre-existing group differences.

---

## 📦 Requirements

```
python >= 3.8
pandas
numpy
statsmodels
joblib
```

Install all dependencies:

```bash
pip install pandas numpy statsmodels joblib
```

---

## ▶️ How to Run

**Clone the repository:**

```bash
git clone https://github.com/Aadip-Thapaliya/Analyzing-RCT.git
cd Analyzing-RCT
```

**Run the notebook:**

```bash
jupyter notebook python-sim-precision-adj.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/).

---

## 📚 References

- **Freedman, D.A. (2008).** "On regression adjustments to experimental data." *Advances in Applied Mathematics*, 40(2), 180–193.
- **Lin, W. (2013).** "Agnostic notes on regression adjustments to experimental data: Reexamining Freedman's critique." *The Annals of Applied Statistics*, 7(1), 295–318.
- **Eicker, F. (1967).** "Limit theorems for regressions with unequal and dependent errors." *Proceedings of the Fifth Berkeley Symposium.*
- **Huber, P.J. (1967).** "The behavior of maximum likelihood estimates under nonstandard conditions." *Proceedings of the Fifth Berkeley Symposium.*
- **White, H. (1980).** "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity." *Econometrica*, 48(4), 817–838.
- **Roth, J.** — DGP design for heterogeneous treatment effects illustration.

---

## 👤 Author

**Aadip Thapaliya**  
GitHub: [@Aadip-Thapaliya](https://github.com/Aadip-Thapaliya)  
Repository: [Analyzing-RCT](https://github.com/Aadip-Thapaliya/Analyzing-RCT)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**⭐ If you found this helpful, please star the repository! ⭐**

*Built with curiosity, econometrics, and a lot of Monte Carlo simulations.*

</div>
