# Concentration uncertainty of a time-mean distribution

This optional calculation follows the native-bin-total combination in log space.
It does **not** change the merged concentrations, instrument weights, fitted
refractive indices or density. It does not require bootstrap samples or new
alignment fits. The estimate covers temporal sampling variability of the mean,
not a complete instrument error budget. In particular it excludes uncertainty in
alignment, calibration, optical geometry and instrument size response.

## Temporal covariance

Arrange the native-bin **number concentrations** from all instruments in columns
of one common-clock time array. Retain observed zero seconds. Mark missing times
with NaN, not zero. Use exactly the filtering and time samples used for the mean
spectra. Converted bins conserve number, so the time data can remain on the
original native bins while their mean numbers are paired with converted edges.

For column i, let n_i be its valid sample count and Nbar_i its arithmetic mean.
Within time block g, sum (N_ti - Nbar_i)/n_i over observed times. Calling this sum
S_gi, the estimated covariance of the means is

    C_N = B/(B-1) * S.T @ S

where B is the number of blocks. Off-diagonal entries retain fluctuations shared
by different bins and instruments. With complete data this is exactly the sample
covariance of the block means divided by B. Missing times contribute zero to the
centered sum; they do not contribute a zero concentration to the mean.

The prepared one-minute ARCSIX notebook uses twelve five-second blocks. It assumes
these blocks are independent; unknown instrument averaging and longer-lasting
correlation remain limitations. Requiring at least 20 valid seconds in at least
four occupied blocks for every retained positive bin is an availability safeguard,
not a validation of that assumption or a new concentration QC exclusion. Minutes
that fail this requirement keep their concentrations and get missing uncertainty.
A zero empirical variance means no observed variability, not proven zero error.

## Propagation through the combination

Let b be log10 of the positive native-bin mean numbers and z be the fitted log10
output heights. The existing objective F(z,b) combines weighted native-bin errors
and the Tikhonov smoothness penalty. At the fitted solution, its derivative with
respect to z is zero. Differentiating that condition gives

    H K = -B_mixed

Here H is the second derivative of F with respect to z twice, B_mixed is its
mixed derivative with respect to z and b, and K = dz/db is the sensitivity matrix.
The implementation includes residual curvature, not only the approximation from
the residual Jacobian. It checks that H is positive definite before solving.

The concentration covariance is converted to log coordinates using
G = diag(1/(ln(10)*Nbar)). Then

    C_z = K @ G @ C_N @ G.T @ K.T

This is a first-order, local propagation with fixed diameter conversion, bin
selection, weights and smoothing. The APS retained endpoint target remains zero;
its scale is proportional to its neighboring positive bin, and that dependence
is propagated explicitly. No uncertainty is invented for omitted zero-mean bins.

When optional native-bin consensus is enabled, the derivative uses the actual
effective weights (base weights multiplied by agreement factors) saved by that
fit. The agreement factors themselves are held fixed. These bands therefore do
not include sensitivity to changes in which instruments agree or disagree.
They must not be described as uncertainty of the entire adaptive weighting
procedure. Changing consensus requires recalculating these concentration bands;
bands from a previous no-consensus fit must not be copied to the new spectrum.

The code stores a compact block-by-output factor Q with C_z = Q.T @ Q when full
cross-bin covariance is needed later (for example, integrated number uncertainty).
It does not assume output bins are independent. Per-bin standard uncertainty is
sqrt(diag(C_z)). A non-positive fit curvature gives unavailable uncertainty rather
than an automatically substituted covariance approximation.

## Output fields and plotting

The original ICARTT columns, including `DNLOG_001` through `DNLOG_100`, retain their
names and positions. Append two sets of columns **after** all existing columns:

| Fields | Meaning | Units |
| --- | --- | --- |
| `SD_LOG10_001` ... `SD_LOG10_100` | One standard uncertainty in log10 concentration | dimensionless |
| `SD_DNLOG_001` ... `SD_DNLOG_100` | First-order linear standard uncertainty: ln(10) × DNLOG × SD_LOG10 | cm⁻³ |

These are **errors, not lower/upper limits**. They describe the same uncertainty
in two coordinate systems and must not be added together. They share the existing
diameter-bin edges. Missing uncertainty is encoded with the ICARTT missing value.
NetCDF stores the same two standard errors, availability status and method
metadata; per-period diagnostic files explain unavailable estimates. Bounds
are calculated when plotting, rather than stored separately for each choice of k.

For the logarithmic distribution plot:

```python
k = 1  # Change to 2 for two standard deviations.
lower = concentration * 10**(-k * sd_log10)
upper = concentration * 10**( k * sd_log10)
```

These are asymmetric bands on a linear axis and are not validated confidence
intervals. For small relative uncertainty, symmetric error bars k × SD_DNLOG
approximate these bands; for large uncertainty use the log-space bands above.

ICARTT readers that use the declared header length and column names can keep
reading the original columns. A reader that hardcodes the total column count,
header length, or assumes every column after DNLOG_001 is a concentration must be
updated. For example, with a standard comma-separated ICARTT header:

```python
import pandas as pd

with open(path) as f:
    header_lines = int(f.readline().split(',')[0])
df = pd.read_csv(path, skiprows=header_lines-1, skipinitialspace=True,
                 na_values=[-9999.0])
concentration = df.filter(regex=r'^DNLOG_\d{3}$').to_numpy()
sd_log10 = df.filter(regex=r'^SD_LOG10_\d{3}$').to_numpy()
```
