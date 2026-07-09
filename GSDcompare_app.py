import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Photo vs. Field — Clast Size", layout="wide")

st.title("Comparison of clast size distributions (b-axis): photo vs. field")
st.caption(
    "Paste your two data series (one number per line, or separated "
    "by commas or spaces). They don't need to have the same number of measurements."
)

EXAMPLE_PHOTO = "57.3, 10.6, 18.3, 12.4, 16.5, 11.4, 4.6, 21.3, 6.3, 5.3"
EXAMPLE_FIELD = "4.7, 13.0, 8.1, 4.2, 6.1, 29.7, 9.9, 2.6, 7.9, 11.0"


def parse_numbers(text):
    """Converts a text block with numbers separated by commas, spaces, or line
    breaks into a float array. Ignores empty or non-numeric lines."""
    if not text:
        return np.array([])
    clean = text.replace(",", " ").replace(";", " ").replace("\n", " ")
    values = []
    for tok in clean.split():
        try:
            values.append(float(tok))
        except ValueError:
            continue
    return np.array(values)


col_input1, col_input2 = st.columns(2)
with col_input1:
    text_photo = st.text_area(
        "PHOTO data (mm)", value="", height=180, placeholder=EXAMPLE_PHOTO
    )
with col_input2:
    text_field = st.text_area(
        "FIELD data (mm)", value="", height=180, placeholder=EXAMPLE_FIELD
    )

use_example = st.checkbox("Use example data (to try out the app)")

if use_example:
    photo = parse_numbers(
        "57.3,10.6,18.3,12.4,16.5,11.4,4.6,21.3,6.3,5.3,"
        "29.3,2.8,4.9,7.5,1.9,29.2,9.7,38.2,18.7,2.5,"
        "7.4,15.9,14.6,2.5,0.6,21.3,14.9,21.2,9.4,3.6,"
        "5.1,4.0,21.2,9.6,2.5,11.5"
    )
    field = parse_numbers(
        "4.7,13.0,8.1,4.2,6.1,29.7,9.9,2.6,7.9,11.0,"
        "8.3,19.2,19.2,17.3,20.6,26.7,24.4,16.4,14.5,6.9,"
        "8.6,11.1,9.5,20.7,15.2,12.2,9.8,31.3,24.5,10.5,"
        "18.5,22.9,12.8,12.2,63.0,24.9"
    )
else:
    photo = parse_numbers(text_photo)
    field = parse_numbers(text_field)

if len(photo) < 4 or len(field) < 4:
    st.info(
        "Enter at least 4 values in each series (or check the "
        "example data box) to see the results."
    )
    st.stop()

photo_sorted = np.sort(photo)
field_sorted = np.sort(field)

# -------------------------------------------------------------
# Descriptive statistics
# -------------------------------------------------------------
st.header("1. Descriptive statistics")
c1, c2 = st.columns(2)
for col, name, arr in [(c1, "Photo", photo), (c2, "Field", field)]:
    with col:
        st.subheader(name)
        st.write(f"n = {len(arr)}")
        st.write(f"Mean = {arr.mean():.2f} mm")
        st.write(f"Median = {np.median(arr):.2f} mm")
        st.write(f"Standard deviation = {arr.std(ddof=1):.2f} mm")

# -------------------------------------------------------------
# Percentiles and sorting
# -------------------------------------------------------------
st.header("2. Percentiles and sorting")
c1, c2 = st.columns(2)
for col, name, arr in [(c1, "Photo", photo), (c2, "Field", field)]:
    with col:
        d16, d50, d84 = np.percentile(arr, [16, 50, 84])
        sorting = np.sqrt(d84 / d16)
        st.subheader(name)
        st.write(f"D16 = {d16:.2f} mm")
        st.write(f"D50 = {d50:.2f} mm")
        st.write(f"D84 = {d84:.2f} mm")
        st.write(f"Sorting = {sorting:.2f}")

# -------------------------------------------------------------
# Statistical tests
# -------------------------------------------------------------
st.header("3. Statistical tests (independent distributions)")

ks_stat, ks_p = stats.ks_2samp(photo, field)
t_log, p_log = stats.ttest_ind(np.log(photo), np.log(field), equal_var=False)

st.table(
    {
        "Test": [
            "Kolmogorov-Smirnov",
            "Welch t-test (log)",
        ],
        "Statistic": [
            f"D={ks_stat:.3f}",
            f"t={t_log:.3f}",
        ],
        "p-value": [
            f"{ks_p:.4f}",
            f"{p_log:.4f}",
        ],
    }
)
st.caption(
    "Grain size usually follows a log-normal distribution, so the "
    "Welch t-test on log values tends to be the most reliable."
)

# -------------------------------------------------------------
# Additive model (regression on paired quantiles)
# -------------------------------------------------------------
st.header("4. Additive model: field = a·photo + b")

n_min = min(len(photo_sorted), len(field_sorted))
if len(photo_sorted) != len(field_sorted):
    q = np.linspace(0, 100, n_min)
    photo_q = np.percentile(photo_sorted, q)
    field_q = np.percentile(field_sorted, q)
else:
    photo_q, field_q = photo_sorted, field_sorted

slope, intercept, r_value, p_value, std_err = stats.linregress(photo_q, field_q)
resid_add = field_q - (slope * photo_q + intercept)
sd_add = resid_add.std(ddof=2) if n_min > 2 else float("nan")

st.write(f"**field = {slope:.3f} · photo + {intercept:.3f}**  (R² = {r_value**2:.3f})")
st.write(f"Residual SD (noise after removing the bias): {sd_add:.2f} mm")

# -------------------------------------------------------------
# Multiplicative model (foreshortening)
# -------------------------------------------------------------
st.header("5. Multiplicative model: photo = k·field")

k = np.sum(photo_q * field_q) / np.sum(field_q ** 2)
pred_mult = k * field_q
resid_mult = photo_q - pred_mult
sd_mult = resid_mult.std(ddof=1) if n_min > 1 else float("nan")
ss_res = np.sum(resid_mult ** 2)
ss_tot = np.sum((photo_q - photo_q.mean()) ** 2)
r2_mult = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
theta_deg = np.degrees(np.arccos(min(k, 1.0)))

st.write(f"**photo = {k:.4f} · field**  (R² = {r2_mult:.3f})")
st.write(f"Implied angle (if k = cos(θ)): {theta_deg:.1f}°")
st.write(f"Residual SD: {sd_mult:.2f} mm")

# -------------------------------------------------------------
# +/- 2 sigma band
# -------------------------------------------------------------
st.header("6. ±2σ band (on the multiplicative model)")

band = 2 * sd_mult
within = np.sum(np.abs(resid_mult) <= band)
st.write(f"Band: ± {band:.2f} mm")
st.write(f"Points within band: {within}/{n_min} ({within/n_min*100:.1f}%)")

outside_idx = np.where(np.abs(resid_mult) > band)[0]
if len(outside_idx) > 0:
    st.write("Points outside the band (photo, field):")
    for i in outside_idx:
        st.write(f"  ({photo_q[i]:.1f}, {field_q[i]:.1f})")

# -------------------------------------------------------------
# Bootstrap D50
# -------------------------------------------------------------
st.header("7. Bootstrap: sampling uncertainty of D50")

rng = np.random.default_rng(42)
n_boot = 5000
c1, c2 = st.columns(2)
for col, name, arr in [(c1, "Photo", photo), (c2, "Field", field)]:
    with col:
        boot = [
            np.percentile(rng.choice(arr, size=len(arr), replace=True), 50)
            for _ in range(n_boot)
        ]
        ci = np.percentile(boot, [2.5, 97.5])
        st.write(f"**{name}**: D50 = {np.median(arr):.2f} mm")
        st.write(f"95% bootstrap CI = [{ci[0]:.2f}, {ci[1]:.2f}] mm")

# -------------------------------------------------------------
# Plots
# -------------------------------------------------------------
st.header("8. Plots")

max_val = max(photo.max(), field.max()) * 1.1

# --- Plot 1: Q-Q plot ---
diff_11 = field_q - photo_q
sd_11 = diff_11.std(ddof=1) if n_min > 1 else float("nan")
band_11 = 2 * sd_11
within_11 = np.abs(diff_11) <= band_11

x_11 = np.array([0, max_val])
y_reg = slope * x_11 + intercept

fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
ax1.fill_between(
    x_11, x_11 - band_11, x_11 + band_11, color="gray", alpha=0.15,
    label=f"1:1 band ±2σ ({band_11:.2f} mm)",
)
ax1.plot(x_11, x_11, "--", color="gray", label="1:1 line")
ax1.plot(
    x_11, y_reg, "-", color="#52514e", linewidth=1.8,
    label=f"Regression: field={slope:.3f}·photo+{intercept:.3f}  (R²={r_value**2:.3f})",
)
ax1.scatter(
    photo_q[within_11], field_q[within_11], color="#2a78d6", zorder=3,
    label=f"Within 1:1 band ({within_11.sum()})",
)
ax1.scatter(
    photo_q[~within_11], field_q[~within_11], color="#e34948", zorder=3,
    label=f"Outside 1:1 band ({(~within_11).sum()})",
)
eq_text = (
    f"field = {slope:.3f}·photo + {intercept:.3f}\n"
    f"R² = {r_value**2:.3f}\n"
    f"Residual SD = {sd_add:.2f} mm"
)
ax1.text(
    0.03, 0.97, eq_text, transform=ax1.transAxes, fontsize=9, va="top", ha="left",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
)
ax1.set_xlabel("Photo (mm)")
ax1.set_ylabel("Field (mm)")
ax1.set_title("Q-Q plot: photo vs. field quantiles")
ax1.set_xlim(0, max_val)
ax1.set_ylim(0, max_val)
ax1.legend(fontsize=7, loc="lower right")
ax1.set_aspect("equal")
fig1.tight_layout()

# --- Plot 2: Cumulative frequency curves ---
fig2, ax2 = plt.subplots(figsize=(7, 5))
pct_photo = np.arange(1, len(photo_sorted) + 1) / len(photo_sorted) * 100
pct_field = np.arange(1, len(field_sorted) + 1) / len(field_sorted) * 100
ax2.step(photo_sorted, pct_photo, where="post", color="#2a78d6", label="Photo")
ax2.step(field_sorted, pct_field, where="post", color="#e34948", label="Field")
ax2.set_xlabel("B-axis (mm)")
ax2.set_ylabel("Cumulative %")
ax2.set_title("Cumulative frequency curves")
ax2.legend()
fig2.tight_layout()

# --- Plot 3: Multiplicative model with band ---
fig3, ax3 = plt.subplots(figsize=(7, 6))
x_line = np.array([0, max_val])
y_line = k * x_line
ax3.plot(x_line, y_line, "--", color="#52514e", label=f"Model photo = {k:.3f}·field")
ax3.fill_between(
    x_line, y_line - band, y_line + band, color="gray", alpha=0.15,
    label=f"±2σ band ({band:.2f} mm)",
)
within_mask = np.abs(resid_mult) <= band
ax3.scatter(
    field_q[within_mask], photo_q[within_mask], color="#2a78d6", zorder=3,
    label=f"Within band ({within_mask.sum()})",
)
ax3.scatter(
    field_q[~within_mask], photo_q[~within_mask], color="#e34948", zorder=3,
    label=f"Outside band ({(~within_mask).sum()})",
)
eq_text3 = (
    f"photo = {k:.4f}·field\n"
    f"R² = {r2_mult:.3f}\n"
    f"θ = arccos(k) = {theta_deg:.1f}°\n"
    f"Residual SD = {sd_mult:.2f} mm"
)
ax3.text(
    0.03, 0.97, eq_text3, transform=ax3.transAxes, fontsize=9, va="top", ha="left",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
)
ax3.set_xlabel("Field (mm)")
ax3.set_ylabel("Photo (mm)")
ax3.set_title(f"Foreshortening model (θ≈{theta_deg:.1f}°) with 2σ band")
ax3.set_xlim(0, max_val)
ax3.legend(fontsize=8, loc="lower right")
fig3.tight_layout()

# --- Plot 4: KDE with D50/D84 ---
fig4, ax4 = plt.subplots(figsize=(8, 5))
x_grid = np.linspace(0, max_val, 500)
kde_photo = gaussian_kde(photo)
kde_field = gaussian_kde(field)
ax4.plot(x_grid, kde_photo(x_grid), color="#2a78d6", label="Photo")
ax4.fill_between(x_grid, kde_photo(x_grid), color="#2a78d6", alpha=0.15)
ax4.plot(x_grid, kde_field(x_grid), color="#e34948", label="Field")
ax4.fill_between(x_grid, kde_field(x_grid), color="#e34948", alpha=0.15)

d50_photo, d84_photo = np.percentile(photo, [50, 84])
d50_field, d84_field = np.percentile(field, [50, 84])
ax4.axvline(d50_photo, color="#2a78d6", linestyle="--", linewidth=1.5)
ax4.axvline(d84_photo, color="#2a78d6", linestyle=":", linewidth=1.5)
ax4.axvline(d50_field, color="#e34948", linestyle="--", linewidth=1.5)
ax4.axvline(d84_field, color="#e34948", linestyle=":", linewidth=1.5)

ymax = max(kde_photo(x_grid).max(), kde_field(x_grid).max())
ax4.text(d50_photo, ymax * 1.02, f"D50={d50_photo:.1f}", color="#2a78d6", fontsize=8, ha="center")
ax4.text(d84_photo, ymax * 1.08, f"D84={d84_photo:.1f}", color="#2a78d6", fontsize=8, ha="center")
ax4.text(d50_field, ymax * 1.02, f"D50={d50_field:.1f}", color="#e34948", fontsize=8, ha="center")
ax4.text(d84_field, ymax * 1.08, f"D84={d84_field:.1f}", color="#e34948", fontsize=8, ha="center")

ax4.set_xlabel("B-axis (mm)")
ax4.set_ylabel("Density")
ax4.set_title("Density function (KDE) with D50 (--) and D84 (:) marked")
ax4.set_ylim(0, ymax * 1.18)
ax4.legend()
fig4.tight_layout()

g1, g2 = st.columns(2)
with g1:
    st.pyplot(fig1)
with g2:
    st.pyplot(fig2)

g3, g4 = st.columns(2)
with g3:
    st.pyplot(fig3)
with g4:
    st.pyplot(fig4)
