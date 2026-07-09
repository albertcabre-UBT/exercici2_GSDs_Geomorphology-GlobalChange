import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Foto vs. Camp — Clastometria", layout="wide")

st.title("Comparació de distribucions de mida de clast (eix-b): foto vs. camp")
st.caption(
    "Enganxa les teves dues sèries de dades (una xifra per línia, o separades "
    "per comes o espais). No cal que tinguin el mateix nombre de mesures."
)

EXEMPLE_FOTO = "57.3, 10.6, 18.3, 12.4, 16.5, 11.4, 4.6, 21.3, 6.3, 5.3"
EXEMPLE_CAMPO = "4.7, 13.0, 8.1, 4.2, 6.1, 29.7, 9.9, 2.6, 7.9, 11.0"


def parse_numeros(text):
    """Converteix un bloc de text amb números separats per comes, espais o salts
    de línia en un array de floats. Ignora línies buides o no numèriques."""
    if not text:
        return np.array([])
    net = text.replace(",", " ").replace(";", " ").replace("\n", " ")
    valors = []
    for tok in net.split():
        try:
            valors.append(float(tok))
        except ValueError:
            continue
    return np.array(valors)


col_input1, col_input2 = st.columns(2)
with col_input1:
    text_foto = st.text_area(
        "Dades FOTO (mm)", value="", height=180, placeholder=EXEMPLE_FOTO
    )
with col_input2:
    text_campo = st.text_area(
        "Dades CAMP (mm)", value="", height=180, placeholder=EXEMPLE_CAMPO
    )

usar_exemple = st.checkbox("Utilitza dades d'exemple (per provar l'app)")

if usar_exemple:
    foto = parse_numeros(
        "57.3,10.6,18.3,12.4,16.5,11.4,4.6,21.3,6.3,5.3,"
        "29.3,2.8,4.9,7.5,1.9,29.2,9.7,38.2,18.7,2.5,"
        "7.4,15.9,14.6,2.5,0.6,21.3,14.9,21.2,9.4,3.6,"
        "5.1,4.0,21.2,9.6,2.5,11.5"
    )
    campo = parse_numeros(
        "4.7,13.0,8.1,4.2,6.1,29.7,9.9,2.6,7.9,11.0,"
        "8.3,19.2,19.2,17.3,20.6,26.7,24.4,16.4,14.5,6.9,"
        "8.6,11.1,9.5,20.7,15.2,12.2,9.8,31.3,24.5,10.5,"
        "18.5,22.9,12.8,12.2,63.0,24.9"
    )
else:
    foto = parse_numeros(text_foto)
    campo = parse_numeros(text_campo)

if len(foto) < 4 or len(campo) < 4:
    st.info(
        "Introdueix com a mínim 4 valors a cada sèrie (o marca la casella "
        "de dades d'exemple) per veure els resultats."
    )
    st.stop()

foto_sorted = np.sort(foto)
campo_sorted = np.sort(campo)

# -------------------------------------------------------------
# Descriptius
# -------------------------------------------------------------
st.header("1. Descriptius")
c1, c2 = st.columns(2)
for col, nom, arr in [(c1, "Foto", foto), (c2, "Camp", campo)]:
    with col:
        st.subheader(nom)
        st.write(f"n = {len(arr)}")
        st.write(f"Mitjana = {arr.mean():.2f} mm")
        st.write(f"Mediana = {np.median(arr):.2f} mm")
        st.write(f"Desviació estàndard = {arr.std(ddof=1):.2f} mm")

# -------------------------------------------------------------
# Percentils i sorting
# -------------------------------------------------------------
st.header("2. Percentils i sorting")
c1, c2 = st.columns(2)
for col, nom, arr in [(c1, "Foto", foto), (c2, "Camp", campo)]:
    with col:
        d16, d50, d84 = np.percentile(arr, [16, 50, 84])
        sorting = np.sqrt(d84 / d16)
        st.subheader(nom)
        st.write(f"D16 = {d16:.2f} mm")
        st.write(f"D50 = {d50:.2f} mm")
        st.write(f"D84 = {d84:.2f} mm")
        st.write(f"Sorting = {sorting:.2f}")

# -------------------------------------------------------------
# Tests estadístics
# -------------------------------------------------------------
st.header("3. Tests estadístics (distribucions independents)")

ks_stat, ks_p = stats.ks_2samp(foto, campo)
u_stat, u_p = stats.mannwhitneyu(foto, campo, alternative="two-sided")
t_stat, t_p = stats.ttest_ind(foto, campo, equal_var=False)
t_log, p_log = stats.ttest_ind(np.log(foto), np.log(campo), equal_var=False)

st.table(
    {
        "Test": [
            "Kolmogorov-Smirnov",
            "Mann-Whitney U",
            "Welch t-test (bruts)",
            "Welch t-test (log)",
        ],
        "Estadístic": [
            f"D={ks_stat:.3f}",
            f"U={u_stat:.1f}",
            f"t={t_stat:.3f}",
            f"t={t_log:.3f}",
        ],
        "p-valor": [
            f"{ks_p:.4f}",
            f"{u_p:.4f}",
            f"{t_p:.4f}",
            f"{p_log:.4f}",
        ],
    }
)
st.caption(
    "La mida de gra sol seguir una distribució log-normal, per la qual cosa "
    "el Welch t-test sobre els valors log sol ser el més fiable."
)

# -------------------------------------------------------------
# Model additiu (regressió sobre quantils aparellats)
# -------------------------------------------------------------
st.header("4. Model additiu: camp = a·foto + b")

n_min = min(len(foto_sorted), len(campo_sorted))
if len(foto_sorted) != len(campo_sorted):
    q = np.linspace(0, 100, n_min)
    foto_q = np.percentile(foto_sorted, q)
    campo_q = np.percentile(campo_sorted, q)
else:
    foto_q, campo_q = foto_sorted, campo_sorted

slope, intercept, r_value, p_value, std_err = stats.linregress(foto_q, campo_q)
resid_add = campo_q - (slope * foto_q + intercept)
sd_add = resid_add.std(ddof=2) if n_min > 2 else float("nan")

st.write(f"**camp = {slope:.3f} · foto + {intercept:.3f}**  (R² = {r_value**2:.3f})")
st.write(f"SD residual (soroll un cop tret el biaix): {sd_add:.2f} mm")

# -------------------------------------------------------------
# Model multiplicatiu (foreshortening)
# -------------------------------------------------------------
st.header("5. Model multiplicatiu: foto = k·camp")

k = np.sum(foto_q * campo_q) / np.sum(campo_q ** 2)
pred_mult = k * campo_q
resid_mult = foto_q - pred_mult
sd_mult = resid_mult.std(ddof=1) if n_min > 1 else float("nan")
ss_res = np.sum(resid_mult ** 2)
ss_tot = np.sum((foto_q - foto_q.mean()) ** 2)
r2_mult = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
theta_deg = np.degrees(np.arccos(min(k, 1.0)))

st.write(f"**foto = {k:.4f} · camp**  (R² = {r2_mult:.3f})")
st.write(f"Angle implícit (si k = cos(θ)): {theta_deg:.1f}°")
st.write(f"SD residual: {sd_mult:.2f} mm")

# -------------------------------------------------------------
# Banda +/- 2 sigma
# -------------------------------------------------------------
st.header("6. Banda ±2σ (sobre el model multiplicatiu)")

band = 2 * sd_mult
dins = np.sum(np.abs(resid_mult) <= band)
st.write(f"Banda: ± {band:.2f} mm")
st.write(f"Punts dins banda: {dins}/{n_min} ({dins/n_min*100:.1f}%)")

fora_idx = np.where(np.abs(resid_mult) > band)[0]
if len(fora_idx) > 0:
    st.write("Punts fora de banda (foto, camp):")
    for i in fora_idx:
        st.write(f"  ({foto_q[i]:.1f}, {campo_q[i]:.1f})")

# -------------------------------------------------------------
# Bootstrap D50
# -------------------------------------------------------------
st.header("7. Bootstrap: incertesa de mostreig del D50")

rng = np.random.default_rng(42)
n_boot = 5000
c1, c2 = st.columns(2)
for col, nom, arr in [(c1, "Foto", foto), (c2, "Camp", campo)]:
    with col:
        boot = [
            np.percentile(rng.choice(arr, size=len(arr), replace=True), 50)
            for _ in range(n_boot)
        ]
        ci = np.percentile(boot, [2.5, 97.5])
        st.write(f"**{nom}**: D50 = {np.median(arr):.2f} mm")
        st.write(f"IC95% bootstrap = [{ci[0]:.2f}, {ci[1]:.2f}] mm")

# -------------------------------------------------------------
# Gràfics
# -------------------------------------------------------------
st.header("8. Gràfics")

max_val = max(foto.max(), campo.max()) * 1.1

# --- Gràfic 1: Q-Q plot ---
diff_11 = campo_q - foto_q
sd_11 = diff_11.std(ddof=1) if n_min > 1 else float("nan")
band_11 = 2 * sd_11
dins_11 = np.abs(diff_11) <= band_11

x_11 = np.array([0, max_val])
y_reg = slope * x_11 + intercept

fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
ax1.fill_between(
    x_11, x_11 - band_11, x_11 + band_11, color="gray", alpha=0.15,
    label=f"Banda 1:1 ±2σ ({band_11:.2f} mm)",
)
ax1.plot(x_11, x_11, "--", color="gray", label="Línia 1:1")
ax1.plot(
    x_11, y_reg, "-", color="#52514e", linewidth=1.8,
    label=f"Regressió: camp={slope:.3f}·foto+{intercept:.3f}  (R²={r_value**2:.3f})",
)
ax1.scatter(
    foto_q[dins_11], campo_q[dins_11], color="#2a78d6", zorder=3,
    label=f"Dins banda 1:1 ({dins_11.sum()})",
)
ax1.scatter(
    foto_q[~dins_11], campo_q[~dins_11], color="#e34948", zorder=3,
    label=f"Fora de banda 1:1 ({(~dins_11).sum()})",
)
eq_text = (
    f"camp = {slope:.3f}·foto + {intercept:.3f}\n"
    f"R² = {r_value**2:.3f}\n"
    f"SD residual = {sd_add:.2f} mm"
)
ax1.text(
    0.03, 0.97, eq_text, transform=ax1.transAxes, fontsize=9, va="top", ha="left",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
)
ax1.set_xlabel("Photo (mm)")
ax1.set_ylabel("Field (mm)")
ax1.set_title("Q-Q plot: quantils foto vs. camp")
ax1.set_xlim(0, max_val)
ax1.set_ylim(0, max_val)
ax1.legend(fontsize=7, loc="lower right")
ax1.set_aspect("equal")
fig1.tight_layout()

# --- Gràfic 2: Corbes de freqüència acumulada ---
fig2, ax2 = plt.subplots(figsize=(7, 5))
pct_foto = np.arange(1, len(foto_sorted) + 1) / len(foto_sorted) * 100
pct_campo = np.arange(1, len(campo_sorted) + 1) / len(campo_sorted) * 100
ax2.step(foto_sorted, pct_foto, where="post", color="#2a78d6", label="Foto")
ax2.step(campo_sorted, pct_campo, where="post", color="#e34948", label="Camp")
ax2.set_xlabel("B-axis (mm)")
ax2.set_ylabel("% acumulat")
ax2.set_title("Corbes de freqüència acumulada")
ax2.legend()
fig2.tight_layout()

# --- Gràfic 3: Model multiplicatiu amb banda ---
fig3, ax3 = plt.subplots(figsize=(7, 6))
x_line = np.array([0, max_val])
y_line = k * x_line
ax3.plot(x_line, y_line, "--", color="#52514e", label=f"Model foto = {k:.3f}·camp")
ax3.fill_between(
    x_line, y_line - band, y_line + band, color="gray", alpha=0.15,
    label=f"Banda ±2σ ({band:.2f} mm)",
)
dins_mask = np.abs(resid_mult) <= band
ax3.scatter(
    campo_q[dins_mask], foto_q[dins_mask], color="#2a78d6", zorder=3,
    label=f"Dins banda ({dins_mask.sum()})",
)
ax3.scatter(
    campo_q[~dins_mask], foto_q[~dins_mask], color="#e34948", zorder=3,
    label=f"Fora de banda ({(~dins_mask).sum()})",
)
eq_text3 = (
    f"foto = {k:.4f}·camp\n"
    f"R² = {r2_mult:.3f}\n"
    f"θ = arccos(k) = {theta_deg:.1f}°\n"
    f"SD residual = {sd_mult:.2f} mm"
)
ax3.text(
    0.03, 0.97, eq_text3, transform=ax3.transAxes, fontsize=9, va="top", ha="left",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
)
ax3.set_xlabel("Camp (mm)")
ax3.set_ylabel("Foto (mm)")
ax3.set_title(f"Model de foreshortening (θ≈{theta_deg:.1f}°) amb banda 2σ")
ax3.set_xlim(0, max_val)
ax3.legend(fontsize=8, loc="lower right")
fig3.tight_layout()

# --- Gràfic 4: KDE amb D50/D84 ---
fig4, ax4 = plt.subplots(figsize=(8, 5))
x_grid = np.linspace(0, max_val, 500)
kde_foto = gaussian_kde(foto)
kde_campo = gaussian_kde(campo)
ax4.plot(x_grid, kde_foto(x_grid), color="#2a78d6", label="Foto")
ax4.fill_between(x_grid, kde_foto(x_grid), color="#2a78d6", alpha=0.15)
ax4.plot(x_grid, kde_campo(x_grid), color="#e34948", label="Camp")
ax4.fill_between(x_grid, kde_campo(x_grid), color="#e34948", alpha=0.15)

d50_foto, d84_foto = np.percentile(foto, [50, 84])
d50_campo, d84_campo = np.percentile(campo, [50, 84])
ax4.axvline(d50_foto, color="#2a78d6", linestyle="--", linewidth=1.5)
ax4.axvline(d84_foto, color="#2a78d6", linestyle=":", linewidth=1.5)
ax4.axvline(d50_campo, color="#e34948", linestyle="--", linewidth=1.5)
ax4.axvline(d84_campo, color="#e34948", linestyle=":", linewidth=1.5)

ymax = max(kde_foto(x_grid).max(), kde_campo(x_grid).max())
ax4.text(d50_foto, ymax * 1.02, f"D50={d50_foto:.1f}", color="#2a78d6", fontsize=8, ha="center")
ax4.text(d84_foto, ymax * 1.08, f"D84={d84_foto:.1f}", color="#2a78d6", fontsize=8, ha="center")
ax4.text(d50_campo, ymax * 1.02, f"D50={d50_campo:.1f}", color="#e34948", fontsize=8, ha="center")
ax4.text(d84_campo, ymax * 1.08, f"D84={d84_campo:.1f}", color="#e34948", fontsize=8, ha="center")

ax4.set_xlabel("B-axis (mm)")
ax4.set_ylabel("Densitat")
ax4.set_title("Funció de densitat (KDE) amb D50 (--) i D84 (:) marcats")
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
