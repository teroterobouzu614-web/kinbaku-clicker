import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import wofz
from tkinter import Tk, filedialog
import os


# -------------------------------
# Voigt
# -------------------------------
def voigt(x, x0, sigma, gamma):
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


# -------------------------------
# fit: raw intensity = linear baseline + Voigt
# -------------------------------
def fit_spectrum(wave, intensity):
    wave = np.asarray(wave, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    # 波長順に並べ替え
    sort_idx = np.argsort(wave)
    wave = wave[sort_idx]
    intensity = intensity[sort_idx]

    peak_i = np.argmax(intensity)
    peak_wave = wave[peak_i]

    # ピーク中心の±3 nmでフィット
    mask = (wave > peak_wave - 3) & (wave < peak_wave + 3)
    x = wave[mask]
    y = intensity[mask]

    if len(x) < 8:
        raise ValueError("Not enough points in fitting window.")

    xref = x.mean()

    def model(p, x_in):
        A, x0, s, g, b0, b1 = p
        return b0 + b1 * (x_in - xref) + A * voigt(x_in, x0, s, g)

    def res(p):
        A, x0, s, g, _, _ = p
        if s <= 0 or g <= 0 or A < 0:
            return np.ones_like(y) * 1e6
        # Poisson-like weighting
        weights = 1.0 / np.sqrt(np.clip(y, 1.0, None))
        return (y - model(p, x)) * weights

    # A is area coefficient for normalized Voigt (not peak height)
    peak_h = max(y.max() - np.median(y), 1e-6)
    A0 = peak_h * 0.5 * np.sqrt(2 * np.pi)

    p0 = [A0, peak_wave, 0.4, 0.4, np.median(y), 0.0]
    lb = [0.0, peak_wave - 1.5, 0.02, 0.02, -np.inf, -np.inf]
    ub = [np.inf, peak_wave + 1.5, 3.0, 3.0, np.inf, np.inf]

    result = least_squares(res, p0, bounds=(lb, ub))
    A, x0, s, g, b0, b1 = result.x

    # local window outputs
    fit_local = model(result.x, x)
    baseline_local = b0 + b1 * (x - xref)

    # full-range outputs for plotting
    baseline_all = b0 + b1 * (wave - xref)
    fit_all = model(result.x, wave)

    data_integral = np.trapezoid(np.clip(y - baseline_local, 0, None), x)
    fit_integral = np.trapezoid(np.clip(fit_local - baseline_local, 0, None), x)

    return wave, intensity, baseline_all, fit_all, A, x0, s, g, data_integral, fit_integral


# -------------------------------
# MAIN
# -------------------------------
def main():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select spectrum file",
        filetypes=[("Text files", "*.txt")]
    )

    if file_path == "":
        print("No file selected")
        return

    df = pd.read_csv(file_path, sep=r"\s+")

    required_cols = ["#X", "#Y", "#Wave", "#Intensity"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in file.")

    groups = df.groupby(["#X", "#Y"])

    # 出力先は元データファイルと同じフォルダ
    out_dir = os.path.dirname(os.path.abspath(file_path))

    results = []

    for i, ((gx, gy), group_df) in enumerate(groups):
        wave = group_df["#Wave"].values
        inten = group_df["#Intensity"].values

        try:
            wave_sorted, inten_sorted, baseline_all, fit_all, A, x0, s, gamma, data_int, fit_int = fit_spectrum(wave, inten)

            plt.figure(figsize=(6, 5))
            plt.scatter(wave_sorted, inten_sorted, s=5, label="data")
            plt.plot(wave_sorted, baseline_all, label="baseline")
            plt.plot(wave_sorted, fit_all, label="Voigt fit")

            plt.title(f"X={gx} Y={gy}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")
            plt.legend()

            name = os.path.join(out_dir, f"fit_{i:02d}.png")
            plt.savefig(name, dpi=150, bbox_inches="tight")
            plt.close()

            results.append({
                "index": i,
                "X": gx,
                "Y": gy,
                "center_nm": x0,
                "sigma_nm": s,
                "gamma_nm": gamma,
                "area_coeff_A": A,
                "data_integral": data_int,
                "fit_integral": fit_int,
            })

        except Exception as e:
            print(f"Fit failed for X={gx}, Y={gy}: {e}")
            results.append({
                "index": i,
                "X": gx,
                "Y": gy,
                "center_nm": np.nan,
                "sigma_nm": np.nan,
                "gamma_nm": np.nan,
                "area_coeff_A": np.nan,
                "data_integral": np.nan,
                "fit_integral": np.nan,
            })

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "fit_summary.csv"), index=False)

    print("Finished")
    print(f"Images and CSV saved in: {out_dir}")


if __name__ == "__main__":
    main()
