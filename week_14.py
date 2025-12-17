# %%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import os

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(layout="wide")
st.title("🧲 Magnetic Anomaly Analysis Tool")

# =========================
# FUNGSI LOAD DATA (AMAN)
# =========================
def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "test_magnetic.csv")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("File 'test_magnetic.csv' tidak ditemukan! Pastikan file ada di folder yang sama.")
        return None

df = load_data()

# =========================
# JIKA DATA ADA
# =========================
if df is not None:

    # ===== SIDEBAR =====
    st.sidebar.header("Settings")

    cmap_options = ['jet', 'viridis', 'seismic', 'RdBu_r', 'magma']
    selected_cmap = st.sidebar.selectbox("Pilih Colormap", cmap_options)

    scale_mode = st.sidebar.radio("Mode Skala Warna", ["Auto", "Manual"])

    val_min = float(df['t_obs'].min())
    val_max = float(df['t_obs'].max())

    if scale_mode == "Manual":
        vmin = st.sidebar.slider("vmin", val_min, val_max, val_min)
        vmax = st.sidebar.slider("vmax", val_min, val_max, val_max)
    else:
        vmin, vmax = None, None

    st.sidebar.subheader("Filter Data")
    x_min_data = float(df['x'].min())
    x_max_data = float(df['x'].max())
    x_range = st.sidebar.slider(
        "Rentang Trace X",
        x_min_data, x_max_data,
        (x_min_data, x_max_data)
    )

    # ===== PROCESSING =====
    df_filtered = df[(df['x'] >= x_range[0]) & (df['x'] <= x_range[1])]

    grid_x, grid_y = np.mgrid[
        df_filtered.x.min():df_filtered.x.max():100j,
        df_filtered.y.min():df_filtered.y.max():100j
    ]

    zi_obs = griddata(
        (df_filtered.x, df_filtered.y),
        df_filtered.t_obs,
        (grid_x, grid_y),
        method='linear'
    )

    zi_filled = np.nan_to_num(zi_obs, nan=np.nanmean(zi_obs))
    zi_reg = gaussian_filter(zi_filled, sigma=3)
    zi_res = zi_obs - zi_reg

    # ===== PLOTTING =====
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    data_to_plot = [zi_obs, zi_reg, zi_res]
    titles = [
        "Observasi (Total Field)",
        "Regional (Trend)",
        "Residual (Target)"
    ]

    for i, ax in enumerate(axs):
        im = ax.imshow(
            data_to_plot[i].T,
            extent=(x_range[0], x_range[1], df.y.min(), df.y.max()),
            origin='lower',
            cmap=selected_cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.colorbar(im, ax=ax, label='Anomali (nT)')

    st.pyplot(fig)

    # ===== SAVE IMAGE =====
    if st.button("💾 Simpan Plot sebagai Gambar"):
        fig.savefig("hasil_anomali.png", dpi=300)
        st.success("Gambar berhasil disimpan sebagai 'hasil_anomali.png'")

# %%
