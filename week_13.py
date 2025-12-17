# %%
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PERSIAPAN DATA (Dummy Seismic Data)
# ==========================================
# Fungsi untuk membuat data seismik sintetik sederhana
def generate_seismic_data(n_traces=100, n_samples=500):
    np.random.seed(42)
    # Membuat wavelet Ricker sederhana
    t = np.linspace(-0.1, 0.1, 100)
    f = 25  # frekuensi dominan
    ricker = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    
    # Membuat reflettivitas acak (sparse)
    reflectivity = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        indices = np.random.randint(0, n_samples, 5) # 5 reflektor per trace
        reflectivity[indices, i] = np.random.randn(5)
    
    # Konvolusi untuk membuat seismogram
    seismic_data = np.zeros_like(reflectivity)
    for i in range(n_traces):
        seismic_data[:, i] = np.convolve(reflectivity[:, i], ricker, mode='same')
    
    return seismic_data

# Judul Aplikasi
st.title("Visualisasi Data Seismik Interaktif")
st.markdown("Aplikasi untuk eksplorasi data seismik dengan kontrol tampilan dinamis.")

# Generate Data
data = generate_seismic_data()
n_samples, n_traces = data.shape

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Kontrol Visualisasi")

# --- A. Pilihan Colormap (Cmap) ---
st.sidebar.subheader("1. Warna (Colormap)")
cmap_options = ['gray', 'seismic', 'RdBu', 'viridis', 'magma', 'Greys']
selected_cmap = st.sidebar.selectbox("Pilih Colormap:", cmap_options, index=1)

# --- B. Opsi Auto/Manual Scale & Slider vmin/vmax ---
st.sidebar.subheader("2. Kontrol Amplitudo (Clipping)")
scale_mode = st.sidebar.radio("Mode Skala:", ("Auto Scale", "Manual Scale"))

vmin_val, vmax_val = None, None # Default untuk Auto

if scale_mode == "Manual Scale":
    # Menghitung batas absolut data untuk referensi slider
    abs_max = np.max(np.abs(data))
    
    st.sidebar.markdown("Atur batas pemotongan amplitudo:")
    # Slider vmin (biasanya negatif)
    vmin_val = st.sidebar.slider(
        "Nilai Minimum (vmin)", 
        min_value=-abs_max, max_value=0.0, value=-abs_max*0.5, step=0.01
    )
    # Slider vmax (biasanya positif)
    vmax_val = st.sidebar.slider(
        "Nilai Maksimum (vmax)", 
        min_value=0.0, max_value=abs_max, value=abs_max*0.5, step=0.01
    )
    
    st.info(f"Rentang aktif: {vmin_val:.2f} s.d. {vmax_val:.2f}")

# --- C. Opsi Tambahan: Memilih Rentang Trace ---
st.sidebar.subheader("3. Navigasi Data")
trace_range = st.sidebar.slider(
    "Pilih Rentang Trace untuk Ditampilkan:",
    min_value=0, max_value=n_traces, value=(0, n_traces)
)

# Fitur Tambahan: Membalik Sumbu Waktu (Opsional Kecil)
flip_time = st.sidebar.checkbox("Balik Sumbu Waktu (Reverse Time)", value=False)

# ==========================================
# 3. PLOTTING
# ==========================================
# Slice data berdasarkan rentang trace yang dipilih
start_trace, end_trace = trace_range
data_subset = data[:, start_trace:end_trace]

# Setup Figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot menggunakan imshow
# aspect='auto' agar gambar menyesuaikan ukuran layar
img = ax.imshow(
    data_subset, 
    aspect='auto', 
    cmap=selected_cmap, 
    vmin=vmin_val, 
    vmax=vmax_val,
    extent=[start_trace, end_trace, n_samples, 0] # Mapping sumbu X dan Y
)

# Labeling
ax.set_xlabel("Trace Number")
ax.set_ylabel("Time Sample")
ax.set_title(f"Seismic Section (Traces {start_trace}-{end_trace})")
ax.grid(False)

# Opsi Membalik Sumbu Y (Waktu)
if not flip_time:
    # Standar seismik: waktu 0 di atas, makin ke bawah makin besar
    ax.set_ylim(n_samples, 0) 
else:
    ax.set_ylim(0, n_samples)

# Menambahkan Colorbar
cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Amplitudo")

# Tampilkan di Streamlit
st.pyplot(fig)

# --- Fitur Tambahan: Download Gambar ---
st.markdown("### Unduh Hasil Plot")
fn = f"seismic_plot_{selected_cmap}.png"
plt.savefig(fn)
with open(fn, "rb") as file:
    btn = st.download_button(
        label="Download Plot sebagai Gambar",
        data=file,
        file_name=fn,
        mime="image/png"
    )