"""
Dashboard Keterprediksian Inflasi Kota di Indonesia
Skripsi: Akmal Harizulhaq (23.11.5783), Informatika Universitas AMIKOM Yogyakarta

Semua angka dibaca langsung dari ringkasan_hasil.xlsx,
yaitu keluaran notebook Source_Code_Skripsi_Akmal_Final_v2.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Keterprediksian Inflasi Kota",
    page_icon="📈",
    layout="wide",
)

FILE_DATA = "ringkasan_hasil.xlsx"

# Warna tetap untuk tiga kategori fitur dan penanda
WARNA_KATEGORI = {"Inertia (lag)": "#2563eb", "Makro": "#ea580c", "Kalender": "#059669"}
WARNA_UTAMA = "#2563eb"
WARNA_NEGATIF = "#dc2626"
WARNA_REDUP = "#9ca3af"
WARNA_REF = "#6b7280"

# Nama di file Excel -> nama yang dipakai di draf skripsi
NAMA_MODEL = {
    "Naive (inflasi bulan t)": "Naive",
    "Musiman (rata-rata kota-bulan)": "Seasonal",
}
NAMA_KATEGORI = {"Inersia (lag)": "Inertia (lag)", "Musiman (kalender)": "Kalender", "Makro": "Makro"}
NAMA_VARIABEL = {
    "kurs_usd_idr_chg": "Kurs USD/IDR", "reer_chg": "REER", "cadangan_devisa_chg": "Cadangan devisa",
    "harga_minyak_chg": "Harga minyak Brent", "ihsg_chg": "IHSG", "bi_rate_chg": "Suku bunga acuan",
    "m2_chg": "M2", "PLASEBO acak (rata-rata 20 undian)": "Placebo acak (rata-rata 20 deret)",
    "kurs_usd_idr": "Kurs USD/IDR", "reer": "REER", "cadangan_devisa": "Cadangan devisa",
    "harga_minyak": "Harga minyak Brent", "ihsg": "IHSG", "bi_rate": "Suku bunga acuan", "m2": "M2",
}
NAMA_ABLASI = {
    "Lengkap (lag + makro + kalender)": "Lengkap (lag, makro, kalender)",
    "Tanpa makro": "Tanpa makro (7)",
    "Tanpa kalender": "Tanpa kalender (11)",
    "Tanpa event tambahan (kalender dasar saja)": "Tanpa additional event (6)",
    "Tanpa lag inflasi": "Tanpa lag inflasi (5)",
    "Hanya lag": "Hanya lag (18)",
    "Hanya kalender": "Hanya kalender (12)",
    "Lag + kalender + 4 makro berteori": "Lag, kalender, dan 4 makro berteori (3)",
    "Lengkap + inflasi nasional bulan t": "Lengkap dan inflasi nasional bulan t (+1)",
    "Lengkap tanpa kurs_usd_idr_chg": "Lengkap tanpa kurs (1)",
    "Lengkap tanpa reer_chg": "Lengkap tanpa REER (1)",
    "Lengkap tanpa cadangan_devisa_chg": "Lengkap tanpa cadangan devisa (1)",
    "Lengkap tanpa harga_minyak_chg": "Lengkap tanpa harga minyak (1)",
    "Lengkap tanpa ihsg_chg": "Lengkap tanpa IHSG (1)",
    "Lengkap tanpa bi_rate_chg": "Lengkap tanpa suku bunga acuan (1)",
    "Lengkap tanpa m2_chg": "Lengkap tanpa M2 (1)",
}
NAMA_EVENT = {
    "XGBoost - kalender dasar": "XGBoost, base calendar (5 fitur)",
    "XGBoost - kalender dasar + event tambahan": "XGBoost, base calendar dan additional event (11 fitur)",
    "Ridge - kalender dasar": "Ridge Regression, base calendar (5 fitur)",
    "Ridge - kalender dasar + event tambahan": "Ridge Regression, base calendar dan additional event (11 fitur)",
}
NAMA_DIAGNOSIS = {
    "R2_oof_2017": "R² LSTM out-of-fold 2017", "R2_oof_2018": "R² LSTM out-of-fold 2018",
    "R2_oof_2019": "R² LSTM out-of-fold 2019", "R2_oof_2020": "R² LSTM out-of-fold 2020",
    "R2_oof_2021": "R² LSTM out-of-fold 2021", "R2_oof_2017_2021": "R² LSTM out-of-fold 2017–2021",
    "R2_lstm_uji": "R² LSTM data uji 2022–2023", "korelasi_oof_aktual": "Korelasi LSTM dengan aktual, 2017–2021",
    "korelasi_lstm_uji_aktual": "Korelasi LSTM dengan aktual, data uji",
    "porsi_gain_lstm_%": "Porsi gain fitur LSTM (%)", "R2_hybrid_uji": "R² hybrid data uji",
}
KOLOM_PREDIKSI = {
    "XGBoost": "pred_XGBoost",
    "Hybrid LSTM-XGBoost": "pred_Hybrid LSTM-XGBoost",
    "LSTM": "pred_LSTM",
    "Ridge Regression": "pred_Ridge Regression",
    "XGBoost (sampel meta)": "pred_XGBoost (sampel meta)",
    "Seasonal": "pred_Musiman (rata-rata kota-bulan)",
    "Naive": "pred_Naive (inflasi bulan t)",
}
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


# ---------- Format angka Indonesia (koma desimal) ----------
def f(v, d=3, persen=False):
    """Format angka dengan koma desimal dan tanda minus asli, misalnya 0.0362 -> '0,036'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, str):
        return v
    s = f"{v:,.{d}f}".replace(",", "#").replace(".", ",").replace("#", ".")
    s = s.replace("-", "−")
    return s + ("%" if persen else "")


def tampil(df, kolom):
    """Ubah kolom angka menjadi teks berformat Indonesia untuk tabel."""
    out = df.copy()
    for k, d in kolom.items():
        out[k] = out[k].map(lambda v: f(v, d))
    return out


def warna_sel(v, skala=1.0, positif=WARNA_UTAMA, negatif=WARNA_NEGATIF):
    """Warna latar sel: biru untuk positif, merah untuk negatif, makin pekat makin besar."""
    if v is None or pd.isna(v):
        return ""
    a = min(1.0, abs(v) / skala) * 0.45 + 0.06
    hexwarna = positif if v >= 0 else negatif
    r, g, b = int(hexwarna[1:3], 16), int(hexwarna[3:5], 16), int(hexwarna[5:7], 16)
    return f"background-color: rgba({r},{g},{b},{a:.2f})"


def gaya_grafik(fig, tinggi=380):
    fig.update_layout(
        height=tinggi, separators=",.", margin=dict(l=10, r=20, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.6)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.6)")
    return fig


def batang_mendatar(df, label, nilai, warna, teks, hover, tinggi_baris=26, galat=None, garis_ref=None, judul_x="R²"):
    """Grafik batang mendatar, urutan baris sesuai DataFrame (atas ke bawah).
    Label nilai selalu diletakkan di sisi kanan batang (atau kanan garis nol untuk nilai negatif)."""
    v = df[nilai].astype(float).tolist()
    e = df[galat].fillna(0).astype(float).tolist() if galat else [0.0] * len(v)
    fig = go.Figure(go.Bar(
        y=df[label], x=v, orientation="h", marker_color=warna,
        error_x=dict(type="data", array=e, color="#374151", thickness=1.2) if galat else None,
        hovertext=hover, hoverinfo="text", showlegend=False,
    ))
    posisi = [max(0.0, a + b) for a, b in zip(v, e)]
    fig.add_trace(go.Scatter(
        x=posisi, y=df[label], mode="text", text=[" " + t for t in teks], textposition="middle right",
        hoverinfo="skip", showlegend=False, textfont=dict(size=12),
    ))
    kiri = min(0.0, min(a - b for a, b in zip(v, e)))
    kanan = max(0.0, max(posisi))
    rentang = (kanan - kiri) or 1.0
    fig.update_xaxes(range=[kiri - 0.03 * rentang, kanan + 0.16 * rentang], title=judul_x)
    if garis_ref is not None:
        fig.add_vline(x=garis_ref, line_dash="dash", line_color=WARNA_REF)
    fig.update_yaxes(autorange="reversed")
    return gaya_grafik(fig, tinggi=max(260, 60 + tinggi_baris * len(df)))


# ---------- Baca data ----------
@st.cache_data
def baca_data():
    x = pd.read_excel(FILE_DATA, sheet_name=None)
    d = {}
    m = x["1_Metrik_Model"].copy()
    m["Model"] = m["Model"].replace(NAMA_MODEL)
    d["model"] = m
    dm = x["2_Uji_DM"].copy()
    dm["Model A"] = dm["Model A"].replace(NAMA_MODEL)
    dm["Model B"] = dm["Model B"].replace(NAMA_MODEL)
    dm["Kesimpulan"] = dm["Kesimpulan"].str.replace("tidak berbeda nyata", "Tidak berbeda nyata")
    d["dm"] = dm
    w = x["3_Metrik_Wilayah"].copy()
    w["model"] = w["model"].replace(NAMA_MODEL)
    d["wilayah"] = w
    d["statistik"] = x["0_Statistik_Deskriptif"]
    adf = x["4_ADF"].copy()
    adf["variabel"] = adf["variabel"].replace(NAMA_VARIABEL)
    d["adf"] = adf
    d["adf_kota"] = x["4b_ADF_Kota"]
    dek = x["5_Dekomposisi"].copy()
    dek["wilayah"] = dek["wilayah"].replace({"SEMUA KOTA": "Seluruh kota"})
    d["dekomposisi"] = dek
    gr = x["6_Granger"].copy()
    gr["variabel"] = gr["variabel"].replace(NAMA_VARIABEL)
    d["granger"] = gr
    gk = x["29_Granger_Per_Kota"].copy()
    gk["variabel"] = gk["variabel"].replace(NAMA_VARIABEL)
    d["granger_kota"] = gk
    ab = x["7_Ablasi"].copy()
    ab["konfigurasi"] = ab["konfigurasi"].replace(NAMA_ABLASI)
    d["ablasi"] = ab
    d["ablasi_model"] = x["7b_Ablasi_Makro_Model"]
    d["plasebo"] = x["8c_Plasebo_Lengkap"]
    d["plasebo_undian"] = x["8b_Plasebo_undian"]
    sg = x["9_SHAP_Global"].copy()
    sg["kategori"] = sg["kategori"].replace(NAMA_KATEGORI)
    d["shap"] = sg
    sw = x["10_SHAP_Wilayah_Porsi"].copy()
    sw["kategori"] = sw["kategori"].replace(NAMA_KATEGORI)
    d["shap_wilayah"] = sw
    d["zscore"] = x["11_SHAP_Zscore_Makro"]
    d["koef_variasi"] = x["12_SHAP_KoefVariasi"]
    d["prediksi"] = x["13_Prediksi"]
    d["ridge"] = x["14_Ridge_Koefisien"]
    ev = x["15_Pengaruh_Event"].copy()
    ev["konfigurasi"] = ev["konfigurasi"].replace(NAMA_EVENT)
    d["event"] = ev
    ss = x["19_SHAP_Stabil_Seed"].copy()
    ss = ss.rename(columns={"Inersia (lag)": "Inertia (lag)", "Musiman (kalender)": "Kalender"})
    d["shap_seed"] = ss
    d["shap_peringkat"] = x["20_SHAP_Stabil_Peringkat"]
    d["diagnosis"] = x["25_Diagnosis_Hybrid"]
    d["tertimbang"] = x["26_Rata2_Tertimbang"]
    lu = x["27_Latih_Ulang_Bulanan"].copy()
    lu["model"] = lu["model"].replace(NAMA_MODEL)
    d["latih_ulang"] = lu
    return d


D = baca_data()

# Angka-angka utama yang dipakai berulang
model = D["model"]
terbaik = model.loc[model["R2"].idxmax()]
plasebo = dict(zip(D["plasebo"]["keterangan"], D["plasebo"]["nilai"]))
p_plasebo = plasebo["Nilai probabilitas empiris"]
dek = D["dekomposisi"].set_index("wilayah")
porsi_kat = D["shap"].groupby("kategori")["porsi_%"].sum()

# ---------- Kepala halaman ----------
st.caption("Skripsi · Akmal Harizulhaq (23.11.5783) · 82 kota IHK · data uji Januari 2022 – Desember 2023")
st.title("Keterprediksian Inflasi Kota di Indonesia")
st.markdown(
    f"Model terbaik, **{terbaik['Model']}**, hanya mencapai R² **{f(terbaik['R2'])}** pada inflasi bulanan kota. "
    f"Tambahan dari tujuh variabel makroekonomi nasional **belum terbukti** (nilai p *placebo* {f(p_plasebo)}, batas 0,05), "
    f"dan pola kalender menjadi pendorong prediksi terbesar ({f(porsi_kat['Kalender'], 1)}% kontribusi SHAP)."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("R² tertinggi pada data uji", f(terbaik["R2"]), help="Rata-rata 5 seed. Nilai 0 sama dengan menebak rata-rata data uji.")
k1.caption(f"{terbaik['Model']} · RMSE {f(terbaik['RMSE'])}. Hanya berbeda nyata dari model *naive*.")
k2.metric("Nilai p *placebo test* makro", f(p_plasebo))
k2.caption(("✅ Terbukti" if p_plasebo < 0.05 else "⚠️ Belum terbukti") + " · 999 deret acak VAR(1), batas 0,05.")
k3.metric("Ragam inflasi yang khas per kota", f(dek.loc["Seluruh kota", "porsi_khas_%"], 1) + "%")
k3.caption(f"Jawa {f(dek.loc['Jawa', 'porsi_khas_%'], 1)}%, Maluku-Papua {f(dek.loc['Maluku-Papua', 'porsi_khas_%'], 1)}%.")
k4.metric("Kontribusi SHAP fitur kalender", f(porsi_kat["Kalender"], 1) + "%")
k4.caption(f"Inertia {f(porsi_kat['Inertia (lag)'], 1)}%, Makro {f(porsi_kat['Makro'], 1)}%.")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Akurasi model (RM2)", "Variabel makro (RM1)", "SHAP (RM3)",
    "Data & uji kelayakan", "Analisis tambahan", "Istilah",
])

# ======================= TAB 1: AKURASI =======================
with tab1:
    st.subheader("Akurasi model pada data uji")
    st.markdown(
        "R² mengukur seberapa besar ragam inflasi yang dijelaskan model. Nilai negatif berarti kesalahan model "
        "lebih besar daripada menebak rata-rata data uji. Angka adalah rata-rata 5 *seed*; garis tipis adalah "
        "simpangan baku antar-*seed*."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        warna = [WARNA_UTAMA if r == terbaik["R2"] else WARNA_REDUP for r in model["R2"]]
        hover = [f"<b>{r.Model}</b><br>R² {f(r.R2, 4)}<br>Simpangan baku {f(r.R2_std, 4)}<br>RMSE {f(r.RMSE, 4)}<br>MAE {f(r.MAE, 4)}"
                 for r in model.itertuples()]
        st.plotly_chart(batang_mendatar(model, "Model", "R2", warna, [f(v) for v in model["R2"]], hover,
                                        tinggi_baris=40, galat="R2_std"), use_container_width=True)
    with c2:
        st.markdown("**Metrik lengkap (Tabel 4.5)**")
        t = model[["Model", "RMSE", "MAE", "R2", "R2_ensemble"]].rename(columns={"R2": "R²", "R2_ensemble": "R² ensemble"})
        st.dataframe(tampil(t, {"RMSE": 4, "MAE": 4, "R²": 4, "R² ensemble": 4}), hide_index=True, use_container_width=True)
        st.caption("R² ensemble dihitung dari prediksi rata-rata kelima seed.")

    st.markdown("#### Rata-rata inflasi aktual dan prediksi per bulan (Gambar 4.1)")
    s1, s2 = st.columns(2)
    pilih_model = s1.selectbox("Model", list(KOLOM_PREDIKSI.keys()))
    pilih_wilayah = s2.selectbox("Wilayah", ["Seluruh kota", "Sumatera", "Jawa", "Bali-Nusra", "Kalimantan", "Sulawesi", "Maluku-Papua"])
    p = D["prediksi"]
    if pilih_wilayah != "Seluruh kota":
        p = p[p["wilayah"] == pilih_wilayah]
    rata = p.groupby("target_date")[["aktual", KOLOM_PREDIKSI[pilih_model]]].mean().reset_index()
    label_bulan = [f"{BULAN[t.month - 1]} {t.year}" for t in rata["target_date"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rata["target_date"], y=rata["aktual"], name="Aktual", mode="lines+markers",
                             line=dict(color="#f59e0b", width=3), customdata=label_bulan,
                             hovertemplate="%{customdata}<br>Aktual %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=rata["target_date"], y=rata[KOLOM_PREDIKSI[pilih_model]], name=pilih_model,
                             mode="lines+markers", line=dict(color=WARNA_UTAMA, width=2.5), customdata=label_bulan,
                             hovertemplate="%{customdata}<br>Prediksi %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(title="Inflasi bulanan (%)")
    fig.update_xaxes(tickvals=rata["target_date"][::3], ticktext=label_bulan[::3])
    st.plotly_chart(gaya_grafik(fig, 380), use_container_width=True)
    st.caption(
        f"Rata-rata sederhana dari {p['kota'].nunique()} kota. Lonjakan September 2022 berasal dari penyesuaian "
        "harga BBM bersubsidi, yang tidak tertangkap model mana pun."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Uji Diebold-Mariano (Tabel 4.6)**")
        t = D["dm"].rename(columns={"DM": "DM", "p_value": "p"})
        st.dataframe(tampil(t, {"DM": 3, "p": 3}), hide_index=True, use_container_width=True, height=(len(t) + 1) * 35 + 3)
        st.caption("DM negatif berarti model A lebih akurat. Berbeda nyata bila p < 0,05.")
    with c2:
        st.markdown("**R² per wilayah (Tabel 4.7)**")
        urut_model = ["Naive", "Seasonal", "Ridge Regression", "XGBoost", "Hybrid LSTM-XGBoost"]
        urut_wil = ["Sumatera", "Jawa", "Bali-Nusra", "Kalimantan", "Sulawesi", "Maluku-Papua"]
        piv = D["wilayah"].pivot(index="wilayah", columns="model", values="R2").loc[urut_wil, urut_model]
        piv.index.name, piv.columns.name = "Wilayah", None
        st.dataframe(piv.style.map(lambda v: warna_sel(v, 0.4)).format(lambda v: f(v)),
                     use_container_width=True)
        st.caption("Biru berarti R² positif, merah negatif. Maluku-Papua paling terbantu model; "
                   "di Jawa semua model negatif karena inflasinya tenang.")

    st.markdown("**Pengaruh fitur *additional event* (Tabel 4.8)**")
    t = D["event"].rename(columns={"konfigurasi": "Konfigurasi", "R2": "R²", "DM_vs_dasar": "DM terhadap base calendar", "p_vs_dasar": "p"})
    st.dataframe(tampil(t, {"RMSE": 4, "MAE": 4, "R²": 4, "DM terhadap base calendar": 3, "p": 3}),
                 hide_index=True, use_container_width=True)
    st.caption("Enam fitur additional event tidak memperbaiki akurasi secara nyata.")

# ======================= TAB 2: MAKRO =======================
with tab2:
    st.subheader("Tambahan daya prediktif variabel makro")
    st.markdown(
        "Tiga uji dipakai: *ablation test* (fitur dibuang), *placebo test* (makro asli diganti deret acak VAR(1) "
        "yang berstruktur mirip), dan uji Diebold-Mariano pada setiap model."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ablation test XGBoost (Tabel 4.9, rata-rata 3 seed)**")
        ab = D["ablasi"]
        lengkap = ab.iloc[0]["R2"]
        warna = [WARNA_UTAMA if v >= 0 else WARNA_NEGATIF for v in ab["R2"]]
        hover = [f"<b>{r.konfigurasi}</b><br>Jumlah fitur {r.n_fitur}<br>R² {f(r.R2, 4)}<br>"
                 f"Simpangan baku {f(r.R2_std, 4)}<br>Selisih {f(r.selisih_vs_lengkap, 4)}" for r in ab.itertuples()]
        st.plotly_chart(batang_mendatar(ab, "konfigurasi", "R2", warna, [f(v) for v in ab["R2"]], hover,
                                        galat="R2_std", garis_ref=lengkap), use_container_width=True)
        st.caption("Garis putus-putus adalah R² model lengkap. Angka dalam kurung adalah jumlah fitur yang dibuang.")
    with c2:
        st.markdown("**Distribusi R² pada 999 undian placebo (Gambar 4.3)**")
        asli = plasebo["R² dengan variabel makro asli"]
        tanpa = plasebo["R² tanpa variabel makro"]
        p95 = plasebo["Persentil ke-95 R² placebo"]
        fig = go.Figure(go.Histogram(x=D["plasebo_undian"]["R2_plasebo"], nbinsx=45, marker_color=WARNA_REDUP,
                                     name="Undian placebo", hovertemplate="R² %{x}<br>%{y} undian<extra></extra>"))
        for nilai, warna, gaya, nama in [(asli, WARNA_UTAMA, "solid", "Makro asli"), (tanpa, WARNA_NEGATIF, "dash", "Tanpa makro"),
                                         (p95, WARNA_REF, "dot", "Persentil ke-95")]:
            fig.add_vline(x=nilai, line_color=warna, line_dash=gaya, line_width=2.5)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=warna, dash=gaya, width=2.5),
                                     name=f"{nama} ({f(nilai, 4)})"))
        fig.update_xaxes(title="R² data uji dengan makro diganti placebo")
        fig.update_yaxes(title="Jumlah undian")
        st.plotly_chart(gaya_grafik(fig, 420), use_container_width=True)
        st.markdown(
            f"**{plasebo['Jumlah undian dengan R² >= R² asli']}** undian menghasilkan R² setara atau lebih tinggi "
            f"daripada makro asli, sehingga **p = {f(p_plasebo)}**. Karena sedikit di atas 0,05, tambahan dari makro "
            "belum dapat dibedakan dari kebetulan."
        )

    st.markdown("**Membuang makro pada setiap model (Tabel 4.10, 5 seed)**")
    t = D["ablasi_model"][["model", "R2_dengan_makro", "R2_tanpa_makro", "selisih", "seed_naik", "DM", "p_value"]].rename(columns={
        "model": "Model", "R2_dengan_makro": "R² dengan makro", "R2_tanpa_makro": "R² tanpa makro", "selisih": "Selisih",
        "seed_naik": "Seed dengan selisih positif", "p_value": "p"})
    st.dataframe(tampil(t, {"R² dengan makro": 4, "R² tanpa makro": 4, "Selisih": 4, "DM": 3, "p": 3}),
                 hide_index=True, use_container_width=True)
    st.caption("Tidak ada model yang berbeda nyata (semua p > 0,05).")

    st.markdown("**Ringkasan placebo test (Tabel 4.11)**")
    t = D["plasebo"].copy()
    t["nilai"] = t["nilai"].map(lambda v: f(v, 4) if not isinstance(v, str) else v)
    st.dataframe(t.rename(columns={"keterangan": "Keterangan", "nilai": "Nilai"}), hide_index=True, use_container_width=True)

# ======================= TAB 3: SHAP =======================
with tab3:
    st.subheader("Faktor pendorong prediksi menurut SHAP")
    st.markdown(
        "Porsi adalah rata-rata nilai mutlak SHAP setiap fitur dibagi totalnya, dari XGBoost pada 1.968 baris data uji "
        "dan 5 *seed*. bulan_sin dan bulan_cos adalah satu pengodean posisi bulan, sehingga dibaca bersama "
        f"({f(D['shap'].set_index('fitur').loc[['bulan_sin', 'bulan_cos'], 'porsi_%'].sum(), 1)}%)."
    )
    st.markdown("🟦 Inertia (lag) &nbsp;&nbsp; 🟧 Makro &nbsp;&nbsp; 🟩 Kalender")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Porsi setiap fitur (Gambar 4.4)**")
        sg = D["shap"].sort_values("porsi_%", ascending=False)
        warna = [WARNA_KATEGORI[k] for k in sg["kategori"]]
        hover = [f"<b>{r.fitur}</b><br>{r.kategori}<br>Porsi {f(r._4, 2)}%<br>Rata-rata |SHAP| {f(r.mean_abs_shap, 4)}"
                 for r in sg.itertuples()]
        st.plotly_chart(batang_mendatar(sg, "fitur", "porsi_%", warna, [f(v, 1, True) for v in sg["porsi_%"]], hover,
                                        tinggi_baris=22, judul_x="Porsi (%)"), use_container_width=True)
    with c2:
        st.markdown("**Porsi kategori per wilayah (Tabel 4.13)**")
        sw = D["shap_wilayah"]
        fig = go.Figure()
        for kat in ["Inertia (lag)", "Makro", "Kalender"]:
            s = sw[sw["kategori"] == kat]
            fig.add_trace(go.Bar(
                x=s["wilayah"], y=s["porsi_%"], name=kat, marker_color=WARNA_KATEGORI[kat],
                error_y=dict(type="data", symmetric=False, array=s["CI95_atas"] - s["porsi_%"],
                             arrayminus=s["porsi_%"] - s["CI95_bawah"], color="#374151", thickness=1.2),
                customdata=[[f(a, 1), f(b, 1), f(c, 1)] for a, b, c in zip(s["porsi_%"], s["CI95_bawah"], s["CI95_atas"])],
                hovertemplate="%{x} · " + kat + "<br>Porsi %{customdata[0]}%<br>Selang 95%: %{customdata[1]} – %{customdata[2]}<extra></extra>",
            ))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Porsi SHAP (%)")
        st.plotly_chart(gaya_grafik(fig, 420), use_container_width=True)
        st.caption("Garis tipis adalah selang kepercayaan 95% dari bootstrap 1.000 kali atas kota. "
                   "Selang yang tidak tumpang tindih menandakan perbedaan yang tegas.")
        t = sw.copy()
        t["teks"] = [f"{f(a, 1)} [{f(b, 1)}–{f(c, 1)}]" for a, b, c in zip(t["porsi_%"], t["CI95_bawah"], t["CI95_atas"])]
        piv = t.pivot(index="wilayah", columns="kategori", values="teks").loc[
            ["Sumatera", "Jawa", "Bali-Nusra", "Kalimantan", "Sulawesi", "Maluku-Papua"], ["Inertia (lag)", "Makro", "Kalender"]]
        piv.index.name, piv.columns.name = "Wilayah", None
        st.dataframe(piv, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kestabilan atribusi antar-seed (Tabel 4.12)**")
        st.dataframe(tampil(D["shap_seed"].rename(columns={"seed": "Seed"}), {"Inertia (lag)": 2, "Makro": 2, "Kalender": 2}), hide_index=True, use_container_width=True)
        pr = dict(zip(D["shap_peringkat"]["keterangan"], D["shap_peringkat"]["nilai"]))
        st.markdown(
            f"Korelasi Spearman peringkat fitur antar-seed minimal **{f(pr['korelasi Spearman peringkat fitur antar-seed (min)'])}** "
            f"dengan rata-rata **{f(pr['korelasi Spearman peringkat fitur antar-seed (rata-rata)'])}**. "
            f"Lima fitur teratas sama pada **{pr['fitur 5 teratas sama di semua seed']}** seed."
        )
    with c2:
        st.markdown("**Z-score kontribusi makro antarwilayah (Tabel 4.14)**")
        z = D["zscore"].set_index("wilayah").rename(columns={
            "kurs_usd_idr_chg": "Kurs", "reer_chg": "REER", "cadangan_devisa_chg": "Cad. devisa", "harga_minyak_chg": "Minyak",
            "ihsg_chg": "IHSG", "bi_rate_chg": "Suku bunga", "m2_chg": "M2"})
        z.index.name = "Wilayah"
        st.dataframe(z.style.map(lambda v: warna_sel(v, 1.8)).format(lambda v: f(v, 2)), use_container_width=True)
        st.caption("Dibaca ke bawah: membandingkan besar kontribusi satu variabel antarwilayah. "
                   "Biru di atas rata-rata enam wilayah, merah di bawahnya.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fitur yang paling membedakan wilayah (koefisien variasi)**")
        kv = D["koef_variasi"].sort_values("koef_variasi_%", ascending=False)
        kat = D["shap"].set_index("fitur")["kategori"]
        warna = [WARNA_KATEGORI.get(kat.get(x), WARNA_REDUP) for x in kv["fitur"]]
        hover = [f"<b>{a}</b><br>Koefisien variasi {f(b, 2)}%" for a, b in zip(kv["fitur"], kv["koef_variasi_%"])]
        st.plotly_chart(batang_mendatar(kv, "fitur", "koef_variasi_%", warna, [f(v, 1, True) for v in kv["koef_variasi_%"]], hover,
                                        tinggi_baris=22, judul_x="Koefisien variasi antarwilayah (%)"), use_container_width=True)
        st.caption("Koefisien variasi = simpangan baku ÷ rata-rata kontribusi SHAP di enam wilayah. "
                   "Makin besar, makin berbeda peran fitur itu antarwilayah.")
    with c2:
        st.markdown("**Arah pengaruh: koefisien Ridge terstandar**")
        rk = D["ridge"].sort_values("koefisien_terstandar", ascending=False)
        warna = [WARNA_UTAMA if v >= 0 else WARNA_NEGATIF for v in rk["koefisien_terstandar"]]
        hover = [f"<b>{a}</b><br>Koefisien {f(b, 4)}" for a, b in zip(rk["fitur"], rk["koefisien_terstandar"])]
        st.plotly_chart(batang_mendatar(rk, "fitur", "koefisien_terstandar", warna, [f(v) for v in rk["koefisien_terstandar"]], hover,
                                        tinggi_baris=22, judul_x="Koefisien terstandar"), use_container_width=True)
        st.caption("Biru menaikkan prediksi, merah menurunkan. Searah dengan beeswarm plot (Gambar 4.5).")

# ======================= TAB 4: DATA & KELAYAKAN =======================
with tab4:
    st.subheader("Data dan uji kelayakan")
    st.markdown("**Statistik deskriptif inflasi bulanan per wilayah, 2014–2023 (Tabel 4.1)**")
    t = D["statistik"].rename(columns={"wilayah": "Wilayah", "jumlah_kota": "Jumlah kota", "rata_rata": "Rata-rata (%)",
                                       "simpangan_baku": "Simpangan baku", "minimum": "Minimum", "maksimum": "Maksimum"})
    st.dataframe(tampil(t, {"Rata-rata (%)": 3, "Simpangan baku": 3, "Minimum": 2, "Maksimum": 2}), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Uji ADF variabel makro (Tabel 4.2)**")
        t = D["adf"].copy()
        for k in ["stasioner_level", "stasioner_perubahan"]:
            t[k] = t[k].map({True: "Ya", False: "Tidak"})
        t = t.rename(columns={"variabel": "Variabel", "p_level": "p level", "stasioner_level": "Level",
                              "p_perubahan": "p perubahan bulanan", "stasioner_perubahan": "Perubahan"})
        st.dataframe(tampil(t, {"p level": 4, "p perubahan bulanan": 4}), hide_index=True, use_container_width=True)
    with c2:
        ak = D["adf_kota"]
        jumlah = int((ak["p_adf_aic"] < 0.05).sum())
        st.markdown(f"**{jumlah} dari {len(ak)} kota stasioner.** Kota yang tidak stasioner dengan lag AIC:")
        t = ak[ak["p_adf_aic"] >= 0.05].rename(columns={"kota": "Kota", "p_adf_aic": "p (lag AIC)", "lag_aic": "Jumlah lag",
                                                        "p_adf_lag1": "p (1 lag)"})
        t = tampil(t, {"p (lag AIC)": 3})
        t["p (1 lag)"] = "< 0,0001"
        st.dataframe(t, hide_index=True, use_container_width=True)
        st.caption("Dengan 1 lag, semua kota stasioner.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Porsi ragam bersama dan khas (Tabel 4.3)**")
        dk = D["dekomposisi"]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=dk["wilayah"], x=dk["porsi_bersama_%"], orientation="h", name="Bersama", marker_color=WARNA_UTAMA,
                             text=[f(v, 1, True) for v in dk["porsi_bersama_%"]], textposition="inside",
                             hovertemplate="%{y}<br>Bersama %{x:.1f}%<extra></extra>"))
        fig.add_trace(go.Bar(y=dk["wilayah"], x=dk["porsi_khas_%"], orientation="h", name="Khas per kota", marker_color=WARNA_REDUP,
                             text=[f(v, 1, True) for v in dk["porsi_khas_%"]], textposition="inside",
                             hovertemplate="%{y}<br>Khas %{x:.1f}%<extra></extra>"))
        fig.update_layout(barmode="stack")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Porsi ragam (%)", range=[0, 100])
        st.plotly_chart(gaya_grafik(fig, 360), use_container_width=True)
    with c2:
        st.markdown("**Persentase kota dengan hubungan Granger signifikan (Tabel 4.4)**")
        t = D["granger"].rename(columns={"variabel": "Variabel", "kota_signifikan_%": "Sebelum koreksi (%)",
                                         "kota_signifikan_BH_%": "Setelah koreksi BH (%)"})
        st.dataframe(tampil(t, {"Sebelum koreksi (%)": 1, "Setelah koreksi BH (%)": 1}), hide_index=True, use_container_width=True)
        gk = D["granger_kota"]
        lolos = gk[gk["signifikan_BH"] == True]  # noqa: E712
        st.markdown("Setelah koreksi Benjamini-Hochberg, yang bertahan hanya: " +
                    ", ".join(f"**{r.kota}** ({r.variabel}, p = {f(r.p_value, 6)})" for r in lolos.itertuples()) + ".")

# ======================= TAB 5: ANALISIS TAMBAHAN =======================
with tab5:
    st.subheader("Mengapa hybrid kalah, dan apakah hasil tetap bertahan")
    dg = D["diagnosis"].set_index("keterangan")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**R² prediksi LSTM per periode (Tabel 4.15)**")
        urut = [("R2_oof_2017", "OOF 2017"), ("R2_oof_2018", "OOF 2018"), ("R2_oof_2019", "OOF 2019"), ("R2_oof_2020", "OOF 2020"),
                ("R2_oof_2021", "OOF 2021"), ("R2_oof_2017_2021", "OOF 2017–2021"), ("R2_lstm_uji", "Data uji 2022–2023")]
        t = pd.DataFrame([{"label": lab, "kunci": k, **dg.loc[k].to_dict()} for k, lab in urut])
        warna = [WARNA_NEGATIF if k == "R2_lstm_uji" else WARNA_UTAMA if k == "R2_oof_2017_2021" else WARNA_REDUP for k in t["kunci"]]
        hover = [f"<b>{r.label}</b><br>Rata-rata 5 seed {f(r.rata_rata)}<br>Minimum {f(r.min)}<br>Maksimum {f(r.max)}" for r in t.itertuples()]
        st.plotly_chart(batang_mendatar(t, "label", "rata_rata", warna, [f(v) for v in t["rata_rata"]], hover, tinggi_baris=36),
                        use_container_width=True)
        st.markdown(
            f"Pada 2017–2021, prediksi LSTM *out-of-fold* cukup informatif sehingga meta-model memberi porsi *gain* "
            f"**{f(dg.loc['porsi_gain_lstm_%', 'rata_rata'], 1)}%** kepadanya. Pada data uji, R² LSTM turun ke "
            f"**{f(dg.loc['R2_lstm_uji', 'rata_rata'])}**, sehingga ketergantungan itu menurunkan akurasi *hybrid*."
        )
        with st.expander("Tabel lengkap"):
            t2 = D["diagnosis"].copy()
            t2["keterangan"] = t2["keterangan"].replace(NAMA_DIAGNOSIS)
            t2 = t2.rename(columns={"keterangan": "Keterangan", "rata_rata": "Rata-rata 5 seed", "min": "Minimum", "max": "Maksimum"})
            st.dataframe(tampil(t2, {"Rata-rata 5 seed": 3, "Minimum": 3, "Maksimum": 3}), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Weighted average XGBoost dan LSTM**")
        t = D["tertimbang"].copy()
        t["bobot_LSTM"] = t["bobot_LSTM"].map(lambda v: f"{round(v * 100)}%")
        st.dataframe(tampil(t.rename(columns={"bobot_LSTM": "Bobot LSTM", "R2": "R² data uji"}), {"R² data uji": 4}),
                     hide_index=True, use_container_width=True)
        st.caption("Bobot LSTM 25% hanya menaikkan R² sedikit, dan bobot itu baru diketahui setelah melihat data uji.")

    st.markdown("**Single training dibandingkan monthly retraining (Tabel 4.16)**")
    t = D["latih_ulang"].rename(columns={"model": "Model", "R2_satu_kali": "R² single training", "R2_bulanan": "R² monthly retraining",
                                         "DM_bulanan_vs_satu_kali": "DM", "p_value": "p", "bulan_bulanan_unggul": "Bulan retraining unggul"})
    st.dataframe(tampil(t, {"R² single training": 3, "R² monthly retraining": 3, "DM": 3, "p": 3}), hide_index=True, use_container_width=True)
    st.caption("Semua perbedaan tidak signifikan.")

# ======================= TAB 6: ISTILAH =======================
with tab6:
    st.subheader("Arti istilah")
    st.markdown("""
- **R²**: porsi ragam inflasi data uji yang dijelaskan model. 0 sama dengan menebak rata-rata; negatif berarti lebih buruk dari itu.
- **RMSE, MAE**: rata-rata besar kesalahan prediksi dalam poin persen. Makin kecil makin baik.
- **Uji Diebold-Mariano**: menguji apakah selisih kesalahan dua model selama 24 bulan data uji cukup konsisten untuk disebut nyata, dengan koreksi HLN untuk sampel kecil.
- **Placebo test**: tujuh variabel makro diganti deret acak buatan VAR(1) sebanyak 999 kali. Nilai p adalah porsi undian yang R²-nya setara atau lebih tinggi daripada makro asli.
- **SHAP**: membagi setiap prediksi menjadi sumbangan tiap fitur. Porsi global adalah rata-rata nilai mutlaknya.
- **Z-score**: membandingkan besar kontribusi satu variabel di satu wilayah dengan rata-rata enam wilayah.
- **Koefisien variasi**: simpangan baku ÷ rata-rata kontribusi satu fitur di enam wilayah; makin besar, makin berbeda perannya antarwilayah.
""")
    st.caption("Sumber: ringkasan_hasil.xlsx, keluaran notebook Source_Code_Skripsi_Akmal_Final_v2. Angka sama dengan tabel Bab IV skripsi.")
