# app.py — versão Exxata visual e explicativa
import time, json, hashlib, io
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt

# ----------- Estilo e identidade visual Exxata -----------
st.set_page_config(page_title="Simulação Monte Carlo – Exxata", layout="wide", page_icon="📈")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Manrope', sans-serif !important;
    }
    h1,h2,h3,h4 { color: #4284D7 !important; }
    .stMetricLabel { color: #78909C !important; }
    .stMetricValue { color: #D51D07 !important; font-weight:700 !important; }
    </style>
""", unsafe_allow_html=True)

# ----------- Cabeçalho -----------
st.title("📈 Simulação de Monte Carlo – Pleitos/Negociações (Triangular A/B/C)")
st.caption(
    "O modelo estima o **resultado provável de uma negociação** considerando três cenários: "
    "**Piso (A)** – valor mínimo esperado, **Provável (B)** – valor mediano, "
    "e **Teto (C)** – limite máximo possível. "
    "A simulação usa **distribuição triangular (A,B,C)** com 10.000+ iterações."
)

# ----------- Entradas -----------
with st.sidebar:
    st.header("Premissas do Cenário")
    item = st.text_input("Item (pleito/negociação)", "Negociação A")

    piso = st.number_input("A — Piso (R$)", min_value=0.0, value=2_000_000.0, step=100_000.0, format="%.2f")
    default_B = max(2_500_000.0, piso)
    provavel = st.number_input("B — Provável (R$)", min_value=piso, value=default_B, step=100_000.0, format="%.2f")
    default_C = max(3_500_000.0, provavel)
    teto = st.number_input("C — Teto (R$)", min_value=provavel, value=default_C, step=100_000.0, format="%.2f")

    iters = st.number_input("Iterações (≥10.000)", min_value=10_000, value=20_000, step=1_000)
    seed = st.number_input("Seed aleatória", value=20251015, step=1)

    st.markdown("---")
    st.caption("Análise de faixas de valor (opcional)")
    faixa1 = st.number_input("Limite 1 (R$)", min_value=0.0, value=2_000_000.0, step=100_000.0)
    faixa2 = st.number_input("Limite 2 (R$)", min_value=faixa1, value=3_000_000.0, step=100_000.0)
    faixa3 = st.number_input("Limite 3 (R$)", min_value=faixa2, value=4_000_000.0, step=100_000.0)

    rodar = st.button("🚀 Rodar simulação", use_container_width=True)

# ----------- Execução Monte Carlo -----------
if rodar:
    start = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    valores = rng.triangular(piso, provavel, teto, size=int(iters))
    sorted_vals = np.sort(valores)

    n = len(valores)
    mean, p50, p95 = np.mean(sorted_vals), np.quantile(sorted_vals, 0.5), np.quantile(sorted_vals, 0.95)
    duration_ms = int((time.perf_counter() - start)*1000)

    # Probabilidades por faixa
    pct_below_f1 = np.mean(sorted_vals < faixa1)
    pct_f1_f2 = np.mean((sorted_vals >= faixa1) & (sorted_vals < faixa2))
    pct_f2_f3 = np.mean((sorted_vals >= faixa2) & (sorted_vals < faixa3))
    pct_above_f3 = np.mean(sorted_vals >= faixa3)

    # Hash de auditoria (integridade)
    meta = dict(item=item, piso=piso, provavel=provavel, teto=teto, iterations=n, seed=int(seed), duration_ms=duration_ms)
    hash_str = json.dumps(meta, ensure_ascii=False).encode()
    verification_hash = hashlib.sha256(hash_str).hexdigest().upper()

    # ----------- KPIs -----------
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("EV (média)", f"R$ {mean:,.0f}".replace(",","."))
    c2.metric("P50 (Mediana)", f"R$ {p50:,.0f}".replace(",","."))
    c3.metric("P95 (Cenário alto)", f"R$ {p95:,.0f}".replace(",","."))
    c4.metric("Simulações", f"{n:,}".replace(",", "."))

    st.caption("**P50** representa o valor central (50% dos resultados são menores). "
               "**P95** é o limite superior provável (95% dos resultados são menores).")

    # ----------- Distribuição ----------- 
    left,right = st.columns([1,1])
    with left:
        st.subheader("📊 Distribuição de Resultados (R$)")
        fig, ax = plt.subplots()
        ax.hist(sorted_vals, bins=25, color="#D51D07", edgecolor="#B2B2BB")
        ax.set_xlabel("Valor Simulado (R$)")
        ax.set_ylabel("Frequência")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("📈 Curva Acumulada (CDF)")
        y = np.linspace(0, 1, n)
        fig2, ax2 = plt.subplots()
        ax2.plot(sorted_vals, y, color="#4284D7")
        ax2.set_xlabel("Valor (R$)")
        ax2.set_ylabel("Probabilidade acumulada")
        st.pyplot(fig2, clear_figure=True)

    # ----------- Faixas de negociação -----------
    st.markdown("### 🎯 Distribuição por Faixa de Acordo")
    colA,colB = st.columns([1,1])
    with colA:
        st.write(f"**{pct_below_f1*100:.2f}%** para acordo abaixo de **{faixa1/1_000_000:.1f}MM**")
        st.write(f"**{pct_f1_f2*100:.2f}%** para acordo entre **{faixa1/1_000_000:.1f}MM** e **{faixa2/1_000_000:.1f}MM**")
        st.write(f"**{pct_f2_f3*100:.2f}%** para acordo entre **{faixa2/1_000_000:.1f}MM** e **{faixa3/1_000_000:.1f}MM**")
        st.write(f"**{pct_above_f3*100:.2f}%** para acordo acima de **{faixa3/1_000_000:.1f}MM**")

    # ----------- Auditoria explicativa -----------
    st.markdown("### 🧾 Auditoria & Comprovação")
    st.write(
        f"Foram realizadas **{n:,} simulações** em {duration_ms} ms.\n\n"
        "O **hash** é um código gerado automaticamente que serve como *carimbo digital* "
        "do experimento: se alguém alterar qualquer parâmetro (piso, provável, teto, seed etc.), "
        "o hash muda — isso comprova a integridade e reprodutibilidade da simulação."
    )
    st.code(verification_hash, language="text")

    # ----------- Exportação CSV -----------
    df = pd.DataFrame([
        ["Item", item],
        ["Piso (A)", piso],
        ["Provável (B)", provavel],
        ["Teto (C)", teto],
        ["Iterações", n],
        ["Seed", seed],
        ["Duração (ms)", duration_ms],
        ["Hash", verification_hash],
        ["P50", p50],
        ["P95", p95],
        ["EV", mean],
        ["< Faixa1", pct_below_f1],
        ["Faixa1–Faixa2", pct_f1_f2],
        ["Faixa2–Faixa3", pct_f2_f3],
        ["> Faixa3", pct_above_f3],
    ], columns=["Parâmetro", "Valor"])

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("📥 Baixar CSV com Resultados", buf.getvalue().encode("utf-8"),
                       file_name=f"montecarlo_exxata_{int(time.time())}.csv", mime="text/csv")

else:
    st.info("Defina Piso (A), Provável (B) e Teto (C), depois clique em **Rodar simulação**.")
