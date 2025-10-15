# app.py
import time
import json
import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------- Config inicial ---------------
st.set_page_config(
    page_title="Monte Carlo – Pleitos/Negociações",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Simulação de Monte Carlo – Pleitos/Negociações (Triangular A/B/C)")
st.write(
    "Insira **A (piso)**, **B (provável)** e **C (teto)**. "
    "A simulação usa **Triangular(A,B,C)** diretamente no valor a receber. "
    "Mínimo de **10.000** iterações. A seção de auditoria comprova execução (n, seed, hash) e **nenhuma cancelada**."
)

# --------------- Entradas ---------------------
with st.sidebar:
    st.header("Premissas")
    item = st.text_input("Item (o que está sendo pleiteado)", value="Pleito/Negociação A")
    piso = st.number_input("A — Piso (R$)", min_value=0.0, value=300000.0, step=1000.0, format="%.2f")
    provavel = st.number_input("B — Provável (R$)", min_value=float(piso), value=600000.0, step=1000.0, format="%.2f")
    teto = st.number_input("C — Teto (R$)", min_value=float(provavel), value=950000.0, step=1000.0, format="%.2f")

    iters = st.number_input("Iterações (≥ 10.000)", min_value=10000, value=20000, step=1000)
    seed = st.number_input("Seed (inteiro)", value=20251015, step=1)

    st.markdown("---")
    st.caption("Opcional – Meta para análise de sucesso")
    meta_valor = st.number_input(
        "Meta (R$) – calc. prob. de atingir/exceder",
        min_value=0.0, value=float(provavel), step=1000.0, format="%.2f"
    )

    rodar = st.button("🚀 Rodar simulação", use_container_width=True)

# --------------- Validação --------------------
def validar_triangular(a, b, c):
    return (a <= b) and (b <= c)

if not validar_triangular(piso, provavel, teto):
    st.error("Triangular inválida: garanta A ≤ B ≤ C (piso ≤ provável ≤ teto).")
    st.stop()

# --------------- Execução ---------------------
if rodar:
    start = time.perf_counter()

    rng = np.random.default_rng(int(seed))
    # numpy já tem triangular(min, mode, max, size)
    valores = rng.triangular(left=piso, mode=provavel, right=teto, size=int(iters))

    # Estatísticas
    n = valores.size
    sorted_vals = np.sort(valores)
    mean = float(sorted_vals.mean())
    p5 = float(np.quantile(sorted_vals, 0.05))
    p50 = float(np.quantile(sorted_vals, 0.50))
    p95 = float(np.quantile(sorted_vals, 0.95))

    # Probabilidade de sucesso vs. metas (opcionais/interpretativas)
    prob_meta = float((sorted_vals >= meta_valor).mean())  # P(Valor ≥ Meta)
    prob_ge_B = float((sorted_vals >= provavel).mean())    # P(Valor ≥ Provável)
    prob_ge_A = float((sorted_vals >= piso).mean())        # P(Valor ≥ Piso) -> sempre 1.0 por definição

    duration_ms = int((time.perf_counter() - start) * 1000)

    # Auditoria/meta
    meta = {
        "item": item,
        "triangular": {"piso": piso, "provavel": provavel, "teto": teto},
        "iterations": int(n),
        "seed": int(seed),
        "duration_ms": duration_ms,
        "cancelled": 0
    }

    # Hash de verificação (premissas + amostra dos resultados)
    sample = sorted_vals[:min(100, n)].tolist()
    hash_input = json.dumps({"meta": meta, "sample": sample}, ensure_ascii=False).encode("utf-8")
    verification_hash = hashlib.sha256(hash_input).hexdigest().upper()

    # --------------- KPIs ---------------------
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("EV (média)", f"R$ {mean:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col2.metric("P50 (mediana)", f"R$ {p50:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col3.metric("P95", f"R$ {p95:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col4.metric("Iterações", f"{n:,}".replace(",", "."))
    col5.metric("Canceladas", "0")
    col6.metric("Duração (ms)", f"{duration_ms}")

    # --------------- Gráficos -----------------
    left, right = st.columns(2)

    # Histograma
    with left:
        st.subheader("Distribuição de Resultado (R$)")
        fig1 = plt.figure()
        plt.hist(sorted_vals, bins=30)  # sem cores específicas
        plt.axvline(p50, linestyle="--", label="P50")
        plt.axvline(p95, linestyle=":", label="P95")
        plt.xlabel("Valor simulado (R$)")
        plt.ylabel("Frequência")
        plt.legend()
        st.pyplot(fig1, clear_figure=True)

    # CDF empírica
    with right:
        st.subheader("Curva Acumulada (CDF)")
        y = np.linspace(0, 1, n, endpoint=True)
        fig2 = plt.figure()
        plt.plot(sorted_vals, y)  # sem cores específicas
        plt.xlabel("Valor simulado (R$)")
        plt.ylabel("Probabilidade acumulada")
        st.pyplot(fig2, clear_figure=True)

    # --------------- Probabilidades de "sucesso" -----------------
    st.subheader("Análises de sucesso")
    st.write(
        f"- **P(Valor ≥ Meta {meta_valor:,.2f})** = **{100*prob_meta:.2f}%**  \n"
        f"- **P(Valor ≥ B (provável) {provavel:,.2f})** = **{100*prob_ge_B:.2f}%**  \n"
        f"- **P(Valor ≥ A (piso) {piso:,.2f})** = **{100*prob_ge_A:.2f}%**"
    )

    # --------------- Auditoria ----------------
    st.subheader("Auditoria & Comprovação")
    st.json({
        "item": meta["item"],
        "triangular": meta["triangular"],
        "iterations": meta["iterations"],
        "seed": meta["seed"],
        "duration_ms": meta["duration_ms"],
        "cancelled": meta["cancelled"],
        "verification_hash": verification_hash
    })

    # --------------- Download CSV -------------
    df_meta = pd.DataFrame([
        ["Item", meta["item"]],
        ["Piso (A)", piso],
        ["Provável (B)", provavel],
        ["Teto (C)", teto],
        ["Iterações", n],
        ["Seed", seed],
        ["Duração (ms)", duration_ms],
        ["Canceladas", 0],
        ["Hash Verificação", verification_hash],
        ["EV (média)", mean],
        ["P50", p50],
        ["P95", p95],
        ["P(≥ Meta)", prob_meta],
        ["P(≥ Provável)", prob_ge_B],
        ["P(≥ Piso)", prob_ge_A],
    ], columns=["Parâmetro", "Valor"])

    # compactar CSV em memória
    csv_buf = io.StringIO()
    df_meta.to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Baixar CSV (metadados + estatísticas)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"resultado_montecarlo_{int(time.time())}.csv",
        mime="text/csv"
    )

else:
    st.info("Defina A, B, C e clique em **Rodar simulação** (mínimo de 10.000 iterações).")
