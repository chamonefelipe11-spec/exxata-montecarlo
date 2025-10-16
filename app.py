# app.py — Exxata Monte Carlo (Triangular A/B/C) com faixas (até 8) e PDF
import io, time, json, hashlib
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ----------------- Identidade visual Exxata -----------------
st.set_page_config(page_title="Exxata | Simulação Monte Carlo", layout="wide", page_icon="📈")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"]  { font-family: 'Manrope', sans-serif !important; }
h1,h2,h3,h4 { color:#4284D7 !important; }
div[data-testid="stMetricValue"] { color:#D51D07 !important; font-weight:800; }
div[data-testid="stMetricLabel"] { color:#78909C !important; }
button[kind="primary"] { background:#D51D07; }
</style>
""", unsafe_allow_html=True)

# ----------------- Cabeçalho -----------------
st.title("📈 Simulação de Monte Carlo – Pleitos/Negociações (Triangular A/B/C)")
st.caption(
    "Use **A (Piso)** = menor valor possível; **B (Provável)** = valor mediano; **C (Teto)** = maior valor possível. "
    "A simulação usa **Triangular(A,B,C)**. Execute **10.000+** iterações."
)

# ----------------- Helpers -----------------
def brl(x: float) -> str:
    return f"R$ {x:,.0f}".replace(",", ".")  # números grandes mais legíveis

def parse_limits(text: str, max_limits: int = 8):
    # Extrai números, ordena, remove duplicados, corta no máximo permitido
    if not text.strip():
        return []
    parts = [p.strip().replace(" ", "") for p in text.split(",")]
    vals = []
    for p in parts:
        p = p.replace("_","")
        if p:
            try:
                vals.append(float(p))
            except:
                pass
    vals = sorted(set(vals))
    return vals[:max_limits]

def bucket_percents(sorted_vals: np.ndarray, limits: list[float]):
    # Dado vetor ordenado e lista de limites (k), retorna k+1 faixas com percentuais
    if len(sorted_vals)==0:
        return []
    limits = sorted(limits)
    edges = [-np.inf] + limits + [np.inf]
    rows = []
    for i in range(len(edges)-1):
        low, high = edges[i], edges[i+1]
        mask = (sorted_vals >= low) & (sorted_vals < high) if i < len(edges)-2 else (sorted_vals >= low)
        pct = float(mask.mean())
        label = (f"Abaixo de {brl(limits[0])}" if i==0 else
                 (f"Entre {brl(limits[i-1])} e {brl(limits[i])}" if i < len(limits) else
                  f"Acima de {brl(limits[-1])}"))
        rows.append((label, pct))
    return rows

def make_pdf(kpis, rows_faixas, meta, hist_png, cdf_png):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=24, bottomMargin=24, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    title = styles["Heading1"]; title.textColor = colors.HexColor("#4284D7")
    h2 = styles["Heading2"]; h2.textColor = colors.HexColor("#4284D7")
    p = styles["BodyText"]

    story = []
    story.append(Paragraph("Exxata – Simulação de Monte Carlo (Triangular A/B/C)", title))
    story.append(Paragraph(f"Item: <b>{meta['item']}</b>", p))
    story.append(Spacer(1, 10))

    # KPIs
    story.append(Paragraph("Resultados Principais", h2))
    data = [["EV (média)", brl(kpis["mean"])],
            ["P50 (mediana)", brl(kpis["p50"])],
            ["P95 (cenário alto)", brl(kpis["p95"])],
            ["Iterações", f"{meta['iterations']:,}".replace(",", ".")],
            ["Seed", str(meta["seed"])],
            ["Duração (ms)", str(meta["duration_ms"])]]
    t = Table(data, hAlign="LEFT"); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#B2B2BB"))]))
    story.append(t); story.append(Spacer(1, 12))

    # Faixas
    story.append(Paragraph("Distribuição por Faixa de Acordo", h2))
    table_rows = [["Faixa","%"]] + [[lbl, f"{100*pct:.2f}%"] for lbl,pct in rows_faixas]
    tf = Table(table_rows, hAlign="LEFT", colWidths=[300, 80])
    tf.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#B2B2BB")),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#78909C"))]))
    story.append(tf); story.append(Spacer(1, 12))

    # Gráficos
    if hist_png:
        story.append(Paragraph("Histograma", h2))
        story.append(Image(hist_png, width=480, height=260)); story.append(Spacer(1, 6))
    if cdf_png:
        story.append(Paragraph("Curva Acumulada (CDF)", h2))
        story.append(Image(cdf_png, width=480, height=260)); story.append(Spacer(1, 12))

    # Auditoria
    story.append(Paragraph("Auditoria & Comprovação", h2))
    story.append(Paragraph(
        f"Foram realizadas <b>{meta['iterations']:,}</b> simulações em {meta['duration_ms']} ms. "
        f"Piso={brl(meta['piso'])}; Provável={brl(meta['provavel'])}; Teto={brl(meta['teto'])}. "
        f"Hash: <font face='Courier'>{meta['hash']}</font>.", p
    ))

    doc.build(story)
    buf.seek(0)
    return buf

# ----------------- Entradas -----------------
with st.sidebar:
    st.header("Premissas do Cenário")
    item = st.text_input("Item (pleito/negociação)", "Negociação A")

    piso = st.number_input("A — Piso (R$)", min_value=0.0, value=2_000_000.0, step=100_000.0, format="%.2f")
    default_B = max(2_500_000.0, piso)
    provavel = st.number_input("B — Provável (R$)", min_value=float(piso), value=default_B, step=100_000.0, format="%.2f")
    default_C = max(3_500_000.0, provavel)
    teto = st.number_input("C — Teto (R$)", min_value=float(provavel), value=default_C, step=100_000.0, format="%.2f")

    iters = st.number_input("Iterações (≥10.000)", min_value=10_000, value=20_000, step=1_000)
    seed = st.number_input("Seed", value=20251015, step=1)

    st.markdown("---")
    st.header("Faixas de Acordo")
    modo_faixas = st.radio("Modo", ["Simples (3 limites)", "Avançado (até 8 limites)"], horizontal=False)
    if modo_faixas.startswith("Simples"):
        f1 = st.number_input("Limite 1 (R$)", min_value=0.0, value=2_000_000.0, step=100_000.0)
        f2 = st.number_input("Limite 2 (R$)", min_value=f1, value=3_000_000.0, step=100_000.0)
        f3 = st.number_input("Limite 3 (R$)", min_value=f2, value=4_000_000.0, step=100_000.0)
        limites_texto = f"{int(f1)},{int(f2)},{int(f3)}"
    else:
        limites_texto = st.text_input("Limites (R$) separados por vírgula (máx. 8)",
                                      value="2_000_000, 3_000_000, 4_000_000")
    rodar = st.button("🚀 Rodar simulação", use_container_width=True)

# ----------------- Execução -----------------
if rodar:
    start = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    valores = rng.triangular(left=piso, mode=provavel, right=teto, size=int(iters))
    sorted_vals = np.sort(valores)
    n = len(sorted_vals)
    mean = float(np.mean(sorted_vals))
    p50 = float(np.quantile(sorted_vals, 0.5))
    p95 = float(np.quantile(sorted_vals, 0.95))
    duration_ms = int((time.perf_counter() - start)*1000)

    # Limites / Faixas (até 8)
    limites = parse_limits(limites_texto, max_limits=8)
    faixas = bucket_percents(sorted_vals, limites)

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("EV (média)", brl(mean))
    c2.metric("P50 (mediana)", brl(p50))
    c3.metric("P95 (cenário alto)", brl(p95))
    c4.metric("Simulações", f"{n:,}".replace(",", "."))

    st.caption("• **P50**: 50% dos resultados são menores que este valor.  • **P95**: 95% dos resultados são menores (limite superior provável).")

    # Gráficos
    left,right = st.columns(2)
    with left:
        st.subheader("📊 Distribuição (Histograma)")
        fig, ax = plt.subplots()
        ax.hist(sorted_vals, bins=25, color="#D51D07", edgecolor="#B2B2BB")
        ax.set_xlabel("Valor (R$)"); ax.set_ylabel("Frequência")
        # exportável
        buf_hist = io.BytesIO(); fig.savefig(buf_hist, format="png", bbox_inches="tight", dpi=160); buf_hist.seek(0)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("📈 Curva Acumulada (CDF)")
        y = np.linspace(0,1,n)
        fig2, ax2 = plt.subplots()
        ax2.plot(sorted_vals, y, color="#4284D7")
        ax2.set_xlabel("Valor (R$)"); ax2.set_ylabel("Probabilidade acumulada")
        buf_cdf = io.BytesIO(); fig2.savefig(buf_cdf, format="png", bbox_inches="tight", dpi=160); buf_cdf.seek(0)
        st.pyplot(fig2, clear_figure=True)

    # Faixas (até 8) – cards enxutos
    st.markdown("### 🎯 Distribuição por Faixa de Acordo")
    for lbl, pct in faixas:
        st.write(f"**{pct*100:.2f}%** — {lbl}")

    # Sugestões de limites (intuitivo)
    st.info(
        "💡 **Como escolher os limites?**\n"
        "- **Quantis**: após rodar, use pontos como P20, P40, P60, P80 (divide em blocos equiprováveis). "
        "Sugestão para este cenário: "
        f"{brl(float(np.quantile(sorted_vals,0.2)))}, {brl(float(np.quantile(sorted_vals,0.4)))}, "
        f"{brl(float(np.quantile(sorted_vals,0.6)))}, {brl(float(np.quantile(sorted_vals,0.8)))}.\n"
        "- **Metas contratuais**: use patamares relevantes (ex.: {brl(2_000_000)}, {brl(3_000_000)}, {brl(4_000_000)}, ...)."
    )

    # Auditoria
    st.markdown("### 🧾 Auditoria & Comprovação")
    meta = dict(
        item=item, piso=piso, provavel=provavel, teto=teto,
        iterations=n, seed=int(seed), duration_ms=duration_ms
    )
    hash_str = json.dumps(meta, ensure_ascii=False).encode()
    verification_hash = hashlib.sha256(hash_str).hexdigest().upper()
    meta["hash"] = verification_hash
    st.write(f"Foram realizadas **{n:,} simulações** em **{duration_ms} ms**. "
             "O **hash** é um *carimbo digital* que muda se qualquer premissa mudar.")
    st.code(verification_hash, language="text")

    # Export CSV
    df = pd.DataFrame([
        ["Item", item], ["Piso (A)", piso], ["Provável (B)", provavel], ["Teto (C)", teto],
        ["Iterações", n], ["Seed", int(seed)], ["Duração (ms)", duration_ms],
        ["Hash", verification_hash], ["P50", p50], ["P95", p95], ["EV", mean],
    ] + [[lbl, pct] for lbl, pct in faixas], columns=["Parâmetro","Valor"])
    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
    st.download_button("📥 Baixar CSV", csv_buf.getvalue().encode("utf-8"),
                       file_name=f"exxata_montecarlo_{int(time.time())}.csv", mime="text/csv")

    # Export PDF
    pdf = make_pdf(
        kpis={"mean":mean,"p50":p50,"p95":p95},
        rows_faixas=faixas, meta=meta,
        hist_png=buf_hist, cdf_png=buf_cdf
    )
    st.download_button("🧾 Baixar Relatório PDF", data=pdf.getvalue(),
                       file_name=f"exxata_montecarlo_{int(time.time())}.pdf", mime="application/pdf")

else:
    st.info("Defina **Piso (A)**, **Provável (B)** e **Teto (C)**, escolha as **faixas**, e clique em **Rodar simulação**.")


