# app.py — Exxata Monte Carlo (Triangular A/B/C) com faixas (até 8), PDF e verificação de hash
import io, time, json, hashlib
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ----------------- Identidade visual Exxata -----------------
st.set_page_config(page_title="Exxata | Simulação Monte Carlo", layout="wide", page_icon="📈")
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --exxata-blue:#4284D7; --exxata-red:#D51D07; --exxata-gray:#B2B2BB; --exxata-slate:#78909C;
}
html, body, [class*="css"]  { font-family: 'Manrope', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important; }
h1,h2,h3,h4 { color: var(--exxata-blue) !important; letter-spacing: .2px;}
.kpi .stMetricValue { color: var(--exxata-red) !important; font-weight:800 !important; }
.kpi .stMetricLabel { color: var(--exxata-slate) !important; }
.stButton>button { background: var(--exxata-red); border:0; }
.block { background:#fff; border:1px solid #E5E7EB; border-radius:16px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,.05); }
.pill { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; background:#EEF2FF; color:#1E293B; border:1px solid #E5E7EB; margin:4px 6px 0 0; font-size:13px;}
.pill small{color:#64748B}
.hint{background:#EEF6FF; border:1px solid #E0ECFF; padding:14px; border-radius:12px;}
.codebox{font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; padding:10px; border:1px dashed #CBD5E1; border-radius:8px; background:#F8FAFC;}
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
    return f"R$ {x:,.0f}".replace(",", ".")  # valores grandes legíveis

def make_pdf(kpis, faixas_rows, meta, hist_png, cdf_png):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=24, bottomMargin=24, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    title = styles["Heading1"]; title.textColor = colors.HexColor("#4284D7")
    h2 = styles["Heading2"]; h2.textColor = colors.HexColor("#4284D7")
    p = styles["BodyText"]
    story = []
    story.append(Paragraph("Exxata – Simulação de Monte Carlo (Triangular A/B/C)", title))
    story.append(Paragraph(f"Item: <b>{meta['item']}</b>", p)); story.append(Spacer(1, 10))
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
    table_rows = [["Faixa","%"]] + [[lbl, f"{100*pct:.2f}%"] for lbl,pct in faixas_rows]
    tf = Table(table_rows, hAlign="LEFT", colWidths=[300, 80])
    tf.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#B2B2BB")),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#78909C"))]))
    story.append(tf); story.append(Spacer(1, 12))
    # Gráficos
    if hist_png: story.append(Paragraph("Histograma", h2)); story.append(Image(hist_png, width=480, height=260)); story.append(Spacer(1,6))
    if cdf_png:  story.append(Paragraph("Curva Acumulada (CDF)", h2)); story.append(Image(cdf_png, width=480, height=260)); story.append(Spacer(1,12))
    # Auditoria
    story.append(Paragraph("Auditoria & Comprovação", h2))
    story.append(Paragraph(
        f"Foram realizadas <b>{meta['iterations']:,}</b> simulações em {meta['duration_ms']} ms. "
        f"Piso={brl(meta['piso'])}; Provável={brl(meta['provavel'])}; Teto={brl(meta['teto'])}. "
        f"Hash (SHA-256 do JSON abaixo): <font face='Courier'>{meta['hash']}</font>.", p))
    doc.build(story); buf.seek(0); return buf

def compute_hash_signature(item, piso, provavel, teto, iterations, seed, duration_ms):
    signature = {
        "item": item,
        "piso": float(piso), "provavel": float(provavel), "teto": float(teto),
        "iterations": int(iterations), "seed": int(seed), "duration_ms": int(duration_ms)
    }
    raw = json.dumps(signature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest().upper()
    return signature, raw.decode("utf-8"), digest

def faixas_from_limits(vals_sorted: np.ndarray, limits: list[float]):
    limits = sorted(set([float(x) for x in limits if x is not None]))
    # Faixas: abaixo do 1º, entre vizinhos e acima do último
    edges = [-np.inf] + limits + [np.inf]
    rows = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        if i < len(edges)-2:
            mask = (vals_sorted >= lo) & (vals_sorted < hi)
            label = f"Entre {brl(lo if np.isfinite(lo) else limits[0])} e {brl(hi)}" if np.isfinite(lo) else f"Abaixo de {brl(hi)}"
        else:
            mask = (vals_sorted >= lo)
            label = f"Acima de {brl(limits[-1])}"
        rows.append((label, float(mask.mean())))
    return rows

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
    n_limits = st.slider("Quantos limites deseja usar?", 1, 8, 3)
    limits = []
    last_min = 1_000_000.0
    for i in range(n_limits):
        if i == 0:
            val = st.number_input(f"Limite {i+1} (R$)", min_value=0.0, value=2_000_000.0, step=100_000.0, key=f"lim{i}")
        else:
            val = st.number_input(f"Limite {i+1} (R$)", min_value=float(limits[-1]), value=float(limits[-1] + 1_000_000.0), step=100_000.0, key=f"lim{i}")
        limits.append(val)

    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        auto_q = st.button("Quantis (P20,P40,P60,P80)", use_container_width=True)
    with col_auto2:
        auto_contract = st.button("Metas (2MM,3MM,4MM...)", use_container_width=True)

    rodar = st.button("🚀 Rodar simulação", use_container_width=True)

# ----------------- Execução -----------------
if rodar:
    start = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    valores = rng.triangular(left=piso, mode=provavel, right=teto, size=int(iters))
    sorted_vals = np.sort(valores); n = len(sorted_vals)
    mean = float(np.mean(sorted_vals)); p50 = float(np.quantile(sorted_vals, 0.5)); p95 = float(np.quantile(sorted_vals, 0.95))
    duration_ms = int((time.perf_counter() - start)*1000)

    # Auto-preenchimento de limites se usuário clicou
    if auto_q:
        qs = [0.2,0.4,0.6,0.8]
        qvals = [float(np.quantile(sorted_vals, q)) for q in qs][:n_limits]
        limits = sorted(qvals)
        st.success("Limites preenchidos pelos quantis.")
    if auto_contract:
        base = 2_000_000.0
        limits = [base + i*1_000_000.0 for i in range(n_limits)]
        st.success("Limites preenchidos por metas contratuais (2MM, 3MM, ...).")

    # Faixas
    faixas_rows = faixas_from_limits(sorted_vals, limits)

    # KPIs
    k1,k2,k3,k4 = st.columns(4, gap="medium")
    with k1: st.metric("EV (média)", brl(mean), label_visibility="visible", help="Valor esperado")
    with k2: st.metric("P50 (mediana)", brl(p50), help="50% dos resultados são menores que este valor")
    with k3: st.metric("P95 (alto)", brl(p95), help="95% dos resultados são menores que este valor")
    with k4: st.metric("Simulações", f"{n:,}".replace(",", "."))

    # Gráficos (e buffers para PDF)
    left,right = st.columns(2)
    with left:
        st.subheader("📊 Distribuição (Histograma)")
        fig, ax = plt.subplots()
        ax.hist(sorted_vals, bins=25, color="#D51D07", edgecolor="#B2B2BB")
        ax.set_xlabel("Valor (R$)"); ax.set_ylabel("Frequência")
        buf_hist = io.BytesIO(); fig.savefig(buf_hist, format="png", bbox_inches="tight", dpi=160); buf_hist.seek(0)
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("📈 Curva Acumulada (CDF)")
        y = np.linspace(0,1,n); fig2, ax2 = plt.subplots()
        ax2.plot(sorted_vals, y, color="#4284D7"); ax2.set_xlabel("Valor (R$)"); ax2.set_ylabel("Probabilidade acumulada")
        buf_cdf = io.BytesIO(); fig2.savefig(buf_cdf, format="png", bbox_inches="tight", dpi=160); buf_cdf.seek(0)
        st.pyplot(fig2, clear_figure=True)

    # Pré-visualização das faixas
    st.markdown("### 🎯 Distribuição por Faixa de Acordo")
    preview = " ".join([f"<span class='pill'><b>{i+1}</b> <small>{brl(limits[i])}</small></span>" for i in range(len(limits))])
    st.markdown(preview, unsafe_allow_html=True)
    for lbl, pct in faixas_rows:
        st.write(f"**{pct*100:.2f}%** — {lbl}")

    # Auditoria & Assinatura
    st.markdown("### 🧾 Auditoria & Assinatura do Experimento")
    signature, signature_json, verification_hash = compute_hash_signature(item, piso, provavel, teto, n, seed, duration_ms)
    st.write(f"Foram realizadas **{n:,} simulações** em **{duration_ms} ms**.")
    st.write("**Como validar externamente?** Calcule **SHA-256** desta string JSON (UTF-8) em qualquer site de hash. O resultado deve ser igual ao código abaixo.")
    st.markdown(f"<div class='codebox'>{signature_json}</div>", unsafe_allow_html=True)
    st.write("**Hash SHA-256:**")
    st.code(verification_hash, language="text")

    # Verificador interno
    st.markdown("#### Verificar um JSON manualmente")
    test_json = st.text_area("Cole aqui o JSON para validar", value=signature_json, height=120)
    if st.button("Validar JSON → SHA-256"):
        try:
            raw = test_json.encode("utf-8")
            st.write("Hash calculado:")
            st.code(hashlib.sha256(raw).hexdigest().upper())
        except Exception as e:
            st.error(f"JSON inválido: {e}")

    # Exportações
    df = pd.DataFrame(
        [["Item", item], ["Piso (A)", piso], ["Provável (B)", provavel], ["Teto (C)", teto],
         ["Iterações", n], ["Seed", int(seed)], ["Duração (ms)", duration_ms],
         ["Hash", verification_hash], ["P50", p50], ["P95", p95], ["EV", mean]]
        + [[lbl, pct] for lbl, pct in faixas_rows],
        columns=["Parâmetro","Valor"]
    )
    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
    st.download_button("📥 Baixar CSV", csv_buf.getvalue().encode("utf-8"),
                       file_name=f"exxata_montecarlo_{int(time.time())}.csv", mime="text/csv")

    pdf = make_pdf({"mean":mean, "p50":p50, "p95":p95}, faixas_rows,
                   {**signature, "hash":verification_hash}, buf_hist, buf_cdf)
    st.download_button("🧾 Baixar Relatório PDF", data=pdf.getvalue(),
                       file_name=f"exxata_montecarlo_{int(time.time())}.pdf", mime="application/pdf")

else:
    st.info("Defina **Piso (A)**, **Provável (B)** e **Teto (C)**, ajuste as **faixas (1–8)** e clique em **Rodar simulação**.")

