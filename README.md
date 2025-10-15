# Monte Carlo – Pleitos/Negociações (Triangular A/B/C) – Streamlit

Simulação de resultado de pleitos/negociações usando **TRIANGULAR(A, B, C)** com **10k+ iterações**, gráficos (histograma e CDF), auditoria (n, seed, duração, hash) e **nenhuma simulação cancelada**.

## Como rodar localmente
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
