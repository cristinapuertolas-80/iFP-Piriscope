import pandas as pd

pd.set_option("display.width", 200)
p = pd.read_parquet("data/gold/predicciones.parquet", columns=["fecha_objetivo", "version_modelo", "nivel_riesgo"])
p["fecha_objetivo"] = pd.to_datetime(p["fecha_objetivo"])
p = p[p.fecha_objetivo >= "2026-07-01"]
p["medio_alto"] = p.nivel_riesgo.isin(["medio", "alto"])
# % de predicciones en medio o alto, por semana y modelo
semana = p.groupby([pd.Grouper(key="fecha_objetivo", freq="W"), "version_modelo"])["medio_alto"].mean().mul(100).round(1)
print(semana.unstack())