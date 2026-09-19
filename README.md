# Piriscope
Proyecto para construir sistema predictivo basado en aprendizaje supervisado que permita estimar diariamente el riesgo de incendio forestal por municipio.

## Objetivo
 
Estimar diariamente el nivel de riesgo de incendio forestal por municipio, evaluando el rendimiento del modelo frente a un baseline (índice FWI) y proporcionando una visualización explicable de los resultados, con el fin de apoyar la priorización de recursos de vigilancia y prevención forestal.
 
Éxito mínimo: generar una predicción diaria de riesgo para al menos el 80% de los municipios gallegos (región piloto), utilizando exclusivamente fuentes públicas de datos (AEMET, EFFIS).

## Miembros
 
Cristina Puértolas Rebollar
Hikari Lheku Barrio Martín
 
Reparto de tareas detallado en `docs/NF1_Presentacion_y_Viabilidad.pdf`.

## Organización
 
- Todo cambio se desarrolla en una rama propia y se integra mediante **Pull Request (PR)**.
- Cada tarea tiene una **issue** asignada antes de empezar a trabajar en ella.
- Las decisiones de diseño relevantes (elección de modelo, fuentes de datos, cambios de alcance, etc.) se documentan en `docs/decisiones.md`, con fecha, decisión y motivo por entrada.
- Se respeta el reparto de tareas, pero se presta apoyo entre miembros si hay bloqueos.
