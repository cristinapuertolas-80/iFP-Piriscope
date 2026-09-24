# Piriscope
Proyecto para construir sistema predictivo basado en aprendizaje supervisado que permita estimar diariamente el riesgo de incendio forestal por municipio.

![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-orange)
![Fase](https://img.shields.io/badge/Fase-0-lightgrey)

## 🎯 Objetivo
 
Estimar diariamente el nivel de riesgo de incendio forestal por municipio, evaluando el rendimiento del modelo frente a un baseline (índice FWI) y proporcionando una visualización explicable de los resultados, con el fin de apoyar la priorización de recursos de vigilancia y prevención forestal.
 
Éxito mínimo: generar una predicción diaria de riesgo para al menos el 80% de los municipios gallegos (región piloto), utilizando exclusivamente fuentes públicas de datos (AEMET, EFFIS).

## 👥 Miembros
 
- Cristina Puértolas Rebollar
- Hikari Lheku Barrio Martín
- Guillermo Ramón Serrano
 
Reparto de tareas detallado en [Presentación y viabilidad](docs/NF1_Presentacion_y_Viabilidad.pdf).

## 📄 Documentación

- [Presentación y viabilidad](docs/NF1_Presentacion_y_Viabilidad.pdf)
- [Guía de contribución](CONTRIBUTING.md)
- [Decisiones de diseño](docs/decisiones.md)

## 📁 Estructura del repositorio
```
├── docs/          #Documentación del proyecto
├── src/           #Código fuente
├── data/          #Datos (no versionados)
└── environment/   #Configuración del entorno
```