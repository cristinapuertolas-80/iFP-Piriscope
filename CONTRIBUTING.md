# Guía de contribución

## Reglas base
- Toda tarea tiene una issue asociada.
- No se realiza **merge** sin **PR** aprobado.
- Documentar decisiones en **"docs/decisiones.md"**.

## Ramas
- `main`: nunca se commitea directo.
- Todo cambio se realiza con un **Pull Request**.

## Pull Requests

### Nomenclatura

Los cambios de realizan en ramas con la nomenclatura **tipo/descripción**.

#### Tipos

- `fix`: arreglo de bugs.
- `feat`: añadir features.
- `build`: actualizaciones que afectan el sistema/proceso de build.
- `chore`: commits misceláneos que no afectan al código (formateo, espacios en blanco, typos en código, comentarios, etc.).
- `docs`: cambios a los archivos de documentación.
- `test`: añadir o arreglar tests existentes.
- `refactor`: reestructuración de código sin cambiar funcionalidad.
- `ci`: cambios en configuración de integración continua (CI) (GitHub Actions, etc.).
- `localize`: cambios relacionados con la traducción y localización.
- `bump`: actualizar versiones de dependencia.
- `revert`: revertir commits.
- `data`: relacionado a datasets o muestras.

**Ejemplo**: `docs/setup-repo`

## Commits
- Mensajes en **inglés** con una descripción breve.