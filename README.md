# Wilson-Cowan + PINN

Investigación sobre el modelo neuronal de **Wilson-Cowan** y su aprendizaje
mediante **Physics-Informed Neural Networks (PINNs)**.

## Objetivo

Generar datos a partir del modelo de Wilson-Cowan, entrenar una PINN que
reproduzca/identifique su dinámica, y evaluar los resultados.

## Estructura

```
investigacion/
├── src/
│   ├── wilson_cowan/      # Modelo de Wilson-Cowan (ODEs + simulación)
│   ├── data/              # Generación y carga de datasets
│   ├── pinn/              # Red, pérdidas y entrenamiento de la PINN
│   └── utils/             # Configuración, semillas, plots
├── scripts/              # Puntos de entrada (generar datos, entrenar, evaluar)
├── configs/              # Configuraciones YAML
├── data/
│   ├── raw/              # Datos crudos
│   └── processed/        # Datasets listos para entrenar
├── notebooks/           # Exploración y figuras
├── results/
│   ├── figures/
│   └── models/          # Checkpoints
└── tests/
```

## Uso

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar datos
python scripts/generate_data.py --config configs/default.yaml

# 3. Entrenar la PINN
python scripts/train_pinn.py --config configs/default.yaml

# 4. Evaluar
python scripts/evaluate.py --checkpoint results/models/pinn.pt
```

## Estado

Estructura inicial (scaffold). Las implementaciones están marcadas con
`raise NotImplementedError` / `TODO`.
