# Quantum Retro-Causal Engine

## Présentation

Moteur quantique rétro-causal pour la génération, la sélection et l’analyse de futurs massifs appliqués au trading, à l’optimisation et à la recherche scientifique. Architecture modulaire, extensible, hybride Python/C++.

## Structure du projet

```
quantum-retrocausal-engine/
│
├── src/
│   └── quantum_engine/
│       ├── state.py
│       ├── generator.py
│       ├── selector.py
│       ├── resonance.py
│       ├── adapter.py
│       ├── backtester.py
│       ├── checkpoint.py
│       ├── visualization.py
│       ├── logging_config.py
│       ├── configs.py
│       ├── demo.py
│       └── __init__.py
│
├── tests/
├── config/
├── docker/
├── monitoring/
├── logs/
├── checkpoints/
├── backups/
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── README.md
└── ...
```

## Installation

```bash
# Installer les dépendances principales
pip install -r requirements.txt
# Installer les dépendances de dev (optionnel)
pip install -r requirements-dev.txt
```

## Lancer la démo

```bash
python -m src.quantum_engine.demo
```

## Lancer les tests

```bash
pytest tests/
```

## Déploiement Docker

```bash
cd docker
# Build et run avec docker-compose
docker-compose up --build
```

## Configuration

- Modifier `config/config.yaml` pour ajuster les paramètres du moteur, du trading, de l’API, etc.
- Utiliser `.env.example` pour les variables d’environnement.

## Points forts
- Architecture modulaire, testable, extensible
- Prêt pour la production et la recherche
- Intégration possible de modules C++ pour la performance
- Backtesting, visualisation, monitoring intégrés

## Licence
Projet open source, tous droits réservés. 