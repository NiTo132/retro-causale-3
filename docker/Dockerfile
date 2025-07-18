# syntax=docker/dockerfile:1.4
# Dockerfile ultra-professionnel pour Quantum Retro-Causal Engine
# Techniques avancées: multi-stage, layer caching, security hardening, distroless

# ===== ARGUMENTS GLOBAUX =====
ARG PYTHON_VERSION=3.11
ARG PYTHON_IMAGE=python:${PYTHON_VERSION}-slim
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILDKIT_INLINE_CACHE=1

# Build arguments pour flexibilité
ARG BUILD_ENV=production
ARG ENABLE_CPP=true
ARG ENABLE_PROFILING=false
ARG TARGET_ARCH=x86_64

# ===== STAGE: Base Dependencies =====
FROM ${PYTHON_IMAGE} as base-deps

# Metadata labels selon les standards OCI
LABEL org.opencontainers.image.title="Quantum Retro-Causal Engine"
LABEL org.opencontainers.image.description="AI-powered quantum trading engine with C++ optimization"
LABEL org.opencontainers.image.vendor="Quantum Technologies"
LABEL org.opencontainers.image.authors="Quantum Dev Team"
LABEL org.opencontainers.image.source="https://github.com/quantum-engine"
LABEL org.opencontainers.image.documentation="https://docs.quantum-engine.io"
LABEL org.opencontainers.image.version="2.0.0"

# Configuration apt avancée
RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
rm -rf /var/lib/apt/lists/*
apt-get clean
EOF

# Installation de tini pour proper signal handling
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini /tini
RUN chmod +x /tini

# ===== STAGE: Build Tools =====
FROM base-deps as build-tools

ARG ENABLE_CPP
ARG TARGET_ARCH

# Installation sélective des outils de build
RUN <<EOF
apt-get update
if [ "$ENABLE_CPP" = "true" ]; then
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        ccache \
        libomp-dev \
        pkg-config \
        git \
        libpython${PYTHON_VERSION}-dev
fi
# Outils de build Python toujours nécessaires
apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev
rm -rf /var/lib/apt/lists/*
apt-get clean
EOF

# Configuration ccache pour accélérer les rebuilds
ENV CCACHE_DIR=/opt/ccache
ENV PATH="/usr/lib/ccache:$PATH"
RUN mkdir -p /opt/ccache && chmod 777 /opt/ccache

# ===== STAGE: Python Dependencies =====
FROM build-tools as python-deps

# Configuration pip optimisée
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copie des fichiers requirements avec cache mount
COPY requirements.txt requirements-dev.txt ./

# Installation avec mount cache pour pip
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/opt/ccache \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-deps -r requirements.txt

# Installation conditionnelle des dépendances de développement
ARG BUILD_ENV
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUILD_ENV" = "development" ]; then \
        pip install --no-deps -r requirements-dev.txt; \
    fi

# Installation pybind11 si C++ activé
ARG ENABLE_CPP
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$ENABLE_CPP" = "true" ]; then \
        pip install pybind11[global] numpy; \
    fi

# ===== STAGE: C++ Module Builder =====
FROM python-deps as cpp-builder

ARG ENABLE_CPP
ARG TARGET_ARCH
ARG BUILD_ENV

# Skip this stage if C++ disabled
RUN if [ "$ENABLE_CPP" != "true" ]; then exit 0; fi

WORKDIR /app

# Copie sélective pour optimiser le cache
COPY cpp/CMakeLists.txt cpp/
COPY cpp/*.cpp cpp/
COPY cpp/*.hpp cpp/ 2>/dev/null || true

# Configuration CMake avancée avec cache
RUN --mount=type=cache,target=/opt/ccache \
    --mount=type=cache,target=/app/cpp/build \
    if [ "$ENABLE_CPP" = "true" ]; then \
        cd cpp && \
        mkdir -p build && \
        cd build && \
        cmake .. \
            -GNinja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
            -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native -flto=auto -ffast-math" \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
        ninja -j$(nproc) && \
        # Test de base du module
        python3 -c "import sys; sys.path.insert(0, '.'); import quantum_selector; print('✅ C++ module OK')" || \
        (echo "❌ C++ module test failed" && exit 1); \
    fi

# ===== STAGE: Code Quality & Security Scan =====
FROM python-deps as code-quality

ARG BUILD_ENV
ARG ENABLE_PROFILING

# Installation des outils de qualité seulement si nécessaire
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUILD_ENV" = "development" ] || [ "$ENABLE_PROFILING" = "true" ]; then \
        pip install \
            bandit \
            safety \
            mypy \
            black \
            isort \
            flake8; \
    fi

WORKDIR /app
COPY src/ src/
COPY pyproject.toml setup.py 2>/dev/null ./

# Analyse de sécurité et qualité
RUN if [ "$BUILD_ENV" = "development" ]; then \
        echo "🔍 Security scan..." && \
        bandit -r src/ -f json -o /tmp/bandit-report.json || true && \
        echo "🔍 Dependency security check..." && \
        safety check --json --output /tmp/safety-report.json || true && \
        echo "✅ Quality checks completed"; \
    fi

# ===== STAGE: Application Builder =====
FROM python-deps as app-builder

WORKDIR /app

# Copie des sources avec .dockerignore optimisé
COPY --chown=1001:1001 . .

# Copie du module C++ compilé
COPY --from=cpp-builder /app/cpp/build/*.so ./cpp/build/ 2>/dev/null || true
COPY --from=cpp-builder /app/cpp/build/*.pyd ./cpp/build/ 2>/dev/null || true

# Installation du package en mode optimisé
ARG BUILD_ENV
RUN if [ "$BUILD_ENV" = "production" ]; then \
        pip install --no-deps -e .; \
    else \
        pip install -e .; \
    fi

# Création de la structure pour le module C++
RUN mkdir -p src/quantum_engine/native/cpp/build && \
    if [ -f cpp/build/quantum_selector.so ]; then \
        cp cpp/build/quantum_selector*.so src/quantum_engine/native/cpp/build/ || true; \
    fi

# ===== STAGE: Runtime Base =====
FROM ${PYTHON_IMAGE} as runtime-base

# Installation des runtime libraries minimales
RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends \
    libomp5 \
    libgomp1 \
    curl \
    netcat-traditional \
    procps
rm -rf /var/lib/apt/lists/*
apt-get clean
EOF

# Création utilisateur non-root avec UID/GID fixes
RUN groupadd --gid 1001 quantum && \
    useradd --uid 1001 --gid quantum --shell /bin/bash --create-home quantum

# Configuration des dossiers avec permissions appropriées
RUN mkdir -p \
    /app/{logs,checkpoints,backups,monitoring/data} \
    /opt/quantum \
    /var/log/quantum && \
    chown -R quantum:quantum \
    /app \
    /opt/quantum \
    /var/log/quantum

# ===== STAGE: Production Runtime =====
FROM runtime-base as production

# Copie de tini
COPY --from=base-deps /tini /tini

# Copie des dépendances Python
COPY --from=python-deps /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copie de l'application
COPY --from=app-builder --chown=quantum:quantum /app /app

# Configuration des variables d'environnement
ENV PYTHONPATH=/app \
    QUANTUM_ENV=production \
    QUANTUM_USER=quantum \
    QUANTUM_UID=1001 \
    QUANTUM_GID=1001 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=2 \
    PIP_NO_CACHE_DIR=1 \
    QUANTUM_CPP_ENABLED=1 \
    OMP_NUM_THREADS=4 \
    OMP_WAIT_POLICY=passive \
    OMP_PROC_BIND=true

# Configuration de l'utilisateur et permissions finales
USER quantum
WORKDIR /app

# Test final de l'installation
RUN python3 -c "
import sys
print(f'🐍 Python: {sys.version}')

try:
    from src.quantum_engine.core.state import QuantumState
    print('✅ Core modules imported successfully')
except Exception as e:
    print(f'❌ Core import error: {e}')
    sys.exit(1)

# Test module C++ si disponible
try:
    sys.path.insert(0, 'src/quantum_engine/native/cpp/build')
    import quantum_selector
    print('✅ C++ acceleration available')
except ImportError:
    print('⚠️ C++ acceleration not available (fallback to Python)')

print('🚀 Quantum Engine ready for production!')
"

# Ports exposés
EXPOSE 8000 9090

# Health check optimisé
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint avec tini pour proper signal handling
ENTRYPOINT ["/tini", "--", "/app/docker/entrypoint.sh"]
CMD ["uvicorn", "src.quantum_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ===== STAGE: Development Runtime =====
FROM runtime-base as development

# Installation d'outils de développement
RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends \
    vim \
    git \
    htop \
    tree \
    strace \
    tcpdump \
    iputils-ping
rm -rf /var/lib/apt/lists/*
EOF

# Copie des dépendances (incluant dev)
COPY --from=python-deps /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copie de l'application
COPY --from=app-builder --chown=quantum:quantum /app /app

# Variables d'environnement pour développement
ENV QUANTUM_ENV=development \
    DEBUG=true \
    PYTHONDONTWRITEBYTECODE=0 \
    PYTHONOPTIMIZE=0 \
    RELOAD_ON_CHANGE=true

USER quantum
WORKDIR /app

# Hot-reload par défaut en développement
CMD ["uvicorn", "src.quantum_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ===== STAGE: Distroless Production (Ultra-Sécurisé) =====
FROM gcr.io/distroless/python3-debian11 as distroless

# Copie des dépendances et de l'app depuis production
COPY --from=production /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=production --chown=1001:1001 /app /app

# Variables d'environnement minimales
ENV PYTHONPATH=/app \
    QUANTUM_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 1001
WORKDIR /app

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "uvicorn", "src.quantum_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ===== STAGE: Debug/Profiling =====
FROM production as debug

USER root

# Installation d'outils de profiling
RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends \
    gdb \
    valgrind \
    strace \
    perf-tools-unstable \
    linux-perf
rm -rf /var/lib/apt/lists/*
EOF

# Installation d'outils Python de profiling
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        py-spy \
        memory-profiler \
        line-profiler \
        pyflame \
        viztracer

USER quantum

ENV QUANTUM_PROFILING_ENABLED=true
CMD ["py-spy", "top", "--pid", "1"]