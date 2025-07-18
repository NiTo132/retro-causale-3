"""
🌐 API REST pour Quantum Retro-Causal Engine
Extension enterprise pour intégration externe
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import uvicorn

app = FastAPI(
    title="Quantum Retro-Causal Engine API",
    description="API enterprise pour moteur quantique de trading",
    version="4.3.0"
)

# Models Pydantic
class QuantumStateModel(BaseModel):
    spatial: List[float]
    temporal: float
    probabilistic: List[float]
    complexity: float
    emergence_potential: float
    
class TradingSignalRequest(BaseModel):
    market_data: List[Dict[str, float]]
    mode: str = "normal"  # fast, normal, deep
    confidence_threshold: Optional[float] = None
    
class TradingSignalResponse(BaseModel):
    signal: str  # BUY, SELL, HOLD
    confidence: float
    strength: float
    quantum_metrics: Dict[str, float]
    processing_time: float
    futures_analyzed: int

# Instance globale du moteur (à initialiser au startup)
quantum_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialisation du moteur au démarrage"""
    global quantum_engine
    # Initialiser votre EnhancedTradingSystem ici
    print("🚀 Quantum Engine API démarrée")

@app.get("/health")
async def health_check():
    """Vérification de santé du système"""
    return {
        "status": "healthy",
        "version": "4.3.0",
        "engine_active": quantum_engine is not None
    }

@app.post("/api/v1/trading/signal", response_model=TradingSignalResponse)
async def generate_trading_signal(request: TradingSignalRequest):
    """Génération d'un signal de trading quantique"""
    if quantum_engine is None:
        raise HTTPException(status_code=503, detail="Quantum engine not initialized")
    
    try:
        # Conversion des données de marché en état quantique
        market_state = quantum_engine.convert_market_to_quantum_state(request.market_data)
        
        # Génération du signal
        start_time = time.time()
        signal_result = quantum_engine.predict_market_evolution(
            market_state, 
            mode=request.mode,
            confidence_threshold=request.confidence_threshold
        )
        processing_time = time.time() - start_time
        
        return TradingSignalResponse(
            signal=signal_result["direction"],
            confidence=signal_result["confidence"],
            strength=signal_result["strength"],
            quantum_metrics={
                "resonance": signal_result.get("resonance", 0),
                "emergence": signal_result.get("emergence", 0),
                "causal_entropy": signal_result.get("causal_entropy", 0)
            },
            processing_time=processing_time,
            futures_analyzed=signal_result.get("futures_count", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")

@app.get("/api/v1/system/stats")
async def get_system_statistics():
    """Statistiques détaillées du système"""
    if quantum_engine is None:
        raise HTTPException(status_code=503, detail="Quantum engine not initialized")
    
    return {
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "active_threads": threading.active_count(),
        "engine_stats": quantum_engine.get_comprehensive_statistics(),
        "uptime_seconds": time.time() - startup_time
    }

@app.post("/api/v1/system/checkpoint")
async def create_checkpoint(background_tasks: BackgroundTasks):
    """Création d'un checkpoint système"""
    if quantum_engine is None:
        raise HTTPException(status_code=503, detail="Quantum engine not initialized")
    
    # Création en arrière-plan pour ne pas bloquer l'API
    background_tasks.add_task(quantum_engine.create_checkpoint, "api_manual")
    
    return {"message": "Checkpoint creation initiated", "timestamp": time.time()}

@app.get("/api/v1/visualization/quantum-field")
async def get_quantum_field_data():
    """Données pour visualisation du champ quantique"""
    if quantum_engine is None:
        raise HTTPException(status_code=503, detail="Quantum engine not initialized")
    
    # Récupération des dernières données de sélection
    last_selection = quantum_engine.get_last_selection_result()
    
    if not last_selection:
        raise HTTPException(status_code=404, detail="No recent quantum field data available")
    
    return {
        "resonances": last_selection["resonances"].tolist(),
        "coherences": last_selection["coherences"].tolist(), 
        "emergence_potentials": last_selection["emergence_potentials"].tolist(),
        "final_scores": last_selection["final_scores"].tolist(),
        "optimal_index": last_selection["optimal_index"],
        "selection_time": last_selection["selection_time"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "quantum_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Désactivé en production
        workers=1      # Une instance par worker pour partager l'état
    )