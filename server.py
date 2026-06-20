import os
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import shutil

from federated import FederatedOrchestrator
from synthetic_data import generate_mri_slice

app = FastAPI(title="Privacy-Preserving Federated Learning Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the orchestrator
orchestrator = FederatedOrchestrator()
loop_ref = None

# WebSocket Manager to handle multiple active browser sessions
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection might be closed already
                pass

manager = ConnectionManager()

# Thread-safe callback to push updates from background thread to asyncio websockets
def fl_update_callback(event: dict):
    global loop_ref
    if loop_ref is not None:
        asyncio.run_coroutine_threadsafe(manager.broadcast(event), loop_ref)

@app.on_event("startup")
async def startup_event():
    global loop_ref
    loop_ref = asyncio.get_running_loop()
    
    # Ensure static and temp directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("temp_inference", exist_ok=True)
    
    # Create default dataset if none exists
    if not os.path.exists("data"):
        print("Data directory not found. Pre-generating synthetic datasets...")
        orchestrator.generate_data()

# Request Models
class FLParams(BaseModel):
    num_rounds: Optional[int] = None
    local_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    dp_enabled: Optional[bool] = None
    dp_noise_multiplier: Optional[float] = None
    dp_clip_norm: Optional[float] = None
    data_distribution: Optional[str] = None

# REST Endpoints
@app.get("/api/config")
def get_config():
    return orchestrator.params

@app.post("/api/config")
def update_config(params: FLParams):
    updates = {k: v for k, v in params.model_dump().items() if v is not None}
    orchestrator.update_params(updates)
    return {"status": "success", "params": orchestrator.params}

@app.get("/api/status")
def get_status():
    clients_meta = {}
    if os.path.exists(orchestrator.data_dir):
        for client in orchestrator.clients:
            clients_meta[client] = orchestrator._get_client_distribution(client)
            
    return {
        "status": orchestrator.status,
        "current_round": orchestrator.current_round,
        "params": orchestrator.params,
        "clients": clients_meta
    }

@app.post("/api/generate-data")
def generate_data(background_tasks: BackgroundTasks):
    if orchestrator.status == "training":
        raise HTTPException(status_code=400, detail="Cannot generate data while training is in progress.")
        
    def do_generate():
        orchestrator.generate_data()
        # Broadcast completed generation status
        clients_meta = {}
        for client in orchestrator.clients:
            clients_meta[client] = orchestrator._get_client_distribution(client)
        fl_update_callback({
            "type": "data_generated",
            "clients": clients_meta,
            "message": f"New synthetic dataset generated (mode: {orchestrator.params['data_distribution']})"
        })
        
    background_tasks.add_task(do_generate)
    return {"status": "started", "message": "Data generation started in background."}

@app.post("/api/start-training")
def start_training():
    if orchestrator.status == "training":
        return {"status": "already_running", "message": "Training is already in progress."}
        
    success = orchestrator.start_training(fl_update_callback)
    if success:
        return {"status": "started", "message": "Federated training loop initialized."}
    else:
        raise HTTPException(status_code=500, detail="Failed to start training thread.")

@app.post("/api/stop-training")
def stop_training():
    success = orchestrator.stop_training()
    if success:
        return {"status": "stopped", "message": "Halt instruction sent to training loop."}
    else:
        return {"status": "not_running", "message": "No active training process found."}

@app.get("/api/samples")
def get_samples():
    """
    Returns list of paths of test MRI samples so the UI can easily select them.
    Also returns client train folders for visualization.
    """
    samples = {"tumor": [], "normal": []}
    
    test_dir = os.path.join(orchestrator.data_dir, "test")
    for category in ["normal", "tumor"]:
        cat_dir = os.path.join(test_dir, category)
        if os.path.exists(cat_dir):
            files = [f for f in os.listdir(cat_dir) if f.endswith('.png')]
            # Select up to 10 files
            for f in sorted(files)[:10]:
                samples[category].append(f"/api/image?path=test/{category}/{f}")
                
    return samples

@app.get("/api/image")
def get_image(path: str):
    """
    Serves a specific image from the data folder or temp folder.
    """
    # Sanitize path to prevent directory traversal
    clean_path = path.replace("..", "")
    full_path = os.path.join(orchestrator.data_dir, clean_path)
    
    # Check in temp folder if path starts with temp_inference
    if clean_path.startswith("temp_inference"):
        full_path = clean_path
        
    if os.path.exists(full_path):
        return FileResponse(full_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found.")

@app.post("/api/generate-random-mri")
def generate_random_mri(has_tumor: bool):
    """
    Generates a new random MRI slice on the fly and saves it to temp_inference, returning path.
    """
    img = generate_mri_slice(has_tumor=has_tumor)
    filename = f"temp_inference/random_{'tumor' if has_tumor else 'normal'}_{uuid.uuid4().hex[:8]}.png"
    img.save(filename)
    return {"path": filename, "url": f"/api/image?path={filename}"}

@app.post("/api/predict-image")
def predict_image(image_path: str):
    """
    Runs global model inference on an image from the provided path.
    """
    clean_path = image_path.replace("..", "")
    full_path = os.path.join(orchestrator.data_dir, clean_path)
    
    # Check in temp_inference
    if clean_path.startswith("temp_inference"):
        full_path = clean_path
        
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File does not exist.")
        
    result = orchestrator.run_single_inference(full_path)
    return result

@app.post("/api/upload-inference")
async def upload_inference(file: UploadFile = File(...)):
    """
    Handles user uploading their own image for inference.
    """
    filename = f"temp_inference/uploaded_{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    result = orchestrator.run_single_inference(filename)
    result["url"] = f"/api/image?path={filename}"
    return result

# WebSocket endpoint
@websocket_route := app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current status upon connection
        clients_meta = {}
        if os.path.exists(orchestrator.data_dir):
            for client in orchestrator.clients:
                clients_meta[client] = orchestrator._get_client_distribution(client)
                
        await websocket.send_json({
            "type": "status",
            "status": orchestrator.status,
            "current_round": orchestrator.current_round,
            "params": orchestrator.params,
            "clients": clients_meta,
            "metrics_history": orchestrator.metrics_history
        })
        
        while True:
            # We just wait for incoming client messages if any (optional heartbeat)
            data = await websocket.receive_text()
            # Do nothing or handle commands
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.on_event("shutdown")
def shutdown_event():
    # Cleanup temp folder
    if os.path.exists("temp_inference"):
        shutil.rmtree("temp_inference")
