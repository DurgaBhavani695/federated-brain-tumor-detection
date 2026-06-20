import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import threading
import time

from model import BrainTumorCNN, MRIDataset, train_one_epoch, evaluate_model, apply_differential_privacy
from synthetic_data import setup_federated_dataset

class FederatedOrchestrator:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.clients = ["hospital_a", "hospital_b", "hospital_c"]
        self.global_model = BrainTumorCNN()
        
        # Default Parameters
        self.params = {
            "num_rounds": 10,
            "local_epochs": 3,
            "learning_rate": 0.005,
            "batch_size": 16,
            "dp_enabled": True,
            "dp_noise_multiplier": 0.1,
            "dp_clip_norm": 1.0,
            "data_distribution": "non_iid"  # "iid" or "non_iid"
        }
        
        # State tracking
        self.status = "idle"  # "idle", "training", "stopped"
        self.current_round = 0
        self.metrics_history = []
        
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self._training_thread = None

    def update_params(self, new_params):
        with self.lock:
            self.params.update(new_params)

    def generate_data(self):
        self.status = "generating_data"
        setup_federated_dataset(
            base_dir=self.data_dir,
            mode=self.params["data_distribution"],
            num_train_per_client=100,
            num_val_per_client=20,
            num_test=50
        )
        self.status = "idle"

    def stop_training(self):
        if self.status == "training":
            self.stop_event.set()
            self.status = "stopped"
            return True
        return False

    def start_training(self, on_update_callback):
        """
        Starts the FL training in a separate background thread if not already training.
        """
        with self.lock:
            if self.status == "training":
                return False
            
            # Reset state for a new run
            self.status = "training"
            self.current_round = 0
            self.metrics_history = []
            self.stop_event.clear()
            
            # Initialize fresh global model
            self.global_model = BrainTumorCNN()
            
            # Start background thread
            self._training_thread = threading.Thread(
                target=self._run_fl_loop,
                args=(on_update_callback,),
                daemon=True
            )
            self._training_thread.start()
            return True

    def _run_fl_loop(self, on_update_callback):
        """
        Core FL execution loop, run inside the background thread.
        """
        try:
            # 1. Prepare global test set
            test_dataset = MRIDataset(os.path.join(self.data_dir, "test"))
            if len(test_dataset) == 0:
                # If data does not exist, generate it first
                on_update_callback({"type": "info", "message": "No dataset found. Generating default dataset..."})
                self.generate_data()
                test_dataset = MRIDataset(os.path.join(self.data_dir, "test"))
                
            test_loader = DataLoader(test_dataset, batch_size=self.params["batch_size"], shuffle=False)
            
            # Send initial state
            initial_eval = evaluate_model(self.global_model, test_loader)
            on_update_callback({
                "type": "init",
                "round": 0,
                "metrics": initial_eval,
                "clients": {client: {"size": 0, "distribution": self._get_client_distribution(client)} for client in self.clients}
            })
            
            num_rounds = self.params["num_rounds"]
            
            for round_idx in range(1, num_rounds + 1):
                if self.stop_event.is_set():
                    on_update_callback({"type": "status", "status": "stopped", "message": "Training stopped by user."})
                    self.status = "stopped"
                    return
                
                self.current_round = round_idx
                on_update_callback({
                    "type": "round_start",
                    "round": round_idx,
                    "message": f"Starting Federated Round {round_idx}/{num_rounds}"
                })
                
                # We collect client updates and local evaluation metrics
                round_client_weights = []
                round_client_sizes = []
                client_round_metrics = {}
                
                global_weights = copy.deepcopy(self.global_model.state_dict())
                
                for client in self.clients:
                    if self.stop_event.is_set():
                        self.status = "stopped"
                        return
                    
                    on_update_callback({
                        "type": "client_start",
                        "client": client,
                        "round": round_idx,
                        "message": f"{client.replace('_', ' ').title()} local training..."
                    })
                    
                    # Load client dataset
                    client_dir = os.path.join(self.data_dir, client)
                    train_dataset = MRIDataset(os.path.join(client_dir, "train"))
                    val_dataset = MRIDataset(os.path.join(client_dir, "val"))
                    
                    train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=self.params["batch_size"], shuffle=False)
                    
                    # Instantiate client model and load current global weights
                    client_model = BrainTumorCNN()
                    client_model.load_state_dict(copy.deepcopy(global_weights))
                    
                    # Optimizer & Criterion
                    optimizer = optim.Adam(client_model.parameters(), lr=self.params["learning_rate"])
                    criterion = nn.BCELoss()
                    
                    # Train locally
                    local_loss = 0.0
                    local_acc = 0.0
                    for epoch in range(1, self.params["local_epochs"] + 1):
                        if self.stop_event.is_set():
                            self.status = "stopped"
                            return
                        loss_val, acc_val = train_one_epoch(client_model, train_loader, optimizer, criterion)
                        local_loss += loss_val
                        local_acc += acc_val
                    
                    local_loss /= self.params["local_epochs"]
                    local_acc /= self.params["local_epochs"]
                    
                    # Evaluate locally before applying DP (actual capability of local weights)
                    eval_metrics_before_dp = evaluate_model(client_model, val_loader, criterion)
                    
                    # Apply Differential Privacy if enabled
                    trained_weights = client_model.state_dict()
                    update_norm = 0.0
                    if self.params["dp_enabled"]:
                        final_weights, update_norm = apply_differential_privacy(
                            global_weights, 
                            trained_weights,
                            clip_norm=self.params["dp_clip_norm"],
                            noise_multiplier=self.params["dp_noise_multiplier"]
                        )
                    else:
                        final_weights = copy.deepcopy(trained_weights)
                        # Compute norm without clipping just for visualization
                        sum_sq = 0.0
                        for name in global_weights.keys():
                            if global_weights[name].is_floating_point():
                                sum_sq += torch.sum((trained_weights[name] - global_weights[name]) ** 2).item()
                        update_norm = np.sqrt(sum_sq)
                    
                    # Save weights and dataset size
                    round_client_weights.append(final_weights)
                    client_size = len(train_dataset)
                    round_client_sizes.append(client_size)
                    
                    # Evaluate locally with local validation data after DP (simulated noise influence)
                    client_model_dp = BrainTumorCNN()
                    client_model_dp.load_state_dict(final_weights)
                    eval_metrics_after_dp = evaluate_model(client_model_dp, val_loader, criterion)
                    
                    client_round_metrics[client] = {
                        "size": client_size,
                        "local_train_loss": float(local_loss),
                        "local_train_accuracy": float(local_acc),
                        "val_accuracy": float(eval_metrics_before_dp["accuracy"]),
                        "val_accuracy_dp": float(eval_metrics_after_dp["accuracy"]),
                        "val_loss": float(eval_metrics_after_dp["loss"]),
                        "update_norm": float(update_norm)
                    }
                    
                    on_update_callback({
                        "type": "client_complete",
                        "client": client,
                        "round": round_idx,
                        "metrics": client_round_metrics[client]
                    })
                    
                    # Small delay for UI visualization effect
                    time.sleep(0.4)
                    
                # 2. Server Aggregation (FedAvg)
                if self.stop_event.is_set():
                    self.status = "stopped"
                    return
                
                on_update_callback({
                    "type": "aggregation_start",
                    "round": round_idx,
                    "message": "Aggregating client weights via FedAvg..."
                })
                
                total_samples = sum(round_client_sizes)
                aggregated_weights = {}
                
                first_weights = round_client_weights[0]
                for name in first_weights.keys():
                    if first_weights[name].is_floating_point():
                        # Weighted average
                        temp = torch.zeros_like(first_weights[name])
                        for client_w, client_size in zip(round_client_weights, round_client_sizes):
                            weight_factor = client_size / total_samples
                            temp += client_w[name] * weight_factor
                        aggregated_weights[name] = temp
                    else:
                        # For non-floating-point parameters/buffers (like batch norm running counts)
                        # We just take client A's value
                        aggregated_weights[name] = copy.deepcopy(first_weights[name])
                        
                # Update global model with aggregated weights
                self.global_model.load_state_dict(aggregated_weights)
                
                # Evaluate global model on global test set
                global_eval = evaluate_model(self.global_model, test_loader)
                
                # Compile round result
                round_result = {
                    "round": round_idx,
                    "metrics": global_eval,
                    "clients": client_round_metrics
                }
                
                self.metrics_history.append(round_result)
                
                on_update_callback({
                    "type": "round_complete",
                    "round": round_idx,
                    "metrics": global_eval,
                    "clients": client_round_metrics,
                    "message": f"Round {round_idx} complete. Global Test Accuracy: {global_eval['accuracy']:.2%}"
                })
                
                # Add delay between rounds for visualization
                time.sleep(1.0)
                
            self.status = "idle"
            on_update_callback({
                "type": "status",
                "status": "completed",
                "message": "Federated Learning training completed successfully!"
            })
            
        except Exception as e:
            self.status = "idle"
            import traceback
            error_msg = traceback.format_exc()
            on_update_callback({
                "type": "error",
                "message": f"Training failed with error: {str(e)}",
                "detail": error_msg
            })

    def _get_client_distribution(self, client):
        """
        Helper to return train size and percentage of positive cases for UI metadata.
        """
        try:
            client_dir = os.path.join(self.data_dir, client, "train")
            norm_count = len(os.listdir(os.path.join(client_dir, "normal")))
            tumor_count = len(os.listdir(os.path.join(client_dir, "tumor")))
            total = norm_count + tumor_count
            return {
                "total": total,
                "normal": norm_count,
                "tumor": tumor_count,
                "tumor_ratio": tumor_count / total if total > 0 else 0
            }
        except Exception:
            return {"total": 0, "normal": 0, "tumor": 0, "tumor_ratio": 0}
            
    def run_single_inference(self, img_path):
        """
        Runs prediction on a single MRI image.
        """
        self.global_model.eval()
        
        img = Image.open(img_path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]
        
        with torch.no_grad():
            output = self.global_model(img_tensor)
            probability = float(output.item())
            
        return {
            "prediction": "Tumor" if probability >= 0.5 else "Normal",
            "probability": probability,
            "raw_output": probability
        }
