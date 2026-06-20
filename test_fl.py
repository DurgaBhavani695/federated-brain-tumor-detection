import os
import unittest
import torch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Import project modules
from model import BrainTumorCNN, apply_differential_privacy, MRIDataset
from synthetic_data import generate_mri_slice, setup_federated_dataset
from server import app

class TestFederatedBrainTumorDetection(unittest.TestCase):
    
    def test_cnn_dimensions(self):
        """
        Verify that the CNN correctly takes a batch of grayscale 128x128 images
        and outputs a single channel prediction probability.
        """
        model = BrainTumorCNN()
        model.eval()
        
        # Simulating batch size of 4, 1 channel, 128x128 pixels
        dummy_input = torch.randn(4, 1, 128, 128)
        
        with torch.no_grad():
            output = model(dummy_input)
            
        self.assertEqual(output.shape, (4,))
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))
        print("Success: CNN dimensions and output bounds verified.")

    def test_mri_generator(self):
        """
        Verify that the synthetic image generator generates valid 128x128 grayscale slices
        with pixel values in [0, 255].
        """
        img_norm = generate_mri_slice(has_tumor=False, img_size=128)
        img_tumor = generate_mri_slice(has_tumor=True, img_size=128)
        
        self.assertEqual(img_norm.size, (128, 128))
        self.assertEqual(img_norm.mode, 'L')
        self.assertEqual(img_tumor.size, (128, 128))
        self.assertEqual(img_tumor.mode, 'L')
        
        arr_norm = np.array(img_norm)
        self.assertTrue(arr_norm.min() >= 0)
        self.assertTrue(arr_norm.max() <= 255)
        print("Success: MRI slice generator output format verified.")

    def test_differential_privacy_clipping(self):
        """
        Verify that our Local Differential Privacy mechanism bounds updates.
        We check if updates with larger L2 norms are successfully clipped to the limit.
        """
        # Create base model state dicts
        model_orig = BrainTumorCNN()
        orig_weights = model_orig.state_dict()
        
        # Create a trained copy and inflate its weights to force clipping
        trained_weights = {}
        for name, weight in orig_weights.items():
            if weight.is_floating_point():
                # Add a massive delta to exceed clip bound of 1.0
                trained_weights[name] = weight + 50.0
            else:
                trained_weights[name] = weight.clone()
                
        clip_norm = 1.0
        noise_mult = 0.0  # Turn noise off to inspect clipping math purely
        
        dp_weights, update_norm = apply_differential_privacy(
            orig_weights, trained_weights, clip_norm=clip_norm, noise_multiplier=noise_mult
        )
        
        # Recompute delta norm post DP
        sum_sq = 0.0
        for name in orig_weights.keys():
            if orig_weights[name].is_floating_point():
                sum_sq += torch.sum((dp_weights[name] - orig_weights[name]) ** 2).item()
        dp_update_norm = np.sqrt(sum_sq)
        
        self.assertGreater(update_norm, clip_norm)
        # The new update norm must be precisely equal to clip_norm (with floating point tolerance)
        self.assertAlmostEqual(dp_update_norm, clip_norm, places=4)
        print("Success: Differential Privacy gradient clipping bounds verified.")

    def test_api_server_routes(self):
        """
        Test the FastAPI web routes utilizing the TestClient.
        """
        client = TestClient(app)
        
        # Test config GET
        response = client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        config = response.json()
        self.assertIn("num_rounds", config)
        self.assertIn("dp_enabled", config)
        
        # Test config POST
        payload = {"num_rounds": 6, "dp_enabled": False}
        response = client.post("/api/config", json=payload)
        self.assertEqual(response.status_code, 200)
        updated_config = response.json()["params"]
        self.assertEqual(updated_config["num_rounds"], 6)
        self.assertEqual(updated_config["dp_enabled"], False)
        
        # Test status endpoint
        response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        status_data = response.json()
        self.assertIn("status", status_data)
        self.assertIn("clients", status_data)
        
        print("Success: FastAPI server routes verified.")

if __name__ == "__main__":
    unittest.main()
