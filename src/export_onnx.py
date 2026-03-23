import torch
import os
import logging
from transformers import BertForSequenceClassification, BertTokenizer
import onnxruntime as ort
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ONNXExporter:
    """Export PyTorch BERT model to ONNX format for optimized inference."""
    
    def __init__(self, model_path: str = 'models/best_model.pt', 
                 output_path: str = 'models/model.onnx'):
        self.model_path = model_path
        self.output_path = output_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the PyTorch model."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        
        self.model = BertForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=config['num_labels']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        self.max_length = config['max_length']
        self.num_labels = config['num_labels']
        
        logger.info(f"Model loaded with {self.num_labels} classes")
        
        return config
    
    def export_to_onnx(self):
        """Export model to ONNX format."""
        logger.info("Exporting model to ONNX format...")
        
        # Create dummy input
        dummy_input_ids = torch.randint(0, 30000, (1, self.max_length)).to(self.device)
        dummy_attention_mask = torch.ones(1, self.max_length).to(self.device)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            self.output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            },
            verbose=False
        )
        
        logger.info(f"Model exported to ONNX format: {self.output_path}")
        
        # Verify the exported model
        self.verify_onnx_model()
        
        return self.output_path
    
    def verify_onnx_model(self):
        """Verify the exported ONNX model."""
        logger.info("Verifying ONNX model...")
        
        try:
            # Load ONNX model
            onnx_session = ort.InferenceSession(self.output_path)
            
            # Get input info
            input_info = onnx_session.get_inputs()
            output_info = onnx_session.get_outputs()
            
            logger.info("ONNX Model Inputs:")
            for inp in input_info:
                logger.info(f"  - {inp.name}: {inp.shape} ({inp.type})")
            
            logger.info("ONNX Model Outputs:")
            for out in output_info:
                logger.info(f"  - {out.name}: {out.shape} ({out.type})")
            
            # Test with sample input
            sample_text = "Login not working for user account"
            encoded = self.tokenizer(
                sample_text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].numpy()
            attention_mask = encoded['attention_mask'].numpy()
            
            # Run inference
            ort_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            ort_outputs = onnx_session.run(None, ort_inputs)
            logits = ort_outputs[0]
            probabilities = torch.softmax(torch.from_numpy(logits), dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            
            logger.info(f"Sample prediction: {prediction.item()}")
            logger.info("ONNX model verification successful!")
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {str(e)}")
            raise
    
    def compare_pytorch_onnx(self, sample_text: str = "Server down for maintenance"):
        """Compare outputs between PyTorch and ONNX models."""
        logger.info("Comparing PyTorch and ONNX outputs...")
        
        # PyTorch inference
        encoded = self.tokenizer(
            sample_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            pytorch_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pytorch_logits = pytorch_outputs.logits
            pytorch_probs = torch.softmax(pytorch_logits, dim=-1)
        
        # ONNX inference
        onnx_session = ort.InferenceSession(self.output_path)
        ort_inputs = {
            'input_ids': input_ids.cpu().numpy(),
            'attention_mask': attention_mask.cpu().numpy()
        }
        
        onnx_outputs = onnx_session.run(None, ort_inputs)
        onnx_logits = torch.from_numpy(onnx_outputs[0])
        onnx_probs = torch.softmax(onnx_logits, dim=-1)
        
        # Compare
        max_diff = torch.max(torch.abs(pytorch_probs - onnx_probs)).item()
        logger.info(f"Maximum probability difference: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            logger.info("✅ PyTorch and ONNX outputs match!")
        else:
            logger.warning(f"⚠️  Outputs differ by {max_diff:.6f}")
        
        return {
            'pytorch_probs': pytorch_probs.squeeze().tolist(),
            'onnx_probs': onnx_probs.squeeze().tolist(),
            'max_difference': max_diff
        }

def main():
    """Main export function."""
    exporter = ONNXExporter()
    
    # Load model
    config = exporter.load_model()
    
    # Export to ONNX
    onnx_path = exporter.export_to_onnx()
    
    # Compare outputs
    comparison = exporter.compare_pytorch_onnx()
    
    logger.info("ONNX export completed successfully!")
    logger.info(f"ONNX model saved to: {onnx_path}")
    
    return onnx_path, comparison

if __name__ == "__main__":
    main()
