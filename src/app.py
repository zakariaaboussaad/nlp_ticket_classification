from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import logging
import os
import onnxruntime as ort
import json
from src.data_preprocessing import TicketDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NLP Ticket Classification API",
    description="API for classifying technical support tickets using BERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str
    include_probabilities: bool = False

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    include_probabilities: bool = False

class PredictionResponse(BaseModel):
    predicted_class: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int

# Global variables for model and tokenizer
model = None
tokenizer = None
categories = None
device = None
max_length = 128
onnx_session = None
use_onnx = False

class TicketClassifier:
    """Ticket classification service."""
    
    def __init__(self, model_path: str = 'models/best_model.pt', 
                 use_onnx: bool = False, onnx_path: str = 'models/model.onnx'):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.onnx_path = onnx_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.load_model()
        
    def load_model(self):
        """Load the trained model."""
        global model, tokenizer, categories, device, max_length, onnx_session, use_onnx
        
        logger.info(f"Loading model on device: {self.device}")
        
        if self.use_onnx and os.path.exists(self.onnx_path):
            # Load ONNX model
            logger.info("Loading ONNX model...")
            onnx_session = ort.InferenceSession(self.onnx_path)
            use_onnx = True
            
            # Load tokenizer and metadata
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                config = checkpoint['config']
                tokenizer = BertTokenizer.from_pretrained(config['model_name'])
                max_length = config['max_length']
                categories = {
                    0: "Authentication/Access Issues",
                    1: "Server/Infrastructure",
                    2: "File/Storage Issues", 
                    3: "Application/Software",
                    4: "Hardware/System"
                }
        else:
            # Load PyTorch model
            logger.info("Loading PyTorch model...")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config = checkpoint['config']
            
            model = BertForSequenceClassification.from_pretrained(
                config['model_name'],
                num_labels=config['num_labels']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            tokenizer = BertTokenizer.from_pretrained(config['model_name'])
            max_length = config['max_length']
            categories = {
                0: "Authentication/Access Issues",
                1: "Server/Infrastructure",
                2: "File/Storage Issues",
                3: "Application/Software",
                4: "Hardware/System"
            }
            
            use_onnx = False
        
        device = self.device
        logger.info(f"Model loaded successfully with {len(categories)} classes")
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for model input."""
        # Clean text using the same pipeline as training
        processor = TicketDataProcessor(tokenizer_name='bert-base-uncased', max_len=max_length)
        cleaned_text = processor.clean_text(text)
        
        # Tokenize
        encoded = tokenizer(
            cleaned_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {k: v.to(device) for k, v in encoded.items()}
    
    def predict_single(self, text: str, include_probabilities: bool = False) -> Dict[str, Any]:
        """Make prediction for a single text."""
        try:
            # Preprocess
            inputs = self.preprocess_text(text)
            
            if use_onnx:
                # ONNX inference
                input_ids = inputs['input_ids'].cpu().numpy()
                attention_mask = inputs['attention_mask'].cpu().numpy()
                
                ort_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                
                logits = onnx_session.run(None, ort_inputs)[0]
                probabilities = torch.softmax(torch.from_numpy(logits), dim=-1)
                prediction = torch.argmax(probabilities, dim=-1)
                
                confidence = float(torch.max(probabilities))
                class_id = int(prediction.item())
            else:
                # PyTorch inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    prediction = torch.argmax(logits, dim=-1)
                    
                    confidence = float(torch.max(probabilities))
                    class_id = int(prediction.item())
            
            result = {
                'predicted_class': categories[class_id],
                'class_id': class_id,
                'confidence': confidence
            }
            
            if include_probabilities:
                probs = probabilities.squeeze().cpu().numpy()
                result['probabilities'] = {
                    categories[i]: float(probs[i]) for i in range(len(categories))
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, texts: List[str], include_probabilities: bool = False) -> List[Dict[str, Any]]:
        """Make predictions for multiple texts."""
        results = []
        
        for text in texts:
            result = self.predict_single(text, include_probabilities)
            results.append(result)
        
        return results
    
    def export_to_onnx(self, output_path: str = 'models/model.onnx'):
        """Export PyTorch model to ONNX format."""
        if use_onnx:
            logger.info("Model is already in ONNX format")
            return
        
        logger.info("Exporting model to ONNX format...")
        
        # Create dummy input
        dummy_input = {
            'input_ids': torch.randint(0, 30000, (1, max_length)).to(device),
            'attention_mask': torch.ones(1, max_length).to(device)
        }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            output_path,
            export_params=True,
            opset_version=12,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX format: {output_path}")

# Initialize classifier
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize the classifier on startup."""
    global classifier
    try:
        classifier = TicketClassifier()
        logger.info("Ticket classification service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {str(e)}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "NLP Ticket Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if classifier is not None else "unhealthy",
        model_loaded=classifier is not None,
        str(device) if device else "unknown",
        len(categories) if categories else 0
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict class for a single ticket text."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = classifier.predict_single(request.text, request.include_probabilities)
    return PredictionResponse(**result)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict classes for multiple ticket texts."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    predictions = classifier.predict_batch(request.texts, request.include_probabilities)
    
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**pred) for pred in predictions],
        total_processed=len(request.texts),
        processing_time_ms=processing_time
    )

@app.post("/export/onnx")
async def export_model_to_onnx():
    """Export the current model to ONNX format."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        classifier.export_to_onnx()
        return {"message": "Model exported to ONNX format successfully", "path": "models/model.onnx"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get available ticket categories."""
    if categories is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"categories": categories, "count": len(categories)}

@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "ONNX" if use_onnx else "PyTorch",
        "device": str(device),
        "max_length": max_length,
        "num_classes": len(categories),
        "categories": categories
    }

if __name__ == "__main__":
    import uvicorn
    
    # Initialize classifier
    classifier = TicketClassifier()
    
    # Run the app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
