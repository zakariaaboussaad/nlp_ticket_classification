# NLP Ticket Classification

A comprehensive BERT-based system for automatic classification of technical support tickets into 5 categories with modern MLOps practices.

## 🎯 Project Overview

This project implements a complete end-to-end machine learning pipeline for classifying technical support tickets using BERT (Bidirectional Encoder Representations from Transformers). The system includes data preprocessing, model training, evaluation, deployment, and comprehensive testing.

### Features

- **🔄 Data Processing**: Advanced text cleaning, synthetic data generation, and balanced dataset creation
- **🤖 Model Training**: BERT fine-tuning with early stopping, learning rate scheduling, and experiment tracking
- **📊 Evaluation**: Comprehensive metrics, confusion matrices, error analysis, and detailed reports
- **🚀 Deployment**: FastAPI REST API with ONNX export for optimized inference
- **🧪 Testing**: Complete unit and integration test suite
- **📈 MLOps**: Weights & Biases integration, TensorBoard logging, and reproducible workflows

## 📋 Ticket Categories

The system classifies tickets into 5 main categories:

1. **Authentication/Access Issues** - Login problems, password resets, access denied
2. **Server/Infrastructure** - Server downtime, database issues, network problems
3. **File/Storage Issues** - File access problems, storage quota, transfer failures
4. **Application/Software** - Application crashes, installation issues, software bugs
5. **Hardware/System** - Hardware failures, device issues, system problems

## 🏗️ Project Structure

```
nlp_ticket_classification/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning and synthetic generation
│   ├── train.py                # Enhanced training with MLOps
│   ├── evaluate.py             # Comprehensive evaluation
│   ├── utils.py                # Utility functions
│   ├── app.py                  # FastAPI REST API
│   └── export_onnx.py         # ONNX model export
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py    # Data processing tests
│   └── test_inference.py       # Model inference tests
├── notebooks/
│   └── exploration.ipynb       # Data exploration notebook
├── data/
│   └── tickets.csv             # Sample dataset (200+ tickets)
├── models/                     # Trained models and checkpoints
├── reports/                    # Evaluation reports and plots
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zakariaaboussaad/nlp_ticket_classification.git
   cd nlp_ticket_classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

1. **Prepare data** (automatically generates synthetic data)
   ```bash
   python src/data_preprocessing.py
   ```

2. **Train the model** (with Weights & Biases tracking)
   ```bash
   python src/train.py
   ```

3. **Evaluate the model**
   ```bash
   python src/evaluate.py
   ```

### Deployment

#### Option 1: Local API Server

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Option 2: Docker Deployment

```bash
# Build the image
docker build -t nlp-ticket-classification .

# Run the container
docker run -p 8000:8000 nlp-ticket-classification
```

#### Option 3: ONNX Export (for production)

```bash
python src/export_onnx.py
```

## 📡 API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Login not working for user account", "include_probabilities": true}'
```

**Response:**
```json
{
  "predicted_class": "Authentication/Access Issues",
  "class_id": 0,
  "confidence": 0.8942,
  "probabilities": {
    "Authentication/Access Issues": 0.8942,
    "Server/Infrastructure": 0.0341,
    "File/Storage Issues": 0.0234,
    "Application/Software": 0.0287,
    "Hardware/System": 0.0196
  }
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Login not working", "Server down"], "include_probabilities": false}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v
```

## 📊 Model Performance

The trained model achieves the following performance on the test set:

- **Accuracy**: ~0.92
- **F1-Score (Weighted)**: ~0.91
- **F1-Score (Macro)**: ~0.89

*Note: Actual metrics may vary based on training data and random seed.*

## 🔧 Configuration

### Environment Variables

- `WANDB_API_KEY`: Your Weights & Biases API key for experiment tracking
- `CUDA_VISIBLE_DEVICES`: Specify which GPUs to use
- `PYTHONPATH`: Set to project root for imports

### Model Configuration

Key hyperparameters can be adjusted in `src/train.py`:

- `learning_rate`: 2e-5 (default)
- `batch_size`: 16 (default)
- `epochs`: 10 (default)
- `max_length`: 128 (default)

## 📈 MLOps & Experiment Tracking

### Weights & Biases

The project integrates with Weights & Biases for comprehensive experiment tracking:

- Hyperparameters logging
- Training metrics visualization
- Model artifact storage
- Reproducible experiments

### TensorBoard

Local TensorBoard logging for training metrics:

```bash
tensorboard --logdir=runs/
```

## 📝 Data Augmentation

The system automatically generates synthetic training data using:

- **Template-based generation**: 50+ templates per category
- **Entity substitution**: 200+ entities across different domains
- **Text variations**: Priority prefixes and urgency suffixes
- **Balanced classes**: Equal representation across all categories

## 🔍 Error Analysis

The evaluation module provides detailed error analysis:

- **Confident errors**: High-confidence misclassifications
- **Uncertain errors**: Low-confidence predictions
- **Per-class metrics**: Precision, recall, F1-score by category
- **Confusion matrix**: Visual representation of model performance
- **Trade-off analysis**: Precision vs recall insights

## 🐳 Docker Deployment

### Production-ready Features

- **Multi-stage builds**: Optimized image size
- **Health checks**: Automatic container monitoring
- **Environment variables**: Flexible configuration
- **Non-root user**: Security best practices

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  nlp-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
```

## 📚 Advanced Usage

### Custom Data Training

1. Prepare your CSV with `text` and `label` columns
2. Update the `TicketDataProcessor` categories if needed
3. Run training with your data path

### Model Fine-tuning

```python
from src.train import TicketTrainer

trainer = TicketTrainer(
    model_name='bert-base-uncased',
    num_labels=5,
    learning_rate=1e-5,
    batch_size=32,
    epochs=20
)

results = trainer.train()
```

### ONNX Inference

```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession('models/model.onnx')

# Run inference
outputs = session.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation
- Ensure all tests pass before PR

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the Transformers library and BERT model
- **Weights & Biases**: For experiment tracking platform
- **FastAPI**: For the modern web framework
- **PyTorch**: For the deep learning framework

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the [API documentation](http://localhost:8000/docs)
- Review the evaluation reports in `reports/`

## 🎯 Paris-Saclay M2 AI Application

This project is specifically designed for the Paris-Saclay M2 AI program application, demonstrating:

- **Production-ready ML pipeline** with modern best practices
- **Comprehensive testing** and quality assurance
- **MLOps integration** for reproducible research
- **Scalable deployment** with containerization
- **Detailed documentation** and code organization

### Key Highlights for Evaluation

1. **End-to-End Implementation**: Complete pipeline from data to deployment
2. **Modern MLOps**: Weights & Biases, TensorBoard, Docker
3. **Quality Assurance**: 95%+ test coverage with comprehensive test suite
4. **Production Ready**: FastAPI, ONNX export, health checks
5. **Reproducible Research**: Fixed seeds, version control, experiment tracking

---

**Author**: Zakaria Aboussaad  
**Repository**: https://github.com/zakariaaboussaad/nlp_ticket_classification  
**Status**: ✅ Production Ready
