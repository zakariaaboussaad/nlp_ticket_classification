import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from src.app import TicketClassifier, PredictionRequest, BatchPredictionRequest
from src.evaluate import TicketEvaluator
import json

class TestTicketClassifier:
    """Test suite for TicketClassifier class."""
    
    @pytest.fixture
    def mock_model_checkpoint(self):
        """Create a mock model checkpoint."""
        return {
            'config': {
                'model_name': 'bert-base-uncased',
                'num_labels': 5,
                'max_length': 128
            },
            'model_state_dict': {
                'classifier.weight': torch.randn(5, 768),
                'classifier.bias': torch.randn(5)
            }
        }
    
    @pytest.fixture
    def temp_model_file(self, mock_model_checkpoint):
        """Create a temporary model file."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(mock_model_checkpoint, f.name)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_classifier_initialization(self, temp_model_file):
        """Test classifier initialization."""
        with patch('src.app.BertForSequenceClassification') as mock_model_class:
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            classifier = TicketClassifier(model_path=temp_model_file, use_onnx=False)
            
            assert classifier.model_path == temp_model_file
            assert classifier.use_onnx is False
            assert classifier.device is not None
    
    def test_preprocess_text(self, temp_model_file):
        """Test text preprocessing."""
        with patch('src.app.BertForSequenceClassification') as mock_model_class:
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            with patch('src.app.BertTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer.return_value = {
                    'input_ids': torch.randn(1, 128),
                    'attention_mask': torch.ones(1, 128)
                }
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                classifier = TicketClassifier(model_path=temp_model_file, use_onnx=False)
                
                # Test preprocessing
                result = classifier.preprocess_text("Login not working")
                
                assert 'input_ids' in result
                assert 'attention_mask' in result
                assert isinstance(result['input_ids'], torch.Tensor)
                assert isinstance(result['attention_mask'], torch.Tensor)
    
    def test_predict_single_pytorch(self, temp_model_file):
        """Test single prediction with PyTorch model."""
        with patch('src.app.BertForSequenceClassification') as mock_model_class:
            mock_model = Mock()
            mock_output = Mock()
            mock_output.logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model
            
            with patch('src.app.BertTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer.return_value = {
                    'input_ids': torch.randn(1, 128),
                    'attention_mask': torch.ones(1, 128)
                }
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                classifier = TicketClassifier(model_path=temp_model_file, use_onnx=False)
                
                result = classifier.predict_single("Login not working", include_probabilities=True)
                
                assert 'predicted_class' in result
                assert 'class_id' in result
                assert 'confidence' in result
                assert 'probabilities' in result
                assert isinstance(result['confidence'], float)
                assert 0 <= result['confidence'] <= 1
    
    def test_predict_batch(self, temp_model_file):
        """Test batch prediction."""
        with patch('src.app.BertForSequenceClassification') as mock_model_class:
            mock_model = Mock()
            mock_output = Mock()
            mock_output.logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model
            
            with patch('src.app.BertTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer.return_value = {
                    'input_ids': torch.randn(1, 128),
                    'attention_mask': torch.ones(1, 128)
                }
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                classifier = TicketClassifier(model_path=temp_model_file, use_onnx=False)
                
                texts = ["Login not working", "Server down"]
                results = classifier.predict_batch(texts)
                
                assert len(results) == 2
                for result in results:
                    assert 'predicted_class' in result
                    assert 'class_id' in result
                    assert 'confidence' in result

class TestTicketEvaluator:
    """Test suite for TicketEvaluator class."""
    
    @pytest.fixture
    def mock_processed_data(self):
        """Create mock processed data."""
        return {
            'train': {
                'input_ids': torch.randn(10, 128),
                'attention_mask': torch.ones(10, 128),
                'labels': torch.randint(0, 5, (10,)),
                'texts': [f'text {i}' for i in range(10)]
            },
            'val': {
                'input_ids': torch.randn(5, 128),
                'attention_mask': torch.ones(5, 128),
                'labels': torch.randint(0, 5, (5,)),
                'texts': [f'val text {i}' for i in range(5)]
            },
            'test': {
                'input_ids': torch.randn(5, 128),
                'attention_mask': torch.ones(5, 128),
                'labels': torch.randint(0, 5, (5,)),
                'texts': [f'test text {i}' for i in range(5)]
            },
            'metadata': {
                'categories': {
                    0: "Authentication/Access Issues",
                    1: "Server/Infrastructure",
                    2: "File/Storage Issues",
                    3: "Application/Software",
                    4: "Hardware/System"
                }
            }
        }
    
    @pytest.fixture
    def temp_data_file(self, mock_processed_data):
        """Create a temporary data file."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(mock_processed_data, f.name)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.fixture
    def temp_model_file(self):
        """Create a temporary model file."""
        mock_checkpoint = {
            'config': {
                'model_name': 'bert-base-uncased',
                'num_labels': 5,
                'max_length': 128
            },
            'model_state_dict': {
                'classifier.weight': torch.randn(5, 768),
                'classifier.bias': torch.randn(5)
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(mock_checkpoint, f.name)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_evaluator_initialization(self, temp_model_file, temp_data_file):
        """Test evaluator initialization."""
        with patch('src.evaluate.BertForSequenceClassification') as mock_model_class:
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            evaluator = TicketEvaluator(
                model_path=temp_model_file,
                data_path=temp_data_file
            )
            
            assert evaluator.model_path == temp_model_file
            assert evaluator.data_path == temp_data_file
            assert evaluator.device is not None
            assert evaluator.categories is not None
    
    def test_predict(self, temp_model_file, temp_data_file):
        """Test prediction functionality."""
        with patch('src.evaluate.BertForSequenceClassification') as mock_model_class:
            mock_model = Mock()
            mock_output = Mock()
            mock_output.logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model
            
            with patch('src.evaluate.BertTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer.return_value = {
                    'input_ids': torch.randn(1, 128),
                    'attention_mask': torch.ones(1, 128)
                }
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                evaluator = TicketEvaluator(
                    model_path=temp_model_file,
                    data_path=temp_data_file
                )
                
                predictions, probabilities = evaluator.predict(["Test text"])
                
                assert predictions.shape == (1,)
                assert probabilities.shape == (1, 5)
                assert isinstance(predictions, np.ndarray)
                assert isinstance(probabilities, np.ndarray)
    
    def test_evaluate_dataset(self, temp_model_file, temp_data_file):
        """Test dataset evaluation."""
        with patch('src.evaluate.BertForSequenceClassification') as mock_model_class:
            mock_model = Mock()
            mock_output = Mock()
            # Create logits that will predict class 4 (highest value)
            mock_output.logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model
            
            with patch('src.evaluate.BertTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer.return_value = {
                    'input_ids': torch.randn(1, 128),
                    'attention_mask': torch.ones(1, 128)
                }
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                evaluator = TicketEvaluator(
                    model_path=temp_model_file,
                    data_path=temp_data_file
                )
                
                results = evaluator.evaluate_dataset('test')
                
                assert 'accuracy' in results
                assert 'f1_weighted' in results
                assert 'f1_macro' in results
                assert 'classification_report' in results
                assert 'confusion_matrix' in results
                assert 'predictions' in results
                assert 'true_labels' in results
                assert 'probabilities' in results
                assert 'texts' in results
                
                # Check metric ranges
                assert 0 <= results['accuracy'] <= 1
                assert 0 <= results['f1_weighted'] <= 1
                assert 0 <= results['f1_macro'] <= 1
    
    def test_analyze_errors(self, temp_model_file, temp_data_file):
        """Test error analysis."""
        with patch('src.evaluate.BertForSequenceClassification') as mock_model_class:
            mock_model = Mock()
            mock_output = Mock()
            mock_output.logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model
            
            with patch('src.evaluate.BertTokenizer') as mock_tokenizer_class:
                mock_tokenizer = Mock()
                mock_tokenizer.return_value = {
                    'input_ids': torch.randn(1, 128),
                    'attention_mask': torch.ones(1, 128)
                }
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                evaluator = TicketEvaluator(
                    model_path=temp_model_file,
                    data_path=temp_data_file
                )
                
                # Create mock results
                mock_results = {
                    'true_labels': np.array([0, 1, 2, 3, 4]),
                    'predictions': np.array([0, 0, 2, 3, 4]),  # One error at index 1
                    'probabilities': np.array([
                        [0.8, 0.1, 0.05, 0.03, 0.02],
                        [0.6, 0.3, 0.05, 0.03, 0.02],  # Error: predicted 0, true 1
                        [0.1, 0.1, 0.7, 0.05, 0.05],
                        [0.1, 0.1, 0.1, 0.6, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.6]
                    ]),
                    'texts': ['text1', 'text2', 'text3', 'text4', 'text5']
                }
                
                error_analysis = evaluator.analyze_errors(mock_results, top_k=3)
                
                assert 'total_errors' in error_analysis
                assert 'error_rate' in error_analysis
                assert 'errors_by_class' in error_analysis
                assert 'top_confident_errors' in error_analysis
                assert 'top_uncertain_errors' in error_analysis
                
                assert error_analysis['total_errors'] == 1
                assert error_analysis['error_rate'] == 0.2

class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        classifier = Mock()
        classifier.predict_single.return_value = {
            'predicted_class': 'Authentication/Access Issues',
            'class_id': 0,
            'confidence': 0.85,
            'probabilities': {
                'Authentication/Access Issues': 0.85,
                'Server/Infrastructure': 0.05,
                'File/Storage Issues': 0.04,
                'Application/Software': 0.03,
                'Hardware/System': 0.03
            }
        }
        classifier.predict_batch.return_value = [
            {
                'predicted_class': 'Authentication/Access Issues',
                'class_id': 0,
                'confidence': 0.85
            }
        ]
        return classifier
    
    def test_prediction_request_model(self):
        """Test prediction request model."""
        request = PredictionRequest(text="Login not working", include_probabilities=True)
        assert request.text == "Login not working"
        assert request.include_probabilities is True
    
    def test_batch_prediction_request_model(self):
        """Test batch prediction request model."""
        request = BatchPredictionRequest(
            texts=["Login not working", "Server down"],
            include_probabilities=False
        )
        assert len(request.texts) == 2
        assert request.include_probabilities is False
    
    @patch('src.app.classifier')
    def test_predict_single_endpoint(self, mock_classifier_instance, mock_classifier):
        """Test single prediction endpoint."""
        mock_classifier_instance.predict_single.return_value = mock_classifier.predict_single.return_value
        
        from src.app import predict_single
        request = PredictionRequest(text="Login not working", include_probabilities=True)
        
        response = predict_single(request)
        
        assert response.predicted_class == 'Authentication/Access Issues'
        assert response.class_id == 0
        assert response.confidence == 0.85
        assert response.probabilities is not None
    
    @patch('src.app.classifier')
    def test_predict_batch_endpoint(self, mock_classifier_instance, mock_classifier):
        """Test batch prediction endpoint."""
        mock_classifier_instance.predict_batch.return_value = mock_classifier.predict_batch.return_value
        
        from src.app import predict_batch
        request = BatchPredictionRequest(
            texts=["Login not working"],
            include_probabilities=False
        )
        
        response = predict_batch(request)
        
        assert len(response.predictions) == 1
        assert response.total_processed == 1
        assert response.processing_time_ms >= 0

def test_integration():
    """Integration test for the complete pipeline."""
    # This test would require actual model files
    # For now, just test that the modules can be imported
    from src.app import TicketClassifier, app
    from src.evaluate import TicketEvaluator
    from src.data_preprocessing import TicketDataProcessor
    
    assert TicketClassifier is not None
    assert TicketEvaluator is not None
    assert TicketDataProcessor is not None
    assert app is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
