import pytest
import pandas as pd
import torch
import numpy as np
from src.data_preprocessing import TicketDataProcessor
import tempfile
import os

class TestTicketDataProcessor:
    """Test suite for TicketDataProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for testing."""
        return TicketDataProcessor(max_len=64)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'text': [
                'Login not working for user account',
                'Server down for maintenance',
                'Cannot access shared folder',
                'Application crashes on startup',
                'Hardware failure detected'
            ],
            'label': [0, 1, 2, 3, 4]
        })
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.tokenizer_name == 'bert-base-uncased'
        assert processor.max_len == 64
        assert processor.tokenizer is not None
        assert len(processor.categories) == 5
        assert len(processor.ticket_templates) == 5
        assert len(processor.entities) > 0
    
    def test_clean_text(self, processor):
        """Test text cleaning functionality."""
        # Test HTML removal
        html_text = "<p>Login not working</p><br>Please help"
        cleaned = processor.clean_text(html_text)
        assert '<p>' not in cleaned
        assert '<br>' not in cleaned
        assert 'login not working' in cleaned
        
        # Test URL removal
        url_text = "Visit https://example.com for help"
        cleaned = processor.clean_text(url_text)
        assert 'https://example.com' not in cleaned
        
        # Test email removal
        email_text = "Contact support@example.com for help"
        cleaned = processor.clean_text(email_text)
        assert 'support@example.com' not in cleaned
        
        # Test accent normalization
        accent_text = "café résumé naïve"
        cleaned = processor.clean_text(accent_text)
        assert 'café' not in cleaned  # Should be normalized
        
        # Test lowercase conversion
        upper_text = "LOGIN NOT WORKING"
        cleaned = processor.clean_text(upper_text)
        assert cleaned.islower()
    
    def test_generate_synthetic_tickets(self, processor):
        """Test synthetic ticket generation."""
        num_tickets = 50
        synthetic_df = processor.generate_synthetic_tickets(num_tickets)
        
        # Check shape
        assert len(synthetic_df) == num_tickets
        assert 'text' in synthetic_df.columns
        assert 'label' in synthetic_df.columns
        assert 'raw_text' in synthetic_df.columns
        
        # Check labels are valid
        assert all(label in processor.categories.keys() for label in synthetic_df['label'])
        
        # Check text is not empty
        assert all(len(text.strip()) > 0 for text in synthetic_df['text'])
        
        # Check raw text contains placeholders that were filled
        assert any('system' in text or 'server' in text for text in synthetic_df['raw_text'])
    
    def test_load_and_preprocess_with_csv(self, processor, sample_data):
        """Test loading and preprocessing with CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_csv = f.name
        
        try:
            # Test preprocessing
            train, val, test = processor.load_and_preprocess(
                temp_csv, 
                generate_synthetic=False, 
                synthetic_count=0
            )
            
            # Check splits
            assert len(train) + len(val) + len(test) == len(sample_data)
            assert len(train) > 0
            assert len(val) > 0
            assert len(test) > 0
            
            # Check columns
            assert 'text' in train.columns
            assert 'label' in train.columns
            assert 'text' in val.columns
            assert 'label' in val.columns
            assert 'text' in test.columns
            assert 'label' in test.columns
            
            # Check text cleaning
            assert all(isinstance(text, str) for text in train['text'])
            assert all(text.islower() for text in train['text'])
            
        finally:
            os.unlink(temp_csv)
    
    def test_load_and_preprocess_with_synthetic(self, processor, sample_data):
        """Test loading and preprocessing with synthetic data generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_csv = f.name
        
        try:
            synthetic_count = 100
            train, val, test = processor.load_and_preprocess(
                temp_csv, 
                generate_synthetic=True, 
                synthetic_count=synthetic_count
            )
            
            # Check synthetic data was added
            total_samples = len(train) + len(val) + len(test)
            assert total_samples > len(sample_data)
            
            # Check class balance (should be balanced after synthetic generation)
            label_counts = pd.concat([train, val, test])['label'].value_counts()
            assert len(set(label_counts.values)) == 1  # All classes should have same count
            
        finally:
            os.unlink(temp_csv)
    
    def test_tokenize_texts(self, processor):
        """Test text tokenization."""
        texts = pd.Series([
            'Login not working',
            'Server down',
            'Cannot access files'
        ])
        
        encoded = processor.tokenize_texts(texts)
        
        # Check output structure
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        
        # Check tensor types
        assert isinstance(encoded['input_ids'], torch.Tensor)
        assert isinstance(encoded['attention_mask'], torch.Tensor)
        
        # Check shapes
        assert encoded['input_ids'].shape[0] == len(texts)
        assert encoded['attention_mask'].shape[0] == len(texts)
        assert encoded['input_ids'].shape[1] <= processor.max_len
        assert encoded['attention_mask'].shape[1] <= processor.max_len
    
    def test_save_processed_data(self, processor, sample_data):
        """Test saving processed data."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Split data
            train = sample_data.iloc[:3]
            val = sample_data.iloc[3:4]
            test = sample_data.iloc[4:]
            
            # Save processed data
            processed_data = processor.save_processed_data(train, val, test, temp_path)
            
            # Check file was created
            assert os.path.exists(temp_path)
            
            # Check structure
            assert 'train' in processed_data
            assert 'val' in processed_data
            assert 'test' in processed_data
            assert 'metadata' in processed_data
            
            # Check metadata
            metadata = processed_data['metadata']
            assert 'num_classes' in metadata
            assert 'categories' in metadata
            assert 'max_length' in metadata
            assert 'tokenizer_name' in metadata
            
            # Check data structure
            for split in ['train', 'val', 'test']:
                split_data = processed_data[split]
                assert 'input_ids' in split_data
                assert 'attention_mask' in split_data
                assert 'labels' in split_data
                assert 'texts' in split_data
                
                # Check tensor types
                assert isinstance(split_data['input_ids'], torch.Tensor)
                assert isinstance(split_data['attention_mask'], torch.Tensor)
                assert isinstance(split_data['labels'], torch.Tensor)
                assert isinstance(split_data['texts'], list)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_reproducibility(self, processor):
        """Test that results are reproducible with same seed."""
        # Generate synthetic data twice
        synthetic1 = processor.generate_synthetic_tickets(50)
        synthetic2 = processor.generate_synthetic_tickets(50)
        
        # Results should be different due to randomness
        assert not synthetic1['text'].equals(synthetic2['text'])
    
    def test_edge_cases(self, processor):
        """Test edge cases and error handling."""
        # Test empty text
        empty_cleaned = processor.clean_text("")
        assert empty_cleaned == ""
        
        # Test None input
        none_cleaned = processor.clean_text(None)
        assert isinstance(none_cleaned, str)
        
        # Test very long text
        long_text = "word " * 1000
        cleaned = processor.clean_text(long_text)
        assert len(cleaned) > 0
        
        # Test special characters
        special_text = "Login @#$%^&*() not working!"
        cleaned = processor.clean_text(special_text)
        assert "login" in cleaned
        assert "not working" in cleaned

def test_backward_compatibility():
    """Test backward compatibility functions."""
    from src.data_preprocessing import load_and_split, tokenize_texts
    
    # These should work without errors
    assert callable(load_and_split)
    assert callable(tokenize_texts)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
