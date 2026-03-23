import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from src.data_preprocessing import TicketDataProcessor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class TicketTrainer:
    """Enhanced trainer for BERT ticket classification with experiment tracking."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 5, 
                 learning_rate: float = 2e-5, batch_size: int = 16, 
                 epochs: int = 10, max_length: int = 128, use_wandb: bool = True):
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.use_wandb = use_wandb
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )
        
        # Add dropout layer if needed
        if hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Linear):
                self.model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    self.model.classifier
                )
        
        # Initialize processor
        self.processor = TicketDataProcessor(
            tokenizer_name=model_name, 
            max_len=max_length
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=f'runs/ticket_classification_{int(time.time())}')
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project="nlp-ticket-classification",
                config={
                    "model_name": model_name,
                    "num_labels": num_labels,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "max_length": max_length
                }
            )
    
    def prepare_data(self, data_path: str = "data/processed_tickets.pt"):
        """Load and prepare training data."""
        logger.info(f"Loading data from {data_path}")
        
        if os.path.exists(data_path):
            processed_data = torch.load(data_path)
        else:
            logger.info("Processed data not found. Creating from scratch...")
            train_df, val_df, test_df = self.processor.load_and_preprocess(
                "data/tickets.csv", 
                generate_synthetic=True, 
                synthetic_count=1000
            )
            processed_data = self.processor.save_processed_data(train_df, val_df, test_df, data_path)
        
        # Create datasets
        train_dataset = TensorDataset(
            processed_data['train']['input_ids'],
            processed_data['train']['attention_mask'],
            processed_data['train']['labels']
        )
        
        val_dataset = TensorDataset(
            processed_data['val']['input_ids'],
            processed_data['val']['attention_mask'],
            processed_data['val']['labels']
        )
        
        test_dataset = TensorDataset(
            processed_data['test']['input_ids'],
            processed_data['test']['attention_mask'],
            processed_data['test']['labels']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        logger.info(f"Data loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return processed_data
    
    def train_epoch(self, optimizer, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Collect predictions for metrics
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, f1, accuracy
    
    def evaluate(self, data_loader, phase="Validation"):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating ({phase})"):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, fscore, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        return avg_loss, f1, accuracy, precision, recall, all_predictions, all_labels
    
    def train(self, data_path: str = "data/processed_tickets.pt"):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Prepare data
        processed_data = self.prepare_data(data_path)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        total_steps = len(self.train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0
        train_history = {
            'train_loss': [],
            'train_f1': [],
            'train_acc': [],
            'val_loss': [],
            'val_f1': [],
            'val_acc': []
        }
        
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Training
            train_loss, train_f1, train_acc = self.train_epoch(optimizer, scheduler)
            
            # Validation
            val_loss, val_f1, val_acc, val_precision, val_recall, _, _ = self.evaluate(self.val_loader)
            
            # Log metrics
            train_history['train_loss'].append(train_loss)
            train_history['train_f1'].append(train_f1)
            train_history['train_acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_f1'].append(val_f1)
            train_history['val_acc'].append(val_acc)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('F1/Train', train_f1, epoch)
            self.writer.add_scalar('F1/Validation', val_f1, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            # WandB logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_f1': train_f1,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_val_f1,
                    'config': {
                        'model_name': self.model_name,
                        'num_labels': self.num_labels,
                        'max_length': self.max_length
                    }
                }, 'models/best_model.pt')
                logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info("Early stopping triggered")
                break
        
        # Final evaluation on test set
        logger.info("Final evaluation on test set...")
        test_loss, test_f1, test_acc, test_precision, test_recall, test_predictions, test_labels = self.evaluate(self.test_loader, "Test")
        
        logger.info(f"Test Results - Loss: {test_loss:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
        
        # Save final results
        results = {
            'train_history': train_history,
            'test_metrics': {
                'loss': test_loss,
                'f1': test_f1,
                'accuracy': test_acc,
                'precision': test_precision.tolist(),
                'recall': test_recall.tolist()
            },
            'predictions': test_predictions,
            'labels': test_labels,
            'categories': processed_data['metadata']['categories']
        }
        
        torch.save(results, 'models/training_results.pt')
        
        # Close tensorboard writer
        self.writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        return results

def main():
    """Main training function."""
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize trainer
    trainer = TicketTrainer(
        model_name='bert-base-uncased',
        num_labels=5,
        learning_rate=2e-5,
        batch_size=16,
        epochs=10,
        max_length=128,
        use_wandb=True
    )
    
    # Start training
    results = trainer.train()
    
    logger.info("Training completed successfully!")
    logger.info(f"Best test F1-score: {results['test_metrics']['f1']:.4f}")

if __name__ == "__main__":
    main()
