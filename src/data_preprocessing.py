import pandas as pd
import numpy as np
import re
import torch
import random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import unicodedata
from typing import Tuple, Dict, Any
import logging

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketDataProcessor:
    """Enhanced data processor for ticket classification with text cleaning and augmentation."""
    
    def __init__(self, tokenizer_name: str = 'bert-base-uncased', max_len: int = 128):
        self.tokenizer_name = tokenizer_name
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        # Ticket categories and their descriptions
        self.categories = {
            0: "Authentication/Access Issues",
            1: "Server/Infrastructure", 
            2: "File/Storage Issues",
            3: "Application/Software",
            4: "Hardware/System"
        }
        
        # Templates for synthetic data generation
        self.ticket_templates = {
            0: [
                "Cannot login to {system} with {credential_type}",
                "User {user} unable to access {resource}",
                "Password reset not working for {service}",
                "Two-factor authentication failed for {app}",
                "Account locked out of {platform}",
                "Access denied to {module}",
                "Authentication token expired for {service}",
                "Single sign-on not working for {application}",
                "LDAP authentication failed for {user}",
                "API key authentication invalid for {endpoint}"
            ],
            1: [
                "Server {server_name} is down for {reason}",
                "Database connection timeout on {db_name}",
                "Network latency issues on {network_segment}",
                "Load balancer not responding on {port}",
                "Service {service_name} crashed unexpectedly",
                "Memory usage high on server {hostname}",
                "CPU utilization at {percentage}% on {server}",
                "Disk space full on {storage_location}",
                "Backup failed for {system_name}",
                "Replication lag detected in {database}"
            ],
            2: [
                "Cannot access {file_type} files on {location}",
                "File transfer interrupted for {file_name}",
                "Storage quota exceeded for {user_department}",
                "File corruption detected in {directory}",
                "Cannot save documents to {network_drive}",
                "File sharing permissions broken for {folder}",
                "Upload failed for {file_extension} files",
                "Download speed slow for {file_size}",
                "File synchronization error in {cloud_service}",
                "Archive extraction failed for {archive_type}"
            ],
            3: [
                "Application {app_name} crashes on {action}",
                "Software installation failed for {product}",
                "Update not working for {application}",
                "Plugin compatibility issue with {software}",
                "Memory leak detected in {process_name}",
                "Configuration error in {config_file}",
                "License expired for {software_product}",
                "Build failed for {project_name}",
                "Deployment error for {service}",
                "Integration issue between {system1} and {system2}"
            ],
            4: [
                "Hardware failure detected in {component}",
                "{device_type} not responding to commands",
                "Driver issues with {hardware_name}",
                "Overheating problem in {machine_type}",
                "Power supply failure for {equipment}",
                "Display resolution issue on {monitor_type}",
                "Audio not working on {device}",
                "USB device not recognized: {device_name}",
                "Printer not responding: {printer_model}",
                "Touchscreen calibration failed on {tablet}"
            ]
        }
        
        # Entity placeholders for template filling
        self.entities = {
            'system': ['Active Directory', 'Office 365', 'Salesforce', 'SAP', 'JIRA', 'Confluence', 'GitHub', 'AWS', 'Azure', 'Google Workspace'],
            'credential_type': ['username', 'password', 'API key', 'certificate', 'token'],
            'user': ['john.doe', 'jane.smith', 'admin', 'guest', 'service.account'],
            'resource': ['shared folder', 'database', 'application', 'network drive', 'cloud storage'],
            'service': ['email', 'VPN', 'file server', 'web application', 'mobile app'],
            'app': ['Outlook', 'Chrome', 'Excel', 'Word', 'PowerPoint', 'Slack', 'Teams'],
            'platform': ['Windows', 'Linux', 'macOS', 'iOS', 'Android'],
            'module': ['finance', 'HR', 'inventory', 'reporting', 'analytics'],
            'application': ['CRM', 'ERP', 'BI tool', 'project management', 'time tracking'],
            'endpoint': ['REST API', 'GraphQL', 'SOAP', 'webhook'],
            'server_name': ['web01', 'db02', 'app03', 'cache04', 'proxy05'],
            'reason': ['maintenance', 'hardware failure', 'software update', 'network issue', 'power outage'],
            'db_name': ['production', 'staging', 'development', 'analytics', 'archive'],
            'network_segment': ['DMZ', 'internal', 'external', 'VPN', 'wireless'],
            'port': ['80', '443', '3306', '5432', '8080'],
            'service_name': ['nginx', 'apache', 'mysql', 'postgresql', 'redis'],
            'hostname': ['server01', 'server02', 'server03', 'server04', 'server05'],
            'percentage': ['90', '95', '85', '80', '75'],
            'storage_location': ['/var/log', '/home', '/tmp', '/opt', '/data'],
            'system_name': ['ERP', 'CRM', 'BI', 'HR', 'Finance'],
            'database': ['master', 'slave1', 'slave2', 'backup', 'archive'],
            'file_type': ['PDF', 'Excel', 'Word', 'Image', 'Video'],
            'location': ['network drive', 'local disk', 'cloud storage', 'shared folder'],
            'file_name': ['report.xlsx', 'data.csv', 'presentation.pptx', 'document.pdf'],
            'user_department': ['sales', 'marketing', 'finance', 'HR', 'IT'],
            'directory': ['Documents', 'Downloads', 'Desktop', 'Projects', 'Archives'],
            'network_drive': ['Z:', 'Y:', 'X:', 'W:', 'V:'],
            'folder': ['shared', 'public', 'private', 'temp', 'backup'],
            'file_extension': ['jpg', 'png', 'mp4', 'zip', 'docx'],
            'file_size': ['large files', 'small files', 'medium files', 'archives'],
            'cloud_service': ['OneDrive', 'Google Drive', 'Dropbox', 'iCloud', 'Box'],
            'archive_type': ['ZIP', 'RAR', 'TAR', '7Z', 'GZ'],
            'app_name': ['Photoshop', 'AutoCAD', 'Visual Studio', 'IntelliJ', 'Eclipse'],
            'action': ['startup', 'shutdown', 'save', 'export', 'import'],
            'product': ['Microsoft Office', 'Adobe Creative Suite', 'AutoCAD', 'MATLAB', 'SPSS'],
            'software': ['plugin', 'extension', 'add-on', 'module', 'component'],
            'process_name': ['chrome.exe', 'firefox.exe', 'explorer.exe', 'winword.exe', 'excel.exe'],
            'config_file': ['app.config', 'settings.ini', 'web.xml', 'application.properties'],
            'software_product': ['Windows', 'Office', 'Adobe', 'AutoCAD', 'MATLAB'],
            'project_name': ['website', 'mobile app', 'desktop app', 'API service', 'data pipeline'],
            'system1': ['CRM', 'ERP', 'BI', 'HR', 'Finance'],
            'system2': ['Marketing', 'Sales', 'Support', 'Operations', 'Analytics'],
            'component': ['CPU', 'RAM', 'GPU', 'SSD', 'Motherboard'],
            'device_type': ['printer', 'scanner', 'monitor', 'keyboard', 'mouse'],
            'hardware_name': ['NVIDIA driver', 'Intel graphics', 'AMD chipset', 'Realtek audio', 'Broadcom network'],
            'machine_type': ['laptop', 'desktop', 'server', 'workstation', 'tablet'],
            'equipment': ['UPS', 'router', 'switch', 'firewall', 'load balancer'],
            'monitor_type': ['LCD', 'LED', 'OLED', '4K monitor', 'ultrawide'],
            'device': ['flash drive', 'external HDD', 'webcam', 'microphone', 'headphones'],
            'printer_model': ['HP LaserJet', 'Canon PIXMA', 'Epson EcoTank', 'Brother MFC', 'Xerox WorkCentre'],
            'tablet': ['iPad', 'Surface', 'Galaxy Tab', 'Kindle', 'Fire tablet']
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize accents
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
        
        # Convert to lowercase for BERT
        text = text.lower().strip()
        
        return text
    
    def generate_synthetic_tickets(self, num_tickets: int = 1000) -> pd.DataFrame:
        """Generate synthetic ticket data for training augmentation."""
        logger.info(f"Generating {num_tickets} synthetic tickets...")
        
        synthetic_data = []
        
        for _ in range(num_tickets):
            # Randomly select category
            category = random.choice(list(self.ticket_templates.keys()))
            template = random.choice(self.ticket_templates[category])
            
            # Fill template with random entities
            filled_text = template
            for placeholder in re.findall(r'\{(\w+)\}', template):
                if placeholder in self.entities:
                    filled_text = filled_text.replace(f'{{{placeholder}}}', 
                                                     random.choice(self.entities[placeholder]))
            
            # Add some random variations
            if random.random() < 0.3:
                prefixes = ["urgent:", "critical:", "high priority:", "medium priority:", "low priority:"]
                filled_text = f"{random.choice(prefixes)} {filled_text}"
            
            if random.random() < 0.2:
                suffixes = [" please help", " asap", " need assistance", " urgent fix required", " investigate immediately"]
                filled_text = f"{filled_text}{random.choice(suffixes)}"
            
            # Clean the generated text
            cleaned_text = self.clean_text(filled_text)
            
            synthetic_data.append({
                'text': cleaned_text,
                'label': category,
                'raw_text': filled_text  # Keep original for reference
            })
        
        return pd.DataFrame(synthetic_data)
    
    def load_and_preprocess(self, csv_path: str, generate_synthetic: bool = True, 
                          synthetic_count: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load original data, generate synthetic data, and split into train/val/test."""
        logger.info("Loading and preprocessing data...")
        
        # Load original data
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded original dataset with {len(df)} samples")
        except FileNotFoundError:
            logger.warning(f"Original file {csv_path} not found. Creating synthetic dataset only.")
            df = pd.DataFrame(columns=['text', 'label'])
        
        # Remove duplicates
        original_size = len(df)
        df = df.drop_duplicates(subset=['text'])
        logger.info(f"Removed {original_size - len(df)} duplicates")
        
        # Clean existing text
        df['text'] = df['text'].apply(self.clean_text)
        
        # Generate synthetic data if requested
        if generate_synthetic:
            synthetic_df = self.generate_synthetic_tickets(synthetic_count)
            df = pd.concat([df, synthetic_df[['text', 'label']]], ignore_index=True)
            logger.info(f"Added {synthetic_count} synthetic samples. Total dataset size: {len(df)}")
        
        # Remove any remaining duplicates after synthetic generation
        df = df.drop_duplicates(subset=['text'])
        
        # Ensure we have balanced classes
        min_class_size = df['label'].value_counts().min()
        balanced_df = pd.DataFrame()
        
        for label in df['label'].unique():
            class_samples = df[df['label'] == label].sample(min_class_size, random_state=RANDOM_SEED)
            balanced_df = pd.concat([balanced_df, class_samples])
        
        df = balanced_df.reset_index(drop=True)
        logger.info(f"Balanced dataset: {len(df)} samples across {df['label'].nunique()} classes")
        
        # Split into train/val/test with stratification
        train, temp = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED, stratify=df['label'])
        val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED, stratify=temp['label'])
        
        logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    
    def tokenize_texts(self, texts: pd.Series, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize texts using BERT tokenizer."""
        logger.info(f"Tokenizing {len(texts)} texts...")
        
        # Ensure texts are strings
        texts = texts.astype(str).tolist()
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def save_processed_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                          test_data: pd.DataFrame, output_path: str = "data/processed_tickets.pt"):
        """Save processed datasets to disk."""
        logger.info(f"Saving processed data to {output_path}...")
        
        # Tokenize all datasets
        train_inputs = self.tokenize_texts(train_data['text'])
        val_inputs = self.tokenize_texts(val_data['text'])
        test_inputs = self.tokenize_texts(test_data['text'])
        
        # Convert labels to tensors
        train_labels = torch.tensor(train_data['label'].values, dtype=torch.long)
        val_labels = torch.tensor(val_data['label'].values, dtype=torch.long)
        test_labels = torch.tensor(test_data['label'].values, dtype=torch.long)
        
        # Save everything
        processed_data = {
            'train': {
                'input_ids': train_inputs['input_ids'],
                'attention_mask': train_inputs['attention_mask'],
                'labels': train_labels,
                'texts': train_data['text'].tolist()
            },
            'val': {
                'input_ids': val_inputs['input_ids'],
                'attention_mask': val_inputs['attention_mask'],
                'labels': val_labels,
                'texts': val_data['text'].tolist()
            },
            'test': {
                'input_ids': test_inputs['input_ids'],
                'attention_mask': test_inputs['attention_mask'],
                'labels': test_labels,
                'texts': test_data['text'].tolist()
            },
            'metadata': {
                'num_classes': len(self.categories),
                'categories': self.categories,
                'max_length': self.max_len,
                'tokenizer_name': self.tokenizer_name
            }
        }
        
        torch.save(processed_data, output_path)
        logger.info(f"Processed data saved successfully to {output_path}")
        
        return processed_data

# Convenience functions for backward compatibility
def load_and_split(csv_path: str, generate_synthetic: bool = True, synthetic_count: int = 1000):
    """Legacy function for backward compatibility."""
    processor = TicketDataProcessor()
    return processor.load_and_preprocess(csv_path, generate_synthetic, synthetic_count)

def tokenize_texts(texts: pd.Series, tokenizer_name: str = 'bert-base-uncased', max_len: int = 128):
    """Legacy function for backward compatibility."""
    processor = TicketDataProcessor(tokenizer_name=tokenizer_name, max_len=max_len)
    return processor.tokenize_texts(texts)

if __name__ == "__main__":
    # Example usage
    processor = TicketDataProcessor()
    
    # Load and preprocess data
    train, val, test = processor.load_and_preprocess("data/tickets.csv", 
                                                    generate_synthetic=True, 
                                                    synthetic_count=1000)
    
    # Save processed data
    processor.save_processed_data(train, val, test)
    
    print("Data preprocessing completed successfully!")
    print(f"Dataset sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
