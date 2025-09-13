"""
PERISCAN Main Execution Script
End-to-end pipeline for PERISCAN model training and evaluation.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PERISCANConfig, get_default_config
from utils import set_random_seed, create_directories, print_gpu_info, check_data_quality, plot_training_history
from data.preprocessing import preprocess_adata, check_data_integrity
from data.dataset import create_dataloaders, check_dataloader
from train import train_periscan_model
from evaluate import evaluate_trained_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PERISCAN: Single-cell cancer detection and classification')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--output_dir', type=str, default='./periscan_results',
                        help='Output directory for results')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for training')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Model arguments
    parser.add_argument('--gene_hidden_dim', type=int, default=256,
                        help='Hidden dimension for gene encoder')
    parser.add_argument('--min_genes', type=int, default=50,
                        help='Minimum genes per cell type')
    parser.add_argument('--max_genes', type=int, default=200,
                        help='Maximum genes per cell type')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--train_only', action='store_true',
                        help='Only train model, skip evaluation')
    parser.add_argument('--eval_only', type=str, default=None,
                        help='Only evaluate model using provided checkpoint path')
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment for training."""
    # Set random seed
    set_random_seed(args.seed)
    
    # Print system info
    print("=== PERISCAN Setup ===")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Check GPU availability
    print_gpu_info()
    
    # Create output directories
    create_directories(args.output_dir)
    
    return True


def create_config_from_args(args):
    """Create PERISCAN configuration from arguments."""
    config = get_default_config()
    
    # Update config with command line arguments
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs
    config.learning_rate = args.learning_rate
    config.early_stop_patience = args.early_stop_patience
    config.gene_hidden_dim = args.gene_hidden_dim
    config.min_genes_per_cell_type = args.min_genes
    config.max_genes_per_cell_type = args.max_genes
    config.num_workers = args.num_workers
    config.gene_selection_seed = args.seed
    config.save_dir = args.output_dir
    
    # Force CPU if GPU not requested
    if not args.gpu:
        config.device = 'cpu'
    
    print(f"\nUsing device: {config.device}")
    print(f"Configuration created with {config.max_epochs} max epochs")
    
    return config


def run_preprocessing(args, config):
    """Run data preprocessing pipeline."""
    print("\n" + "="*50)
    print("PREPROCESSING PHASE")
    print("="*50)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Run preprocessing
    adata, gene_lists, max_cells_dict = preprocess_adata(args.data_path, config)
    
    # Check data integrity
    check_data_integrity(adata, gene_lists)
    
    # Display data quality info
    check_data_quality(adata)
    
    print("\n✓ Preprocessing completed successfully")
    
    return adata, gene_lists, max_cells_dict


def run_training(adata, gene_lists, max_cells_dict, config):
    """Run model training pipeline."""
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders, datasets = create_dataloaders(adata, gene_lists, max_cells_dict, config)
    
    # Check dataloader functionality
    print("Checking dataloader functionality...")
    check_dataloader(dataloaders['train'], max_batches=1)
    check_dataloader(dataloaders['val'], max_batches=1)
    
    # Train model
    print(f"\nStarting training with the following configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")
    
    trainer, history = train_periscan_model(
        adata=adata,
        dataloaders=dataloaders,
        gene_lists=gene_lists,
        max_cells_dict=max_cells_dict,
        config=config
    )
    
    # Plot training history
    plot_path = os.path.join(config.save_dir, 'visualizations', 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    print("\n✓ Training completed successfully")
    
    return trainer, dataloaders, history


def run_evaluation(trainer, dataloaders, config, checkpoint_path=None):
    """Run model evaluation pipeline."""
    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)
    
    # Evaluate trained model
    evaluator, summary = evaluate_trained_model(
        trainer=trainer,
        dataloaders=dataloaders,
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    print("\n✓ Evaluation completed successfully")
    
    return evaluator, summary


def main():
    """Main execution function."""
    print("PERISCAN: Single-cell cancer detection and classification")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment(args)
    
    # Create configuration
    config = create_config_from_args(args)
    
    try:
        # Initialize variables
        trainer = None
        dataloaders = None
        
        # Evaluation only mode
        if args.eval_only:
            print("\n🔍 Running in evaluation-only mode")
            if not os.path.exists(args.eval_only):
                raise FileNotFoundError(f"Checkpoint file not found: {args.eval_only}")
            
            # Still need to preprocess data for evaluation
            adata, gene_lists, max_cells_dict = run_preprocessing(args, config)
            dataloaders, _ = create_dataloaders(adata, gene_lists, max_cells_dict, config)
            
            # Create a dummy trainer and load checkpoint
            trainer, _ = train_periscan_model(adata, dataloaders, gene_lists, max_cells_dict, config)
            
            # Run evaluation
            evaluator, summary = run_evaluation(trainer, dataloaders, config, args.eval_only)
        
        else:
            # Full pipeline or training only
            
            # Preprocessing
            adata, gene_lists, max_cells_dict = run_preprocessing(args, config)
            
            # Training
            trainer, dataloaders, history = run_training(adata, gene_lists, max_cells_dict, config)
            
            # Evaluation (unless train_only mode)
            if not args.train_only:
                evaluator, summary = run_evaluation(trainer, dataloaders, config)
            else:
                print("\n🚀 Training completed. Skipping evaluation (train_only mode)")
        
        # Final summary
        print("\n" + "="*60)
        print("PERISCAN EXECUTION COMPLETED")
        print("="*60)
        
        if trainer is not None:
            print(f"📁 Results saved to: {config.save_dir}")
            print(f"🎯 Best model: Epoch {trainer.best_epoch + 1}")
            print(f"📉 Best validation loss: {trainer.best_val_loss:.4f}")
        
        if not args.train_only and not args.eval_only:
            print(f"✅ Full pipeline executed successfully")
        elif args.train_only:
            print(f"🚀 Training pipeline executed successfully")
        else:
            print(f"🔍 Evaluation pipeline executed successfully")
            
        print("\nThank you for using PERISCAN! 🔬")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ Error occurred during execution:")
        print(f"Error: {str(e)}")
        print("\nFor debugging, check the following:")
        print("1. Input data file exists and is readable")
        print("2. Output directory has write permissions")
        print("3. Sufficient memory and disk space")
        print("4. All dependencies are installed correctly")
        
        # Print stack trace for debugging
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()