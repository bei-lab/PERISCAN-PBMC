import os

# 定义项目结构
project_structure = {
    'data': ['raw/', 'processed/', 'external/', 'README.md'],
    'src': {
        'models': ['__init__.py', 'periscan.py', 'layers.py', 'README.md'],
        'preprocessing': ['__init__.py', 'data_loader.py', 'feature_selection.py', 'README.md'],
        'training': ['__init__.py', 'train.py', 'validation.py', 'README.md'],
        'evaluation': ['__init__.py', 'metrics.py', 'performance.py', 'README.md'],
        'interpretability': ['__init__.py', 'feature_distillation.py', 'analysis.py', 'README.md'],
        'utils': ['__init__.py', 'config.py', 'logger.py', 'helpers.py']
    },
    'configs': ['model_config.yaml', 'training_config.yaml', 'data_config.yaml'],
    'scripts': ['train_model.py', 'evaluate_model.py', 'run_analysis.py'],
    'notebooks': ['data_exploration.ipynb', 'model_analysis.ipynb'],
    'tests': ['__init__.py', 'test_models.py', 'test_preprocessing.py'],
    'docs': ['API.md', 'TUTORIAL.md'],
    'results': ['figures/', 'tables/', 'models/', 'README.md']
}

def create_structure(base_path, structure):
    for item, content in structure.items():
        if isinstance(content, dict):
            # 创建目录
            dir_path = os.path.join(base_path, item)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
            # 递归创建子结构
            create_structure(dir_path, content)
        elif isinstance(content, list):
            # 创建目录和文件
            dir_path = os.path.join(base_path, item)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
            for file_name in content:
                if file_name.endswith('/'):
                    # 子目录
                    sub_dir = os.path.join(dir_path, file_name)
                    os.makedirs(sub_dir, exist_ok=True)
                    print(f"Created subdirectory: {sub_dir}")
                else:
                    # 文件
                    file_path = os.path.join(dir_path, file_name)
                    with open(file_path, 'w') as f:
                        if file_name.endswith('.md'):
                            f.write(f"# {file_name.replace('.md', '').replace('_', ' ').title()}\n\nTODO: Add documentation\n")
                        elif file_name.endswith('.py'):
                            f.write(f'"""\n{file_name}\nTODO: Add implementation\n"""\n')
                        elif file_name.endswith('.yaml'):
                            f.write(f"# {file_name} configuration\n# TODO: Add configuration parameters\n")
                    print(f"Created file: {file_path}")

if __name__ == "__main__":
    base_path = "."  # 当前目录
    create_structure(base_path, project_structure)
    
    # 创建根目录文件
    root_files = {
        'README.md': '''# PERISCAN: Peripheral Immune Dynamics for Cancer Detection

A transformer-based deep learning framework for cancer detection using single-cell immune dynamics in peripheral blood.

## Overview
TODO: Add project overview

## Installation
TODO: Add installation instructions

## Usage
TODO: Add usage instructions

## Citation
TODO: Add citation information
''',
        'requirements.txt': '''torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scanpy>=1.8.0
anndata>=0.8.0
pyyaml>=5.4.0
tqdm>=4.62.0
jupyter>=1.0.0
''',
        '.gitignore': '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Data files
*.h5
*.h5ad
*.csv
*.xlsx
data/raw/
data/processed/
!data/external/sample_data.csv

# Models and results
results/models/*.pkl
results/models/*.pth
*.model

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Logs
logs/
*.log
''',
        'LICENSE': '''MIT License

Copyright (c) 2025 [Your Institution]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
    }
    
    for filename, content in root_files.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Created file: {filename}")
    
    print("\n✅ Project structure created successfully!")
    print("Next steps:")
    print("1. Review and customize the created files")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start developing your PERISCAN model!")