
# MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries

![MedDec](assets/figure.png)

> [!TIP]
> This repository is a **contribution** to the code described in **[MedDec (Elgaar et al., Findings of ACL: ACL 2024)](https://aclanthology.org/2024.findings-acl.975/)**.

MedDec is the first dataset specifically developed for extracting and classifying medical decisions from clinical notes. It includes 451 expert-annotated annotated discharge summaries from the MIMIC-III dataset, offering a valuable resource for understanding and facilitating clinical decision-making.


---


# Contributions to MedDec 📝

1. Testing of RoBERTa and Binder models and analysis of their results.
2. Testing of new models, such as ModernBERT and Distill-BioBERT.
3. Implementation of ensemble modeling. 

The user now had the possibility to select multiple models at once to have a new stacked model making the predictions. 
The user can do ensemble modeling by adding the flag:
  ```bash
   --stacked <model_1> <model_2> <model_n> 
  ```
Final analysis:
We confirm good performances of RoBERTa, especially for F1 score, but detected the potential of ModernBERT to surpass it, as it aleady has better accuracy score. 
Esemble modelling has high potentia l as well for better stability and generalization, but needs further implementation improvements to be a real asset. 


---


# Quickstart 🚀

## Install dependencies 📦

Follow these steps to set up your environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/[your-repository-url]
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Download Data 📥

> [!TIP]
> The dataset has been released as of October 16, 2024.

To get started, you need to download the necessary datasets:

1. **Annotations: data/*.json**: Available [here](https://physionet.org/content/meddec/1.0.0/). 
2. **Phenotype Data: ACTdb102003.csv**: Available [here](https://physionet.org/content/phenotype-annotations-mimic/1.20.03/).
3. **Discharge Summaries: NOTEEVENTS.csv**: Available [here](https://physionet.org/content/mimiciii/1.4/).
You must sign a data usage agreement before accessing the dataset.




## Repository Structure 📂

Your repository should have the following structure:

```
├── .venv/                         # Virtual environment directory
├── assets/                         # Directory for additional assets (e.g., models, resources)
├── data_dir/                       # Main directory for data
│   ├── data/                       # Folder containing dataset files
│   │   └── *.json                  # JSON files (e.g., clinical data)
│   ├── ACTdb102003.csv             # Phenotype annotations
│   └── NOTEEVENTS.csv              # MIMIC-III discharge summaries
├── data.py                         # Data loading and processing
├── extract_texts.py                # Script to extract clinical notes text
├── main.py                         # Main script for training and evaluation
├── model.py                        # Model definition and configuration
├── options.py                      # Command-line argument parsing
└── preprocess_phenos.py            # Script for preprocessing phenotype annotations
```



## Preprocess Data 🔄

### 1. Extract Notes Text from MIMIC-III 📝

To extract the clinical notes text from the MIMIC-III dataset, run the following command:

```bash
python extract_notes.py <data_dir> <notes_path (NOTEEVENTS.csv)>
```

- **data_dir**: Directory where the dataset is stored.
- **notes_path**: Path to the `NOTEEVENTS.csv` file.

The extracted text will be saved in the `data_dir/raw_text` directory.


### 2. Aggregate Phenotype Annotations 🧬

To preprocess the phenotype annotations, run the following command:

```bash
python preprocess_phenotypes.py <phenotypes_path (ACTdb102003.csv)>
```

- **phenotypes_path**: Path to the phenotype annotations file.

The aggregated annotations will be saved in `phenos.csv` in the same directory as the input file.



## Run Models ⚙️

The code expects the following directories and files:
- **data_dir**: The directory containing the dataset.
- **raw_text**: The directory containing the extracted notes.
- **phenos.csv**: The phenotype annotations file.

### Train the Baselines 🏋️‍♂️

To train the baseline models, use the following command:

```bash
python main.py --data_dir data_dir/ --label_encoding multiclass --model_name google/electra-base-discriminator --total_steps 2001 --lr 4e-5
```

To train using ensemble modeling, use the following command:

```bash
python main.py --data_dir data_dir/ --label_encoding multiclass --stacked nlpie/distil-biobert answerdotai/ModernBERT-base --total_steps 2001 --lr 4e-5
```

### Evaluate the Baselines 📊

To evaluate the baseline models, run this command:

```bash
python main.py --data_dir <data_dir> --eval_only --ckpt ./checkpoints/[datetime]-[model_name]
```



## Arguments 📝

Here’s an overview of the available command-line arguments:

- **data_dir**: The directory where the dataset is stored. *(Default: `./data/`)*
- **pheno_path**: Path to the phenotype annotations file. *(Default: `./ACTdb102003.csv`)*
- **task**: Choose between `token` (for decision extraction) or `seq` (for phenotype prediction). *(Default: `token`)*
- **eval_only**: If set, the model will only evaluate without training. *(Default: `False`)*
- **label_encoding**: Choose from `multiclass`, `bo` (beginning inside outside), or `boe` (beginning outside end). *(Default: `multiclass`)*
- **truncate_train**: If set, sequences will be truncated to a maximum length. *(Default: `False`)*
- **truncate_eval**: If set, evaluation sequences will be truncated to a maximum length. *(Default: `False`)*
- **use_crf**: Whether to use a CRF layer for token classification. *(Default: `False`)*
- **model_name**: The name of the model from Hugging Face Transformers.
- **total_steps**: The number of training steps.
- **lr**: The learning rate.
- **batch_size**: The batch size.
- **seed**: The random seed.


With these steps and arguments, you should be able to quickly set up, preprocess the data, and run the models! 🎉

---

# Code Explanation 🧑‍💻



## `options.py` 📝

This file is responsible for defining and parsing command-line arguments for configuring the MedDec project. It allows users to specify various hyperparameters, dataset paths, model choices, and training options dynamically when running the code.

### **Data and Checkpoints**
- `--data_dir`: Default path to the dataset (`./data/`).
- `--ckpt`: Path to the model checkpoint (used for loading pre-trained models).
- `--ckpt_dir`: Automatically set to a directory inside `/checkpoints/` based on timestamp, model name, and architecture.

### **Experiment Logging**
- `--aim_repo`: Path for logging experiments (default: current directory).
- `--aim_exp`: Experiment name for tracking (default: `'mimic-decisions-1215'`).

### **Model and Task Settings**
- `--task`: Defines whether the task is:
  - `'seq'`: Sequence classification.
  - `'token'`: Token classification (default).
- `--label_encoding`: Defines the type of label encoding:
  - `'multiclass'`: Expands decision labels to `num_labels = num_decs * 2 + 1`.
  - `'bo'`: Binary encoding (Begin-Other) → `num_labels *= 2`.
  - `'boe'`: Binary encoding (Begin-Other-End) → `num_labels *= 3`.
- `--max_len`: Maximum token length for input sequences (default: 512).
- `--model`: Model backbone (default: `'roberta-base'`).
- `--model_name`: Pretrained model identifier (default: `'google/electra-base-discriminator'`).
- `--stacked`: Allows specifying multiple models as a list.

### **GPU and Computation** 💻
- `--gpu`: Specifies which GPU to use (`'0'` by default).
- `--grad_accumulation`: Number of gradient accumulation steps (default: 2).

### **Training and Evaluation** 🏋️‍♀️
- `--total_steps`: Total training steps (default: 5000).
- `--train_log`: Interval (in steps) for logging training progress.
- `--val_log`: Interval (in steps) for logging validation progress.
- `--batch_size`: Batch size (default: 8).
- `--lr`: Learning rate (default: `4e-5`).
- `--pos_weight`: Class imbalance weight (default: 1.0).

### **Pheno & Decision Settings**
- `--num_phenos`: Number of phenotypes (10 by default).
- `--num_decs`: Number of decisions (9 by default).
- `--pheno_id`: If specified, restricts training to a single phenotype.
- `--unseen_pheno`: If specified, excludes a phenotype from training.

### **Training Controls** ⏳
- `--truncate_train`: If set, truncates training data.
- `--truncate_eval`: If set, truncates evaluation data.
- `--load_ckpt`: If set, loads a checkpoint.
- `--eval_only`: If set, runs evaluation without training.
- `--resample`: Specifies resampling method.

### **Miscellaneous**
- `--debug`: Enables debug mode.
- `--save_losses`: Saves training loss values.
- `--verbose`: Enables verbose logging (True by default).
- `--use_crf`: Enables Conditional Random Fields (CRF) for token classification.

### **Setting `args.num_labels` Based on Task:**

- **For sequence classification (`seq`)**:
  - If `pheno_id` is set → `num_labels = 1`.
  - Otherwise → `num_labels = num_phenos (10)`.

- **For token classification (`token`)**:
  - `num_labels = num_decs (9 by default)`.
  - Adjusts `num_labels` based on `label_encoding` (`multiclass`, `bo`, `boe`).



## `extract_texts.py` 📜

This script extracts and saves raw discharge summaries (clinical notes) from a structured dataset. It takes JSON files containing metadata and matches them with clinical note entries from a CSV file. The extracted text is stored as `.txt` files in a directory.

### **1. Script Purpose**
- Reads JSON metadata files from a dataset directory.
- Loads a CSV file containing discharge summaries (clinical notes).
- Matches each JSON file to its corresponding note using patient identifiers.
- Saves the extracted text as individual `.txt` files in a `raw_text/` directory.

### **2. Command-line Arguments**
The script expects two arguments:
- `<data_dir>` → Path to the dataset directory.
- `<notes_path>` → Path to the CSV file containing clinical notes.

#### **Usage Example**
```bash
python extract_texts.py "data_dir/" "data_dir/NOTEEVENTS.csv"
```

This means:
- The dataset JSON files are inside `"data_dir/data/*.json"`.
- The CSV file containing notes is `"data_dir/NOTEEVENTS.csv"`.
- The extracted texts will be stored in `"data_dir/raw_text/"`.



## `preprocess_phenos.py` 🔄

This script processes phenotype annotations from a dataset, handling multiple annotations per patient and resolving conflicts based on a predefined priority order. The final output is a cleaned `phenos.csv` file containing the aggregated phenotype labels.

### **1. Purpose**
- Aggregates phenotype annotations for the same patient (`SUBJECT_ID`, `HADM_ID`, `ROW_ID`), by loading `ACTdb102003.csv` (or another input CSV file).
- Handles conflicting annotations by prioritizing specific annotators (DAG, PAT, etc.).
- Generates final phenotype labels as a comma-separated string (or `?` if uncertain).
- Saves the processed data to `phenos.csv` for use in downstream analysis.

### **2. Relationship to Other Files**
- **Links to `extract_texts.py`**: `extract_texts.py` generates `raw_text/*.txt` files from clinical notes. `preprocess_phenos.py` processes phenotype annotations, creating `phenos.csv`, which pairs with these text files.



## `data.py` 📊

This file, `data.py`, is responsible for handling data loading and processing for the MedDec project.

### **1. Purpose**
- Defines mappings for phenotype categories.
- Splits the dataset into training, validation, and test sets.
- Implements a PyTorch dataset (`MyDataset`) for loading and tokenizing medical text data.
- Loads phenotype annotations from JSON files.
- Handles different label encoding schemes.
- Provides utility functions for parsing category labels and loading phenotype data.

### **2. Key Components**

#### **Phenotype Mapping**:
- The `pheno_map` assigns numerical IDs to phenotype categories.
- The `rev_pheno_map` allows retrieving the category name from its numeric ID.

#### **Dataset Splitting**:
- Splits subjects into train (80%), validation (10%), and test (10%) sets.
- If `args.unseen_pheno` is specified, it ensures that phenotype is only in the test set.
- Works differently depending on `args.task`:
  - If `token`, it loads from JSON files.
  - Otherwise, it uses phenotype data (`phenos.csv`).
- Outputs lists of file paths or DataFrame subsets.

#### **PyTorch Dataset Class**:
- Implements a dataset for token classification (`token`) or phenotype prediction (`seq`).
- Loads and tokenizes text data using a Hugging Face tokenizer.
- Stores tokenized input IDs, phenotype labels, and token masks.



## `model.py` 🏗️

This `model.py` file defines a transformer-based classification model (`MyModel`) and a function (`load_model`) to initialize and configure it.

### **1. Purpose**
The `model.py` file defines the core deep learning model used in MedDec for medical text classification. It provides a flexible architecture that can work with different types of models (LSTM, CNN, or transformer-based) to process discharge summaries and extract relevant clinical phenotypes.

Its main objectives include:
- Using a pretrained transformer model (e.g., BERT, Electra) as a backbone for text representation.
- Supporting sequence-level and token-level classification, allowing both document-level and word-level predictions.
- Handling long clinical documents by splitting them into manageable segments while maintaining context.
- Providing an adaptable model loading mechanism that integrates with MedDec’s training and evaluation pipeline.

### **2. Link to MedDec and Other Files**
- **Training & Evaluation**: This file is critical for MedDec’s training scripts, as `load_model()` initializes the required model type and configuration.
- **Clinical Phenotype Extraction**: The `phenos()` function is likely used to identify specific medical conditions in patient discharge summaries.
- **Handling Long Texts**: Since medical texts are lengthy, `generate()` ensures that entire documents can be processed efficiently without exceeding memory limits.
- **Checkpoint Loading**: If a pre-trained MedDec model is available, `load_model()` allows resuming training or running inference on new data.

#### **Links to Other Files**:
- Works with `train.py` and `evaluate.py` for training and testing the model.
- Uses `args.model_name` to dynamically select the backbone, likely corresponding to a pre-configured transformer model used in MedDec.



## `main.py` 🚀

### **1. Purpose**
The `main.py` script serves as the primary entry point for evaluating and processing clinical text using MedDec. It facilitates model evaluation, metrics computation, and text processing by loading data, applying models, and generating structured outputs.

### **2. Code Breakdown (Main Functions)**
- **`indicators_to_spans(labels, idx=None)`**: Converts label indicators into spans for entity extraction.
- **`id_to_label(labels)`**: Maps label IDs to human-readable labels (BIO format).
- **`f1_score(ys, preds), recall_score(ys, preds)`**: Computes evaluation metrics for model performance.
- **`calc_metrics_spans(ys, preds, span_ys=None)`**: Calculates F1 scores and other metrics for predicted spans.
- **`save_losses(model, crit, train_dataloader, val_dataloader, test_dataloader)`**: Saves training, validation, and test losses.
- **`evaluate(models, dataloaders, return_losses=False, return_preds=False)`**: Evaluates models on datasets, supporting ensemble prediction.
- **`process(sample, model, tokenizer, out_dir)`**: Processes a given sample, extracts text spans, and saves results as JSON files.

### **3. Relationships with MedDec and Other Files**
- **`data.py` (load_data)**: Handles loading and preprocessing of datasets.
- **`model.py` (load_model)**: Loads the appropriate model for evaluation and prediction.
- **`options.py` (get_args)**: Parses command-line arguments for configuration.
- **External Dependencies**: Utilizes `torch`, `numpy`, and `pandas` for model processing and numerical computations.

The `main.py` script integrates these components to ensure smooth execution of model inference, loss tracking, and evaluation for the MedDec framework.

---

# Citation 📚

If you use this dataset or code, please consider citing the following paper:

```bibtex
@inproceedings{elgaar-etal-2024-meddec,
    title = "{M}ed{D}ec: A Dataset for Extracting Medical Decisions from Discharge Summaries",
    author = "Elgaar, Mohamed and Cheng, Jiali and Vakil, Nidhi and Amiri, Hadi and Celi, Leo Anthony",
    editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.975",
    pages = "16442--16455",
}
```

Additionally, please cite the dataset as follows:

```bibtex
@misc{elgaar2024meddec,
    title = "MedDec: Medical Decisions for Discharge Summaries in the MIMIC-III Database",
    author = "Elgaar, Mohamed and Cheng, Jiali and Vakil, Nidhi and Amiri, Hadi and Celi, Leo Anthony",
    year = "2024",
    version = "1.0.0",
    publisher = "PhysioNet",
    url = "https://doi.org/10.13026/nqnw-7d62"
}
```