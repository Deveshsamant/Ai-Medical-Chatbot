# Medical Chatbot QLoRA Fine-Tuning

A complete pipeline for fine-tuning Llama 2 7B on medical Q&A data using QLoRA (Quantized Low-Rank Adaptation) optimized for consumer GPUs.

## 🎯 Features

- **Multiple Datasets**: Combines HealthCareMagic-100k, iCliniq, and custom medical data
- **QLoRA Training**: Parameter-efficient fine-tuning with LoRA adapters
- **Easy to Use**: Simple scripts for preprocessing, training, and inference
- **Production Ready**: Includes evaluation, checkpointing, and monitoring
- **Local Execution**: Runs entirely locally on your laptop
- **Privacy Focused**: No API keys required, all data stays on your machine

##  Dataset

The dataset used for this project is available on Kaggle:
**[Chat Medic Dataset](https://www.kaggle.com/datasets/deveshsamant/chat-medic)**

## 📋 Requirements

### Hardware
### Hardware
- **Device**: Capable of running local LLMs (e.g., Laptop with dedicated GPU recommended but not required for inference)
- **Storage**: Sufficient space for model and checkpoints

### Software
- Python 3.8+
- CUDA 11.8+ or 12.x
- PyTorch 2.0+

## 🚀 Quick Start

### 1. Reconstruct ChromaDB Database

After cloning the repository, you need to reconstruct the ChromaDB database from its chunks:

```powershell
# Reconstruct the database (merges 3 chunks into chroma.sqlite3)
python reconstruct_file.py
```

This will create the `parquet cromadb/chroma.sqlite3` file (2.56 GB) from the split chunks that are tracked with Git LFS.

### 2. Download Local LLM Model

Download the TinyLlama model for local inference (no API keys required):

```powershell
# Navigate to backend directory
cd backend

# Download the model (638 MB)
python download_tinyllama.py

# Return to project root
cd ..
```

This downloads `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` to `backend/models/` for fully local, GPU-accelerated inference.

### 3. Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 4. Preprocess Data

```powershell
# Process all medical datasets
python scripts/preprocess_data.py

# Or validate first without saving
python scripts/preprocess_data.py --validate

# Process with limited samples for testing
python scripts/preprocess_data.py --max_samples 1000
```

This will combine:
- **HealthCareMagic-100k**: 112k patient-doctor Q&A pairs
- **iCliniq**: 7.3k clinical consultations
- **fine_tune_data**: Medical knowledge Q&A

Output: `data/processed/train.jsonl` and `data/processed/val.jsonl`

### 5. Configure Training

Edit `configs/qlora_config.yaml` to customize:
- Model path (if using local Llama 2)
- Batch size and learning rate
- Number of epochs
- LoRA parameters

**Important**: If you don't have Llama 2 downloaded, you'll need to:
1. Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Login with: `huggingface-cli login`
3. Or download the model locally and update `model_name` in config

### 6. Train the Model

```powershell
# Single GPU training (RTX 3050)
.\train_single_gpu.ps1

# Or run directly
python train_qlora.py --config configs/qlora_config.yaml
```

**Training Time**: ~8-12 hours for 3 epochs on RTX 3050 6GB with ~120k samples

**Monitor Training**:
```powershell
# In a separate terminal
tensorboard --logdir logs
```
Then open http://localhost:6006

### 7. Run Inference

```powershell
# Interactive chat mode
python inference.py --adapter_path checkpoints/llama2-7b-medical-qlora

# Batch inference with example questions
python inference.py --mode batch

# Custom questions
python inference.py --mode batch --questions "What causes diabetes?" "How to treat hypertension?"
```

## 📁 Project Structure

```
Medical cahtbot/
├── json_data/                      # Raw datasets
│   ├── HealthCareMagic-100k.json
│   ├── iCliniq.json
│   └── fine_tune_data.jsonl
├── data/
│   └── processed/                  # Preprocessed data
│       ├── train.jsonl
│       ├── val.jsonl
│       └── dataset_stats.json
├── configs/
│   └── qlora_config.yaml          # Training configuration
├── scripts/
│   └── preprocess_data.py         # Data preprocessing
├── checkpoints/                    # Model checkpoints
│   └── llama2-7b-medical-qlora/
├── logs/                           # Training logs
├── train_qlora.py                 # Main training script
├── inference.py                   # Inference script
├── train_single_gpu.ps1           # Training launcher
└── requirements.txt               # Python dependencies
```

## ⚙️ Configuration Details

### Memory Optimization for RTX 3050 6GB

The configuration is optimized for 6GB VRAM:

- **4-bit Quantization**: Reduces model size from ~14GB to ~4GB
- **Gradient Checkpointing**: Trades compute for memory
- **Batch Size 1**: With gradient accumulation of 8 steps
- **Max Sequence Length 512**: Shorter sequences use less memory
- **Paged AdamW 8-bit**: Memory-efficient optimizer

### LoRA Parameters

- **Rank (r)**: 16 - Balance between performance and efficiency
- **Alpha**: 32 - Scaling factor (typically 2*r)
- **Target Modules**: All attention and MLP layers
- **Dropout**: 0.05 - Regularization

## 📊 Expected Results

### Training Metrics
- **Initial Loss**: ~2.5-3.0
- **Final Loss**: ~0.8-1.2 (after 3 epochs)
- **Validation Loss**: ~1.0-1.5

### GPU Memory Usage
- **Model Loading**: ~4.5 GB
- **During Training**: ~5.5-5.8 GB
- **Peak Usage**: ~5.9 GB

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:

1. **Reduce sequence length**:
   ```yaml
   max_seq_length: 384  # or 256
   ```

2. **Increase gradient accumulation**:
   ```yaml
   gradient_accumulation_steps: 16
   ```

3. **Disable evaluation during training**:
   ```yaml
   evaluation_strategy: "no"
   ```

4. **Use FP16 instead of BF16** (if BF16 causes issues):
   ```yaml
   bf16: false
   fp16: true
   ```

### Slow Training

- **Enable Flash Attention 2** (if supported):
  ```bash
  pip install flash-attn --no-build-isolation
  ```

- **Reduce logging frequency**:
  ```yaml
  logging_steps: 50
  ```

### Model Access Issues

If you can't access Llama 2:
1. Request access at HuggingFace
2. Login: `huggingface-cli login`
3. Or use an alternative model like `mistralai/Mistral-7B-v0.1`

## 📈 Monitoring Training

### TensorBoard

```powershell
tensorboard --logdir logs
```

Metrics to watch:
- **train/loss**: Should decrease steadily
- **eval/loss**: Should decrease but may plateau
- **train/learning_rate**: Should follow cosine schedule

### Checkpoints

Checkpoints are saved every 100 steps in `checkpoints/llama2-7b-medical-qlora/`:
- Only the 3 best checkpoints are kept (by validation loss)
- Each checkpoint contains LoRA adapters (~100-200 MB)

## 🎓 Advanced Usage

### Resume Training

```powershell
python train_qlora.py --config configs/qlora_config.yaml --resume_from_checkpoint checkpoints/llama2-7b-medical-qlora/checkpoint-500
```

### Custom Dataset

To use your own medical data:

1. Format as JSONL with `{"text": "<s>[INST] question [/INST] answer </s>"}`
2. Update paths in `configs/qlora_config.yaml`
3. Run training

### Merge LoRA Adapters

To create a standalone model (without needing base model + adapters):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "checkpoints/llama2-7b-medical-qlora")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_medical_llama2")
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{medical_chatbot_qlora,
  title = {Medical Chatbot QLoRA Fine-Tuning},
  year = {2025},
  note = {QLoRA fine-tuning pipeline for medical chatbots}
}
```

## 📄 License

This project is for educational and research purposes. Please ensure you comply with:
- Llama 2 License Agreement
- Dataset licenses (HealthCareMagic, iCliniq)
- Medical AI regulations in your jurisdiction

## ⚠️ Disclaimer

This chatbot is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for larger models (13B, 70B)
- Multi-GPU training with DeepSpeed
- Evaluation metrics (BLEU, ROUGE, medical accuracy)
- Web interface for the chatbot
- Fine-tuning on specialized medical domains

## 📧 Support

For issues and questions:
1. Check the troubleshooting section
2. Review training logs in `logs/`
3. Check GPU memory with `nvidia-smi`

---

**Happy Training! 🚀🏥**

## 👨‍💻 Developer

- **Name**: Devesh Samant
- **Date**: 2025-11-29
