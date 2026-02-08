# Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification

This repository contains the official implementation for the ICLR 2026 paper {[Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](https://openreview.net/forum?id=xJT4fXJr1Q)}.

## Hardware Dependencies

Hardware requirements for Evidential Uncertainty Quantification (EUQ) are consistent with standard large vision-language model inference. Any system capable of running general LVLM tasks is sufficient for this codebase.

Specific requirements depend on the model size. 7B models require 24 GB VRAM. 13B models require NVIDIA A100 level hardware. 70B models require dual NVIDIA A100 (80 GB) GPUs. The implementation supports FP16 and INT8 quantization to minimize memory overhead.
## Installation

### Clone the Repository

```
git clone https://github.com/your-username/EUQ.git
cd EUQ
```

### Create and Activate Conda Environment

```
conda create -n EUQ python=3.10 -y
conda activate EUQ
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Download Datasets

This repository provides only a toy dataset.
Please download the complete dataset from [Misbehavior-Bench](https://huggingface.co/datasets/thuang5288/Misbehavior-Bench).
Organize the downloaded files according to the following directory structure.

```text
EUQ/
└── datasets/
    └── hallucination/
        ├── hallucination.csv
        └── images/
            ├── COCO_val2014_000000108681.jpg
            ├── COCO_val2014_000000377730.jpg
            └── COCO_val2014_000000563927.jpg

```
### Prepare LVLMs weights

Users must obtain the head weights of LVLMs as `.pth` files.
Place these files in the `EUQ/weights` directory. The **EUQ** framework requires these weights for model evaluation.

```text
EUQ/
└── weights/
    ├── lvlm_head_01.pth
    └── ...
```




### Run the Experiments

```
cd script
bash auto_generation.sh
```

Experiments utilize [Weights & Biases (wandb)](https://wandb.ai/) for logging.
Authenticate with your API key during the initial run.
Users must [apply for access](https://huggingface.co/meta-llama) to use Llama-2 models via Hugging Face.


<!-- For almost all tasks, the dataset is downloaded automatically from the Hugging Face Datasets library upon first execution.
The only exception is BioASQ (task b, BioASQ11, 2023), for which the data needs to be [downloaded](http://participants-area.bioasq.org/datasets) manually and stored at `$SCRATCH_DIR/$USER/semantic_uncertainty/data/bioasq/training11b.json`, where `$SCRATCH_DIR` defaults to `.`. -->



## Quick start

Execute the following commands to reproduce results.
This process evaluates Qwen2_5_VL on the hallucination detection task.

```
cd script
bash auto_generate.sh
```


## Further Instructions

### Repository Structure

We here give an overview of the various components of the code.

By default, a standard run executes the following three scripts in order:

* `generate_answers.py`: Sample responses (and their likelihods/hidden states) from the models for a set of input questions.
* `compute_uncertainty_measures.py`: Compute uncertainty metrics given responses.
* `analyze_results.py`: Compute aggregate performance metrics given uncertainties.

Scripts can be executed individually for recomputing results.

<!-- ### Reproducing the Experiments

To reproduce the experiments of the paper, one needs to execute

```
python generate_answers.py --model_name=$MODEL --dataset=$DATASET $EXTRA_CFG
```

for all combinations of models and datasets, and where `$EXTRA_CFG` is defined to either activate short-phrase or sentence-length generations and their associated hyperparameters.

The results for any run can be obtained by passing the associated `wandb_id` to an evaluation notebook identical to the demo in [notebooks/example_evaluation.ipynb](notebooks/example_evaluation.ipynb). -->
## Citation
Please cite our work if it helps your research.
```bibtex
@inproceedings{huang2026detecting,
  title     = {Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification},
  author    = {Huang, Tao and Wang, Rui and Liu, Xiaofei and Qin, Yi and Duan, Li and Jing, Liping},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
