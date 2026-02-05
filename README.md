# Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification

This repository contains the official implementation for the ICLR 2026 paper 'Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification'.

## Hardware Dependencies

Generally speaking, our experiments require modern computer hardware which is suited for usage with large language models (LLMs).

Requirements regarding the system's CPU and RAM size are relatively modest: any reasonably modern system should suffice, e.g. a system with an Intel 10th generation CPU and 16 GB of system memory or better.

More importantly, all our experiments make use of one or more Graphics Processor Units (GPUs) to speed up LLM inference.
Without a GPU, it is not feasible to reproduce our results in a reasonable amount of time.
The particular GPU necessary depends on the choice of LLM: LLMs with more parameters require GPUs with more memory.
For smaller models (7B parameters), desktop GPUs such as the Nvidia TitanRTX (24 GB) are sufficient.
For larger models (13B), GPUs with more memory, such as the Nvidia A100 server GPU, are required.
Our largest models with 70B parameters require the use of two Nvidia A100 GPUs (2x80GB) simultaneously.

One can reduce the precision to float16 or int8 to reduce memory requirements without significantly affecting model predictions and their accuracy.
We use float16 for 70B models by default, and int8 mode can be enabled for any model by suffixing the model name with `-int8`.

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

```
'下载数据集到datasets目录下'
```

### Run the Experiments

```
'确保script中的数据集路径存在'
cd script
bash auto_generation.sh
```

Our experiments rely on [Weights & Biases (wandb)](https://wandb.ai/) to log and save individual runs.
While wandb will be installed automatically with the above conda script, you may need to log in with your wandb API key upon initial execution.

Our experiments rely on Hugging Face for all LLM models and most of the datasets.
It may be necessary to set the environment variable `HUGGING_FACE_HUB_TOKEN` to the token associated with your Hugging Face account.
Further, it may be necessary to [apply for access](https://huggingface.co/meta-llama) to use the official repository of Meta's LLaMa-2 models.
We further recommend setting the `XDG_CACHE_HOME` environment variable to a directory on a device with sufficient space, as models and datasets will be downloaded to this folder.


For almost all tasks, the dataset is downloaded automatically from the Hugging Face Datasets library upon first execution.
The only exception is BioASQ (task b, BioASQ11, 2023), for which the data needs to be [downloaded](http://participants-area.bioasq.org/datasets) manually and stored at `$SCRATCH_DIR/$USER/semantic_uncertainty/data/bioasq/training11b.json`, where `$SCRATCH_DIR` defaults to `.`.



## Quick start

Execute

```
cd script
bash auto_generate.sh
```

to reproduce results for short-phrase generation with LLaMa-2 Chat (7B) on the BioASQ dataset.

The expected runtime of this demo is 1 hour using an Nvidia A100 GPU (80 GB), 24 cores of a Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz, and 192 GB of RAM.
Runtime may be longer upon first execution, as the LLM needs to be downloaded from Hugging Face first.

To evaluate the run and obtain a barplot similar to those of the paper, open the Jupyter notebook in [notebooks/example_evaluation.ipynb](notebooks/example_evaluation.ipynb), populate the `wandb_id` variable in the first cell with the id assigned to your demo run, and execute all cells of the notebook.


We refer to [https://jupyter.org/](https://jupyter.org/) for more information on how to start the Jupter notebook server.


## Further Instructions

### Repository Structure

We here give an overview of the various components of the code.

By default, a standard run executes the following three scripts in order:

* `generate_answers.py`: Sample responses (and their likelihods/hidden states) from the models for a set of input questions.
* `compute_uncertainty_measures.py`: Compute uncertainty metrics given responses.
* `analyze_results.py`: Compute aggregate performance metrics given uncertainties.

It is possible to run these scripts individually, e.g. when recomputing results, and we are happy to provide guidance on how to do so upon request.


### Reproducing the Experiments

To reproduce the experiments of the paper, one needs to execute

```
python generate_answers.py --model_name=$MODEL --dataset=$DATASET $EXTRA_CFG
```

for all combinations of models and datasets, and where `$EXTRA_CFG` is defined to either activate short-phrase or sentence-length generations and their associated hyperparameters.

The results for any run can be obtained by passing the associated `wandb_id` to an evaluation notebook identical to the demo in [notebooks/example_evaluation.ipynb](notebooks/example_evaluation.ipynb).
