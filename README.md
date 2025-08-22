# DMM: Causal View of Time Series Imputation: Some Identification Results on Missing Mechanism
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

:triangular_flag_on_post:**News**(Jan 16, 2025): After the meeting, we will upload this paper to arXiv.

## Motivation
In real-world scenarios, different types of missing mechanisms, like MAR (Missing At Random), and MNAR (Missing Not At Random) can occur in time series data. However, existing methods often overlook the difference among the aforementioned missing mechanisms and use a single model for time series imputation, which can easily lead to misleading results due to mechanism mismatching. In this paper, we propose a framework for time series imputation problem by exploring **D**ifferent **M**issing **M**echanisms (**DMM** in short) as  shown in Figure 1 and tailoring solutions accordingly. 
<p align="center">
<img src=".\Image\Causal.jpg" alt="" align=center />
<br><br>
<b>Figure 1.</b> Data generation processes of time series data under different missing mechanisms. $z_t$ are temporal latent variables that describe the temporal dependencies. $x_t^o$ are the observed variables, $x_t^m$ are the missing data and $c_t$ denotes the missing cause variables. (a) The data generation process under the missing at random mechanism, where missingness is related to the observed data but not the unobserved data. (b) The data generation process under the missing not at random mechanism, where the missingness is influenced by the observed data and missing data in the previous time step. (c) The data generation process under the missing completely at random mechanism, where missing data is led by random issues, and the latent missing variables can be considered as random noises.
</p>

## Model
<p align="center">
<img src=".\Image\dmm_model.jpg" alt="" align=center />
<br><br>
<b>Figure 2.</b> Illustration of the DMM framework. $X^o$ are the observed variables, $X^m$ are the missing data. The latent state variables $z_{1:T}$ and the missing cause variables $c_{1:T}$ are extracted from the encoder. The latent state and missing cause prior networks for DMM-MAR and DMM-MNAR are used to estimate the prior distributions.

## Requirements

- Python 3.8
- torch == 2.3.1
- reformer-pytorch==1.4.4
- scikit-learn==1.2.2
- einops == 0.4.0
- tqdm == 4.64.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

You can obtain all datasets from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).
## Reproducibility

To easily reproduce the results you can run the following commands:
```
python run_DMM.py -data Exchange -mask_type MAR -mask_rate 0.2 -train_mode 0 -DMM_type MAR
```
Multiple seeds and datasets can be run at one time. The important parameters are in file DMM_config.py and you can go inside to change the parameters you want.

And we provide explanations for the important parameters:
| Parameter name | Description of parameter |
| --- | --- |
| data           | The dataset name                                             |
| root_path      | The root path of the data file (defaults to `./data/exchange_rate/`)    |
| data_path      | The data file name (defaults to `exchange_rate.csv`)                  |
| seq_len | Input sequence length (defaults to 96) |
| mask_type | Dataset missing mechanism |
| mask_rate | Dataset missing rate |
| train_mode | Training methods |
| d_model | Dimension of model |
| train_epochs | Epochs in train |
| learning_rate | Optimizer learning rate |

More parameter information please refer to `run.py`.

## <span id="resultslink">Results</span>

The main results are shown in table 1 and table 2.

<p align="center">
<img src=".\Image\MAR Experiment Result.png" alt="" align=center />
<br><br>
</p>
<p align="center">
<img src=".\Image\MNAR Experiment Result.png" alt="" align=center />
<br><br>
</p>

## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following papers:

```
To be continued...
```
