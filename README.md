# Automatic Noisy Label Correction for Fine-Grained Entity Typing

This is the source code for IJCAI 2022 paper: *Automatic Noisy Label Correction for Fine-Grained Entity Typing*.

### Requirements
```shell script
conda create -n FET36 -y python=3.6 && conda activate FET36
pip install -r requirements.txt
```

### Datasets
We use the Ultra-Fine and OntoNotes datasets released by <a href="http://nlp.cs.washington.edu/entity_type">Choi</a>.
```shell script
sudo chmod +x ./script/get_fet_dataset.sh
./script/get_fet_dataset.sh
```

### Preprocessing
```shell script
python param_init.py
```

### Usage
See <code>./script</code>

### Download the denoised dataset
```shell script
./script/get_denoised_dataset.sh
./script/test.sh
```

### Citation
If you find this code helpful, please kindly cite the following paper.
```
@inproceedings{pan-etal-2021-context-aware,
    title = "Automatic Noisy Label Correction for Fine-Grained Entity Typing",
    author = "Pan, Weiran and Wei, Wei and Zhu, Feida",
    booktitle = "Proceedings of the Thirty-first International Joint Conference on Artificial Intelligence, {IJCAI-22}",
    year = "2022",
}
```
