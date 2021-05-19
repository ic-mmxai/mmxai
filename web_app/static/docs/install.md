# Installation

## MMXAI

Create a virtual environment in your preferred way, then run these commands. These will add mmxai and the dependencies to your pip list and will be enough if you only want to work on mmxai methods. Please see **examples/** for how to use each XAI methods.

```
git clone git@gitlab.doc.ic.ac.uk:g207004202/explainable-multimodal-classification.git
cd explainable-multimodal-classification
pip install --editable .
pip install -r requirements.txt
```

## Running the web app

### Installing [MMF](https://github.com/facebookresearch/mmf)
In the web app, we use several pretrained models from MMF, who does not provice interfaces with these models. The /mmf directory contains the extended and modified files for this purpose. These are integrated in a fork of [MMF](https://github.com/junqi-jiang/mmf). We recommend install from this fork rather than install from MMF and manually replace the files.

```
git clone https://github.com/junqi-jiang/mmf.git
cd mmf
pip install --editable .
```

#### Fixing checkpoints config file
We use [MMBT](https://arxiv.org/abs/1909.02950), late_fusion, [vilbert](https://arxiv.org/abs/1908.02265), [visual bert](https://arxiv.org/abs/1908.03557) models from MMF. Due to a [known issue in MMF](https://github.com/facebookresearch/mmf/issues/784), some downloaded pretrained models have curious config files which will cause error, some extra steps are needed to solve it.

when first [loading each pretrained model](https://gitlab.doc.ic.ac.uk/g207004202/explainable-multimodal-classification/-/tree/master/examples), the files are downloaded automatically to your default torch cache directory. 

```
cd ~/.cache/torch/mmf/data/models/
ls
```

This will list all the pretrained checkpoints. cd into *visual_bert.finetuned.hateful_memes.from_coco* and *vilbert.finetuned.hateful_memes.from_cc_original*, use your preferred text editor to open the **config.yaml**, find and modify the following terms:
- cache_dir: ${resolve_cache_dir:MMF_CACHE_DIR}
- data_dir: ${resolve_dir:MMF_DATA_DIR, data}
- save_dir: ${env:MMF_SAVE_DIR, ./save}
- resume_file: (replace by your absolute path to the cache folder)/data/models/(vilbert.pretrained.cc_original or visualbert)/model.pth

For vilbert, they are on line 217, 220, 221, 41. For visual bert, they are on line 161, 164, 165, 41.

#### Download and configure feature extractor checkpoints for vilbert and visual bert

This step is only needed for using vilbert and visual bert. Download checkpoints from [this link](https://drive.google.com/file/d/1iJ9D_sunUKiJaQWRb4iAY0LCUJu8wR3q/view?usp=sharing), unzip it, put them wherever you want.

Then go to the location where the fork of MMF was installed. Find mmf/tools/scripts/features/frcnn/feature_extraction_frcnn.py, change the paths at line 28 and 29 to the unzipped files config.yaml and model_finetuned.bin.

After these steps, you will be able to load and use pretrained MMF models (mmbt, late_fusion, vilbert, visual_bert), see **examples/**.

### Run the app

```
cd [wherever you cloned the explainable-multimodal-classification repo]
cd web_app
python3 app.py
```

