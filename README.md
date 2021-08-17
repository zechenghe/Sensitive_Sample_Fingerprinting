# Sensitive Sample Fingerprinting

This code demonstrates sensitive-sample fingerprinting in the following paper:

Zecheng He, Tianwei Zhang, and Ruby Lee, "[Sensitive-sample Fingerprinting of Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.html)", IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019

#### Download models and data

This demo generates sensitive samples to detect integrity breaches, e.g., neural network trojans, in a deep neural network. Download models and data using [this link](https://drive.google.com/drive/folders/1awN7O8WC9Pi-f6YQNkzNND3Yh7J22B7B?usp=sharing). The clean and trojaned models (face recognition in Caffe) are obtained from https://github.com/PurduePAML/TrojanNN. The models are transformed from Caffe to Pytorch using [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch).

    git clone https://github.com/zechenghe/Sensitive_Sample_Fingerprinting.git

- Models


    mkdir model
    cd model

Put `VGG-face-clean.pt` and `VGG-face-traojaned.pt` from the [above link](https://drive.google.com/drive/folders/1awN7O8WC9Pi-f6YQNkzNND3Yh7J22B7B?usp=sharing) under `model/`

- Data


    mkdir data
    cd data

Put `names.txt`, `VGGFace-Clean/` and `VGGFace-Trojaned/` from the [above link](https://drive.google.com/drive/folders/1awN7O8WC9Pi-f6YQNkzNND3Yh7J22B7B?usp=sharing) under `data/`


#### Dependencies

This Pytorch implementaion is based on the following configuration.

Python 3.6.8

Numpy 1.19.5

Pytorch 1.4.0

Torchvision 0.5.0

Pickle 4.0

Matplotlib 3.3.2 (for saving images)

Glob, Pathlib


#### Generate sensitive samples

`sensitive-sample-gen.py` is used to generate sensitive samples. The generated sensitive samples are saved in `generated/`.

    python3 sensitive-sample-gen.py --input_dir_clean data/VGGFace-Clean --lr 1e-2 --gpu

To skip sanity check, use `nosanity_check` option. Use `--gpu` to enable gpu (default).

#### Maximum Active Neuron Cover (MANC) sample selection

`manc.py` performs Maximum Active Neuron Cover (MANC) sample selection. Assume the candidate sensitive samples have been generated in `generated/` folder. This will select a small subset of `n_samples` sensitive samples from a large bag of samples.

    python3 manc.py --n_samples 2 --gpu

Similarly, to skip sanity check, use `nosanity_check` option. Use `--gpu` to enable gpu (default).

# Reference
You are encouraged to cite the following paper.
```
@inproceedings{he2019sensitive,
  title={Sensitive-sample fingerprinting of deep neural networks},
  author={He, Zecheng and Zhang, Tianwei and Lee, Ruby},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4729--4737},
  year={2019}
}
```
