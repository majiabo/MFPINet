# Reconstruct high-resolution multi-focal plane images from a single 2D wide field image
The implementation of MFPINet.
## Model
Model and toolkits can be found at lib directory. 
We recommend Pycharm to avoid dependency problems.
## Train
see train.py and configs directory for more details. 
## Test
To test the pretrained model, run following instruction: 
```shell
python predict.py
```
then you will find a generated multi-focal plane images at asset directory.
![generated multi-focal plane images](./assets/images/result_sample_1.jpg)

## citation
@article{ma2020reconstruct,
  title={Reconstruct high-resolution multi-focal plane images from a single 2D wide field image},
  author={Ma, Jiabo and Liu, Sibo and Cheng, Shenghua and Liu, Xiuli and Cheng, Li and Zeng, Shaoqun},
  journal={arXiv preprint arXiv:2009.09574},
  year={2020}
}