# ROW
This repository is the official implementation of 

Learnability and Algorithm for Continual Learning. ICML, 2023 ([paper](https://arxiv.org/pdf/2306.12646.pdf))

****** IMPORTANT ******
Please download the pre-trained transformer network from

https://drive.google.com/file/d/1uEpqe6xo--8jdpOgR_YX3JqTHqj34oX6/view?usp=sharing

and save the file as ./deit_pretrained/best_checkpoint.pth

# Requirements
Please install the following packages
```
	pytorch==1.7.1
	torchvision==0.8.2
	cudatoolkit=11.0
	ftfy
	timm==0.4.12
```

# Training
```
bash configs/<experiment.sh>
```

Substitute <experiment.sh> with the name of the experiment you want to run

# Acknowledgement
The code format follows DER++, HAT

https://github.com/aimagelab/mammoth

https://github.com/joansj/hat
