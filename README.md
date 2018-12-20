# Face Denoising Project for Context Aware Security Analysis for Computer Vision

### Info
Provided models have been trained on 10.000 images 64x64, the dataset is custom made and based on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) with a little modification.
In our project we had to find the best performing model by only looking at papers and testing it with 2 set of faces: *cropped* (as close to no-background as possible) and *large* (as much background as possible)
![Results table](https://github.com/JustAnOwlz/Face-Denoising-CASACV/blob/master/output.png)
Top row is the input images, the original ones that the NN has never seen before
Second row is the noisy images, the input of our NNs
Third row is the output images for the WIN5_LARGE model
Fourth row is the out images for the WIN5 model

We provide a whitepaper (whitepaper.pdf) for better understanding of the process that made this models possible

### Models and Hardware requirements
The models have been trained on a nVidia Quadro P4000, each epochs took 93-95 seconds.
  ***WIN5*** model was trained for 75 epochs
  ***WIN5_BW*** model was trained for 25~ epochs
  ***WIN5_LARGE*** model was trained for 25 epochs
  ***DNCNN_BW*** was trained for 27~ epochs
  ***DNCNN*** was trained for 25~ epochs

### How to install
```
git clone https://github.com/JustAnOwlz/Face-Denoising-CASACV.git
cd face-denoising-casacv
pip install -r requirements.txt
python model_trainer_edited.py
```