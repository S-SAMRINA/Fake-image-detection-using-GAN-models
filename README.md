# Fake-image-detection-using-GAN-models
 Challenging Fake Image Detection using GAN Models
MRI-GAN: 

A Generalized Approach to Detect Deep Fakes using Perceptual Image Assessment

This README provides an overview of the scope of the MRI-GAN project, sample results, and steps required to replicate the work, both from scratch and using pre-trained models. Reproducing the results from scratch involves training all the models. Data processing steps are also outlined below.
The full research paper is available at:  https://arxiv.org/abs/2203.00108
TLDR.


Abstract

DeepFakes are synthetic videos created by replacing a face in an original image with someone else's face. This project focuses on developing deep learning models for classifying DeepFake content. We introduce a novel framework called MRI-GAN, which leverages Generative Adversarial Network (GAN)-based models to detect synthesized videos based on perceptual differences in images. We evaluate our MRI-GAN approach and a plain-frames-based model using the DeepFake Detection Challenge Dataset. Our plain frames-based model achieves 91% test accuracy, while the MRI-GAN framework with Structural Similarity Index Measurement (SSIM) for perceptual differences achieves 74% test accuracy. The MRI-GAN results are preliminary and can potentially be improved further by adjusting loss functions, hyperparameters, or using advanced perceptual similarity metrics.


MRI-GAN

MRI-GAN generates MRIs of input images. The MRI for a DeepFake image contains artifacts that highlight regions of synthesized pixels, while the MRI of a non-DeepFake image is simply a black image.
https://d.docs.live.net/f38bd683e309f3ac/Documents/blank.png

 
Steps to Replicate the Work

Note: This is a complex process.

1.Set Up Development Environment

•	Use conda for Python distribution and related libraries on Ubuntu 20.04 OS.

•	Create a new environment using the provided environment.yml file:

bash code:
conda env create -f environment.yml

•	Activate the environment.

2.Download Datasets and Extract

Download the following datasets:

•	DFDC dataset

•	Celeb-DF-v2 dataset

•	FFHQ dataset

•	FDF dataset

3.Configure Paths and Parameters

•	Update paths and parameters in the config.yml file according to your dataset locations and preferences.

4.Data Pre-processing

Execute the following commands in sequence:

•	python data_preprocess.py --gen_aug_plan

•	python data_preprocess.py --apply_aug_to_all

•	python data_preprocess.py --extract_landmarks

•	python data_preprocess.py --crop_faces

•	python data_preprocess.py --gen_mri_dataset

5.MRI-GAN Training

•	Configure the config.yml file and adjust parameters under ['MRI_GAN']['model_params'] as needed.

•	Train the MRI-GAN model:

css code:
python train_MRI_GAN.py --train_from_scratch

•	Copy trained MRI-GAN weights:

bash code 
cp logs/<date_time_stamp>/MRI_GAN/checkpoint_best_G.chkpt assets/weights/MRI_GAN_weights.chkpt

•	Use the trained MRI-GAN to predict MRIs for the DFDC dataset:

python data_preprocess.py --gen_dfdc_mri

6.Train and Test the DeepFake Detection Model

•	Generate metadata CSV files using:

css code:
python data_preprocess..py  --gen_deepfake_metadata 

•	For the plain-frames method, configure the config.yml file with the following parameters:

1.'train_transform' : 'complex'
2.'dataset' : 'plain'

•	Train the model from scratch or resume training if needed:

css code:
python deep_fake_detectpy --train_from_scratch 

•	Test the saved model:

css code:
python deep_fake_detectpy --test_saved_model <path> 

•	For the MRI-based method, configure the config.yml file with the following parameters:

1.'train_transform' : 'simple'
2.'dataset' : 'mri'

•	Train the model from scratch or resume training if needed, and test the model similarly.

7.Other Notes

•	Check the --help option of all mentioned scripts for more utility methods, such as resuming training of models if it was stopped prematurely.

Pre-trained Models

Download the pre-trained model weights to reproduce the results:

•	MRI-GAN Model with Tau = 0.3 and Generator with the Lowest Loss

•	DeepFake Detection Models:
1.	Plain-Frames Based Model
2.	MRI-Based Model

DeepFake Detection App

Use the model to test a given video file:
1.	Download all pre-trained model weights.
2.	Run the command-line app.
   
css code:
python detect_deepfake_app.py --input_videofile <path to video file> --method <detection method>> 

•	The detection method can be either plain_frames or MRI.



