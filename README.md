# About Dataset
The dataset contains 1120 images divided equally into two classes, where 560 images are of Smoking (smokers) and remaining 560 images belong to NotSmoking (non-smokers) class. The dataset is curated by scanning through various search engines by entering multiple keywords that include cigarette smoking, smoker, person, coughing, taking inhaler, person on the phone, drinking water etc. We tried to consider versatile images in both classes for creating a certain degree of inter-class confusion in order to better train the model. For instance, Smoking class contains images of smokers from multiple angles and various gestures. Moreover, the images in NotSmoking class consists of images of non-smokers with slightly similar gestures as that of smoking images such as people drinking water, using inhaler, holding the mobile phone, coughing etc. The dataset can be used by the prospective researchers to propose deep learning algorithms for automated detection and screening of smoker towards ensuring the green environment and performing surveillance in smart cities. All images in the dataset are preprocessed and resized to a resolution of 250×250. We considered 80% of the data for training and 20% for validation purposes.

### Get started
```
[git clone our repo](https://github.com/edgarhovhannisyan-7/Smokers-and-not_smokers-image-classification.git)
cd repo_name
```
### Create virtual environment
 - For windows
```
./venv/Scripts/actiate
```
 -  For macos|linux
```
source venv/bin/activate
```
### Install pip
```
py -m pip install --upgrade pip
```
### Install required packages
```

pip install opencv-python numpy pandas keras tensorflow matplotlib 
```

###And now you can run and use!
