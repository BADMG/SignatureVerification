# SignatureVerification - Axis Bank AI Challenge
A Web Application to detect whether a given signature is real or forged.

### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

### Description
A system to accept a genuine signature of the customer and store it in the database against the customer ID at the time of registration of the customer. This signature can be used to compare with the input signature received to check whether the input signature is real or forged by returning a Confidence Match Score.

### Python  Implementation

* Network Used - Convolutional Siamese Network
* Framework - Django

### Procedure
  
 * For testing -- 
      1. Start cmd in SignatureVerification folder
      2. Type python `manage.py` runserver 
      3. Navigate to http://127.0.0.1:8000/Main 
      4. Start testing 

 * During testing -- 
      1. Register the customer using the real signatures (You can upload upto 5 signatures)
      2. Click verify tab to enter the number of forms(For bulk verification) 
      3. Now upload the images and enter the customer ID 
      4. Click verify to find the predictions

 * For training the model --
      1. Download the provided source code and open keras-model folder.
      2. Download the provided dataset and extract in folder keras-model such that "../SignatureVerification/keras-model/dataset"
      3. Open `model.py` and set the current python working directory to keras-model/dataset
      4. Run the whole code

![signatureverification](https://user-images.githubusercontent.com/29205181/50221245-89127780-03ba-11e9-879f-9e1ade4898c5.gif)

### Contributors

##### 1) [Saurabh Bora](https://github.com/enthussb) - [LinkedIn](https://linkedin.com/in/saurabh-bora)
##### 2) [Daksh Pokar](https://github.com/dakshpokar) - [LinkedIn](https://linkedin.com/in/dakshpokar)
##### 3) [Aniket Gokhale](https://github.com/aniketgokhale) - [LinkedIn](https://www.linkedin.com/in/aniket-g-552837b6/)
##### 4) [Atharva Dharmadhikari](https://github.com/Atharva13) - [LinkedIn](https://www.linkedin.com/in/atharva-dharmadhikari/)
##### 5) [Manas Shende](https://github.com/ms2607) - [LinkedIn](https://www.linkedin.com/in/manas-shende-6b6656126/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

