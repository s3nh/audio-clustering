## Audio clustering.


Unsupervised algorithms initialize on audio files.

### Description 

audio-clustering allows to perform multiple unsupervised algorithms 
training on your audio samples. 
Training process is performed based on configuration files.
Algorithms preprocess is based on scikit-learn and pyclustering libraries. 

Files are preprocessed as follows:

- padding to mean(len) of characteristics of whole sample
- create mel-spectrogram for every sample. 
- initialize standard scaler 
- initialize pca (n_components predefine in configuration file). 
- perform training process based on methods and parameters included in src.cfg
- choose relatively best algorithm based on Davies-Boulding, Harabasz metrics. 

### Example-usage

#TODO

### Installation 

#TODO

### Resources

#TODO

### License

<a href="http://www.wtfpl.net/"><img
       src="http://www.wtfpl.net/wp-content/uploads/2012/12/wtfpl-badge-4.png"
       width="80" height="15" alt="WTFPL" /></a> 
