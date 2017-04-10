# Instagram Challenge by Unpossibly 

Predicting the number of likes an instagram post will receive in 24 hours 

- This is the 1st place solution
- Challenge description: http://live.unpossib.ly/instagram


The solution:

- Extract image features with vgg16
- Reduce the dimensionality to 128 with PCA
- Standardize the target by subtracting the account's mean and dividing by the std
- Train a ET model
