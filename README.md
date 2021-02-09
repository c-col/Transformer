# Transformer

This is altered pytorch implementation of the transformer model from <a href=https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec>Sam Lynn Evans</a>. This version is designed for our list of strings prediction task and includes:

- validation loss calculation
- prediction on test set
- simple word piece tokenizer (no spacy)
- updated CLI 

# Usage

A folder containing the following files is needed to train the model: 
- train_in.txt
- train_out.txt
- val_in.txt
- val_out.txt
- test_in.txt
- test_out.txt

Each -in.txt / -out.txt file should contain parallel sentences separated by new line characters. 

To begin training, run this code:
```
python train.py -data_path your_data_folder_path
```

Additional parameters:<br />
-epochs : how many epochs to train data for (default=15)<br />
-batch_size : measured as number of tokens fed to model in each iteration (default=3000)<br />
-n_layers : how many layers to have in Transformer model (default=6)<br />
-heads : how many heads to split into for multi-headed attention (default=8)<br />
-no_cuda : adding this will disable cuda, and run model on cpu<br />
-SGDR : adding this will implement stochastic gradient descent with restarts, using cosine annealing<br />
-d_model : dimension of embedding vector and layers (default=512)<br />
-dropout' : decide how big dropout will be (default=0.1)<br />
-printevery : how many iterations run before printing (default=100)<br />
-lr : learning rate (default=6e-5)<br />
-load_weights : if loading pretrained weights, put path to folder where previous weights and pickles were saved <br />
-max_strlen : sentenced with more words will not be included in dataset (default=512)<br />
-checkpoint : enter a number of minutes. Model's weights will then be saved every this many minutes to folder 'weights/'<br />