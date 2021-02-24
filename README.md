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
python train.py -data_path your_data_folder_path -output_dir your_folder_to_save_results -do_test
```

<strong>important or new</strong> parameters:<br />
-epochs : how many epochs to train data for (default=50)<br />
-batch_size : measured as number of tokens fed to model in each iteration (default=3000)<br />
-val_check_every_n : perform a validation accuracy check every n epochs (default=3)<br />
-calculate_val_loss : flag on whether to calculate validation cross-entropy loss every epoch (i don't use this but it's not costly)<br />
-tensorboard_graph : flag on whether to write model architecture to local tensorboard (helpful for debugging compositional dynamic architectures)<br />
-compositional_eval : flag on whether to use compositional algorithms at eval/test time for prediction<br />
-n_val : number of validation examples to use for validation accuracy checks (default=1000)<br />
-n_test : number of test examples to use for test prediction and accuracy checks (default=1000)<br />
-do_test : flag on whether to generate test predictions and calculate test accuracy (important to include)<br />
-load_weights : if loading pretrained weights, put path to folder where previous weights and pickles were saved <br />

old or less-important parameters:<br />
-n_layers : how many layers to have in Transformer model (default=6)<br />
-heads : how many heads to split into for multi-headed attention (default=8)<br />
-no_cuda : adding this will disable cuda, and run model on cpu<br />
-SGDR : adding this will implement stochastic gradient descent with restarts, using cosine annealing (default is not to use this, instead using LR reduce on validation plateau)<br />
-d_model : dimension of embedding vector and layers (default=512)<br />
-dropout' : decide how big dropout will be (default=0.1)<br />
-printevery : how many iterations run before printing (default=100)<br />
-lr : learning rate (default=1e-4)<br />
-load_weights : if loading pretrained weights, put path to folder where previous weights and pickles were saved <br />
-max_strlen : sentenced with more words will not be included in dataset (default=512)<br />
-checkpoint : enter a number of minutes. Model's weights will then be saved every this many minutes to folder 'weights/'<br />
