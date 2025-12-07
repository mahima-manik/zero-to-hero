# makemore

This is a character level language model. 
### Exercise 1
Train a **trigram language model**, i.e. take two characters as an input to predict the 3rd one. 
Feel free to use either counting or a neural net. 
Evaluate the loss; Did it improve over a bigram model?

<details>
  <summary>Solution</summary>
  
  **Data preparation**:
  - Load the dataset into a list of names
  - Form stoi and itos

  **Counting approach**:
  - Fill in the count matrix N
    ```
    N.shape: 27, 27, 27
    ```
  - Calculate probability matrix using N
  - Sample characters from the probability matrix and make more names
  - Calculate loss

  Loss has improved from bigram model which was 4.915 to nll=2.09 in trigram model
  
  **Neural network approach:**
  - Form the input and output vectors xs and ys
  - One-hot encoding. Since there are two characters, we append the one-hot encoding of both the input characters
    ```python
    xenc = F.one_hot(xs, num_classes=27).float() 
    xenc = xenc.view(total_trigrams, 2*27)
    ```
  - Initialize the weights using random normal distribution. Weights should be able to handle 27*2 dimension input
    ```python
    W = torch.randn(27*2, 27, requires_grad=True)
    ```
  - Calulate loss function
  - Write gradient descent loop & improve the loss
  - make more names (sampling)
</details>

### Exercise 2
Split up the dataset randomly into 80% train set, 10% dev set, 10% test set. 
Train the bigram and trigram models only on the training set. 
Evaluate them on dev and test splits. What can you see?

<details>
  <summary>Solution</summary>
  
  **Data preparation**:
  - Load the dataset into a list of names
  - Shuffle the list
  - Divide into 3 lists: train_list, dev_list and test_list
  - Form stoi and itos

  **Bigrams**:
  - Create xs and ys for training
  - Encode xs and ys using one-hot
  - Write gradient descent loop & improve the loss

  To evaluate on dev & test splits, calculate loss value on both the splits:
  - Convert the integer input into one-hot encoding
  - Pass it through layer of neurons (`xs @ W`). This will give logits
  - Exponentiate the logits to get the count
  - Calculate probability distribution over the next character by normalizing across each row
  - Loss is mean of sum of log probabilities. We take logs because, multiplying probabilities will give very small number.
  
  **Trigrams**
  - Same as in Exercise 1
</details>

### Exercise 3

Use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?

<details>
  <summary>Solution</summary>
  
  **Data preparation**:
  - 

  **Trigrams**
  - 
  
</details>