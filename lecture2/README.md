# makemore


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
  - Initialize the weights using random normal distribution
  - Calulate loss function
  - Perform Gradient descent
  - make more names
  
</details>

