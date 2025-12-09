A Neural Probabilistic Language Model 2003 paper [here](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)


Problem with gradient descent approach is that as the context size increases (trigrams and above), count matrix size also increases exponentially. Above paper addresses the same issue. 
- **Curse of dimensionality**: 

Block size: number of characters taken as an input to predict the next one (same as context length)

Lookup table or embedding:
27 possible characters needs to be embedded in our example. 
In paper, 17K words -> 30 dimension space
So, 27 characters -> 2 dimension space

One-hot vector encoding of the number (say `n`) will just return `C[n]` on vector multiplication. So, embedding of integer `n` is `C[n]`.

Now, we need to embed the entire dataset of `X` using `C` 

`C` can be seen as the weight matrix

Pytorch indexing is really powerful. You can index not just with integers, but with tensors & multidimensional tensors
We can index 

@@ Pytorch internals blog post

`F.cross_entropy`
- Negative numbers are okay in logits, they give a very small counts. 
- Positive numbers are not okay in logits. They overflow the counts because of exp() operation and can give `inf`.  
- F.cross_entropy internally subtracts the highest number from logits, so that the result is always well behaved. 


Forward, backward & update on the whole dataset is very time-consuming. We can also do it in small baches. We select the indexes of these data points at random. 

In practice, this works quite well. The decrease in loss is not steady as it is acting on different dataset everytime. But we can run many more iterations to make it converge. 

### Using cross-entropy function
Benefits of using `F.cross_entropy`:
- Avoids creating intermediate tensors, thereby saving memory.
- Provides an efficient backward pass.
- Handles numerical instabilities, especially when dealing with extremely large or small logits.

Previously:
```python
# Forward pass
logits = xenc @ W + bias # predicted log counts

# softmax
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)

loss = -probs[torch.arange(num), ys].log().mean()
```

Now:
```python
import torch.nn.functional as F
logits = xenc @ W + bias # predicted log counts
loss = F.cross_entropy(logits, ys)
```

### Mini-batches of dataset
Instead of training on the entire dataset, which can be computationally expensive, mini-batches are used.
Training on mini-batches provides an approximate gradient, which is often sufficient for training purposes and is much faster than using the full dataset.
    
```python
## Minibatch construction of size 32, over total data points in X
res = X[torch.randint(low=0, high=X.shape[0], size=(32,))]
```

### How to determine learning rate?
- Learning rate is the rate at which weights are updated to minimize the loss
- When its high, we can be stepping too fast and if it is too low, leaning rate change could be too slow
- You can plot loss of y-axis and learning rate on x-axis to find the best value or ranges for learning rate. Choose a learning rate from the plot where the loss is still decreasing but before it starts to increase or become unstable.
- Further decrease the learning rate or reduce by a factor of 10 when the loss try to converge or plateuau. This is known as **learning rate decay**.


### More ways to improve the model
- Encoding the input in higher dimension, and not just 2 dimension
- Increasing number of neurons in the layer
- Increasing number of steps and also changing learning rate (slowing down towards the end)
- Changing the block_size or context length

### Bonus
- Don't calulate the loss on test data ofter, it leads to model learning the data. 
- Another way to split dataset into train, dev and test:
```
n1 = int(0.8 * X.shape[0])
n2 = int(0.9 * X.shape[0])
Xtr, Xdev, Xts = X.tensor_split((n1, n2), dim=0)
Ytr, Ydev, Yts = Y.tensor_split((n1, n2), dim=0)
```
