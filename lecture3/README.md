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