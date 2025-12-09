## Part 3: Activations & Gradients, BatchNorm

At the initialization itself you have an idea what loss to expect based on loss function and problem setup. 

Example:
- In the character-level model, we have 27 possible character
- At the time of initializtion, all 27 chars should be around equally possible (`1/27.0`)
- We calculate loss by taking log of probability = `log(1/27.0)` ~ 3.2
- So our gradient descend should start from the loss value of 3.2

In reality, it started with ~25, which is very high. So many iterations of gradient descent wasted initially.

We want the logits to be rougly 0 when the network is initialized. They just don't have to be zero, but approximately equal. 

Logits are derived by multiplying hidden layer output by `W2` and adding `b2`. To bring the values of logits closer to zero, we can do following:
- Reduce what is being added to logits (b2 * 0) - by multiplying it with 0
- Multiply logits by a very small number, say 0.01 to bring the result closer to 0. 

```python
W2 = torch.randn(n_hidden, vocab_size) * 0.01
b2 = torch.randn(vocab_size) * 0
```

loss = 2.069702386856079 after 60000 iterations 

## Activation problems

Imagine a tanh neuron. If all the outputs are close to 1 or close to -1, then  it is squashing the weights and suppressing lot of meaning in the hidden data. 

![Graph of tanh outputs](tanh_outputs.png)

If the output of tanh is skewed towards 1 and -1 (as shown in the graph above), then during backpropagation, gradient is almost always equals to zero. That means there is no or very little influence of weights and biases on the gradient in backpropagation. It is pushed to 0s.

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1) # tanh(x) is calculated here
    out = Value(t, children=(self, ), _op='tanh')
    
    def _backprop():
        self.grad += (1 - t ** 2) * out.grad # Pay attention here
    out._backprop = _backprop
    return out
```

To fix this, we need the input of tanh, i.e. hpreact, to be closer to 0:
```python
embs = C[Xb]
embcat = embs.view(-1, block_size * n_embd)
hpreact = embcat @ W1 + b1
h = torch.tanh(hpreact)
```

We can multiply W1, b1 by a factor of 0.1 or something. This will bring hpreact closer to 0.

![Graph of tanh](tanh.png)

When all examples give 0 gradient out on tanh, then it is called *dead neuron* - this neuron will never learn.

**Different activation functions**

1. Sigmoid
2. tanh
3. ReLU
4. Leaky ReLU
5. Maxout
6. ELU

![Activation functions](activations.png)

