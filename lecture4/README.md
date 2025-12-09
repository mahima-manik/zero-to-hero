At the initialization itself you have an idea what loss to expect based on loss function and problem setup. 

Example:
- In the character-level model, we have 27 possible character
- At the time of initializtion, all 27 chars should be around equally possible (`1/27.0`)
- We calculate loss by taking log of probability = `log(1/27.0)` ~ 3.2
- So our gradient descend should start from the loss value of 3.2
 