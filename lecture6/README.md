## Lecture 6: 

Continuing from Lecture 4. 

Despite adding more hidden layers, we are still "crushing" the context of the input layer too soon and this limits our ability to increase our context length. Essentially, we want to fuse the initial earlier layers together progressively, like in the WaveNet paper, to avoid this issue.