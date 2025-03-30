# Hopfield networks

It is neural network which is used for associative memory.

## What it is

A Hopfield network is a form of recurrent neural network (RNN) designed
for pattern storage and recall. It can retrieve stored patterns even when
presented with a noisy or incomplete version of them. The network operates
using binary neurons with states −1,1 and is trained using Hebbian learning.

## Components and Parameters

### Parameters

- Size (size of neurons) - Number of neurons
- Weights (weight matrix) - symmetric matrix which encodes pattern associations, updated via Hebbian rule (sum of outer products of patterns)
    - no self-loops

### Functions (components)

- Training
    - Hebbian learning (outer product of patterns)
    - Storing patterns (patterns are stored in the weight matrix)
- Recall - restores noisy pattern using iterrative updates from weight matrix
    - Synchronous update - all neurons are updated at once
    - Asynchronous update - one neuron is updated at a time
- Step function - determines the state of a neuron based on its input
    - Hard limit function - if input > 0, output 1, else -1

## Results

![pattern before](../results/hop_Original%20Pattern.png)
One of the generated patterns

![pattern after noise](../results/hop_Noisy%20Pattern.png)
Above pattern with added noise

![pattern after recall](../results/hop_Restored%20Pattern.png)
Generated pattern after recall

So we can say even if patterns are randomly generated so there is
big probability that two patterns will be simillar it still manages to restore original pattern.