# cellfie

<a href="https://www.teepublic.com/tank-top/2147895-cell-fie"><img src="images/cellfie.png" align=right></a>

`cellfie` (**CELL** **FI** nd **E** r) automatically segments neurons in calcium imaging videos using fully convolutional neural networks.

Neuroscientists use [calcium imaging](https://en.wikipedia.org/wiki/Calcium_imaging) to monitor the activity of large populations of neurons in awake, behaving animals (like in [this](https://www.youtube.com/watch?v=Nxa19uWC_oA) beautiful example). However, calcium imaging can be super noisy, making neuron identification challenging. `cellfie` solves this problem using a two stage neural network approach. First, a *region proposal* convolutional neural network identifies potential neurons. Next, an *instance segmentation* network iteratively identifies individual neurons.

`cellfie` is a work in progress. I'm collaborating with [Eftychios Pnevmatikakis](https://www.simonsfoundation.org/team/eftychios-a-pnevmatikakis/) at the Simon's Foundation to see if neural networks combined with matrix factorization techniques (as used by [CaImAn](https://github.com/flatironinstitute/CaImAn/blob/master/README.md)) outperform current approaches to calcium imaging segmentation.


#### region proposal
todo

#### instance segmentation
todo

#### putting it all together
todo
