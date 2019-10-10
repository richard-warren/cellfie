# cell's kitchen (work in progress)
neuron segmentation for calcium imaging. to be used as initialization for CalmAn

**stage 1 - region proposal**  
identifies potential neuron locations using fully convolutional networks to segment neurons from background

**stage 2 - instance segmentation**  
applies a segmentation network to subframes centered at maxima of segmentation from stage 1, yielding masks for individual neurons as well as confidence scores for each subframe

## instructions
cell's kitchen is currently undergoing pretty major refactoring! documentation to follow when things have stabilized...