# Master's Thesis

This is a code base I created and used for researching information on my Master's Thesis.

## Topic: __Pruning of Long Short-Term Memory Neural Networks: Passes of Redundant Data Patterns__

The research was focused on the domain of pruning of Neural Networks. After gathering of preliminary information, I wanted to experiment in application of own created methodology to pruning of shallow LSTM Neural Networks (trained to predict future values in univariate time-series datasets), applying knowledge learned and several techniques available in the data science. The method was experimental and was compared in its effectivness to the methods described in State of The Art pruning (SOTA) methodologies.

## Summary

In few words as possible: the method researched proposes that it can be efficient to get rid of the units in an LSTM shallow network, that do not show meaningful variations in their activations. To check how activations take places - units are visualized depending on intensity of their activations and those units that do not vary over various patterns of data inputs to the networks - are proposed for omition. This method allowed to experiment whether automatic reduction of huge shallow neural networks (of 256 neurons) could be suggested in some cases to lead them to much smaller sizes (up to 2 neurons). The reults that this method showed, have been quite satisfactory. In essence, this method can be thought of as a variational take on the grid search approach, but not to parameters optimization, but rather network's size. The reduction of the network does not account for change in errors, however, current experiments indicated that the error does not increase too dramatically, when redundant LSTM neurons are omited from the network.

Becauase of this, I wanted to share the code hoping that it will find potential use. On itself, pruning of shallow networks is not a big deal, but considering single layered Neural Networks have their own purposes - this method can be applied to them, or potentially might inspire new variations of it. 
