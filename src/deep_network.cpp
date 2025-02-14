#include "deep_network.hpp"
#include <atomic>
#include <cassert>

DNN::DeepNetwork::DeepNetwork(const cl::vector<int> &dimensions, const cl::vector<ActivationFunction> &activationFunctions) 
    : dimensions(dimensions), activationFunctions(activationFunctions)
{
    const int hLayers = activationFunctions.size();
    assert(dimensions.size() == hLayers + 1 && hLayers > 0);

    weights.resize(hLayers);
    for(int i = 0; i < hLayers; ++i)
        weights[i] = cl::vector<cl::vector<float>>(dimensions[i+1], cl::vector<float>(dimensions[i], 0.f));

    biases.resize(hLayers);
    for(int i = 0; i < hLayers; ++i)
        biases[i] = cl::vector<float>(dimensions[i+1], 0.f);

}
