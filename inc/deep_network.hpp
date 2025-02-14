#pragma once

#include "matrix.hpp"

namespace DNN {
    class DeepNetwork {
    public:
        enum class ActivationFunction {SIGMOID, TANH};

        DeepNetwork() = default;
        DeepNetwork(const cl::vector<int> &dimensions, const cl::vector<ActivationFunction> &activationFunctions);

        inline const cl::vector<int> &getDimensions() const { return dimensions; };
        inline const cl::vector<ActivationFunction> &getActivations() const { return activationFunctions; };

        inline bool areResultsAvailable() const { return _resultsAvailable; };
        const cl::vector<float> &waitForResults() const;
        const cl::vector<float> &getCurrentResults() const;

    protected :
        const cl::vector<int> dimensions;
        const cl::vector<ActivationFunction> activationFunctions;
        cl::vector<cl::vector<cl::vector<float>>> weights;
        cl::vector<cl::vector<float>> biases;
        cl::vector<cl::vector<float>> resultsPublication;

    private:
        bool _lockResults;
        bool _resultsAvailable;
    };
};