#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

//---- Standard Headers ----//
#include <eigen3/Eigen/Dense>
#include <torch/script.h>

//---- User Headers ----//
#include "parameters.hpp"


class Classifier
// Class to interact with the bee classifier
{
public:
    Classifier(Parameters &params);
    float runClassifier(torch::Tensor input_tensor);
    
private:
    // Torch model
    torch::jit::script::Module model;

    // Flags
    bool m_f_verbose = 0;
};


#endif