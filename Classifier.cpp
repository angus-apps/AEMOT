

#include <torch/script.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>

#include "Classifier.hpp"
#include "parameters.hpp"

//------------------------------------//
//-----          PUBLIC          -----//
//------------------------------------//

// Constructor
Classifier::Classifier(Parameters &params){
    // Load the torch model
    std::string model_filepath = params.classifer_directory + params.classifier_model;
    try {
        model = torch::jit::load(model_filepath);
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("Unable to load Classifier model.");
    }

    // Update parameters
    m_f_verbose = params.f_verbose_classifier;
}


// Function to run the classifier
float Classifier::runClassifier(torch::Tensor input_tensor){
    // Normalize the tensor to [-1, 1]
    input_tensor = input_tensor.sub(0.5).div(0.5);

    // Run the model
    torch::Tensor output = model.forward({input_tensor}).toTensor();
    float prediction = output[0][0].item<float>();

    if (m_f_verbose){
        std::cout << "Prediction: " << prediction << std::endl;
    }

    return prediction;
}