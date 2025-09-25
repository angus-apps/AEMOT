#include <eigen3/Eigen/Dense>
#include <iostream>
#include <iomanip>

#include "SAE.hpp"
#include "parameters.hpp"

//---- Constructor Functions ----//
SurfaceOfActiveEvents::SurfaceOfActiveEvents(const Parameters &params){
    //Check if the provided method is allowed
    bool method_status = std::find(std::begin(ALLOWED_METHODS), std::end(ALLOWED_METHODS), params.SAE_method) != std::end(ALLOWED_METHODS) ? true : false;
    if (method_status == false){
        std::cerr << "SAE method is not allowed" << std::endl;
    }

    // Update class values
    m_method = params.SAE_method;
    m_f_verbose = params.f_verbose_SAE;

    // Resize SAE image to the correct dimensions and set to zero
    m_SAEimage.resize(params.height, params.width);
    m_SAEimage.setZero();

    if (m_f_verbose){
        std::cout << "Surface of Active Events initialised" << std::endl;
    }  
}

//---- Add Event to SAE ----//
void SurfaceOfActiveEvents::addEvent(Eigen::Vector4d e){
    int x = e(0);
    int y = e(1);
    double t = e(2);
    int p_signed = (e(3)>0) ? 1 : -1;

    if (((m_method == "positive") & (p_signed == 1)) ||
        ((m_method == "negative") & (p_signed == -1))|| 
        (m_method == "all")){
        m_SAEimage(y, x) = t;
        m_t_current = t;

        if (m_f_verbose){
            std::cout << std::setprecision(7) << std::fixed;
            std::cout << t << "\t(" << x << ", " << y << ")\t " << "SAE updated" << std::endl;
        }
    }
}
