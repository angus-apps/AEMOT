#ifndef SAE_HPP
#define SAE_HPP

#include <eigen3/Eigen/Dense>
#include "parameters.hpp"

class SurfaceOfActiveEvents
{
public:
    //------------------------------------------//
    //-----        PUBLIC FUNCTIONS        -----//
    //------------------------------------------//
    // Constructor functions
    SurfaceOfActiveEvents(const Parameters &params);
    // Add event to SAE
    void addEvent(Eigen::Vector4d e);
    // Access the image
    Eigen::MatrixXd getImage(){return m_SAEimage;};

    //------------------------------------------//
    //-----        PUBLIC VARIABLES        -----//
    //------------------------------------------//
    double m_t_current = 0;

private:

    //------------------------------------------//
    //-----        PRIVATE VARIABLES       -----//
    //------------------------------------------//
    bool m_f_verbose = false;
    Eigen::MatrixXd m_SAEimage;
    std::string m_method;
    std::string ALLOWED_METHODS[3] = {"all", "positive", "negative"};
};

#endif