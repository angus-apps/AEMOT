#ifndef SAEDETECTOR_HPP
#define SAEDETECTOR_HPP

#include <eigen3/Eigen/Dense>
#include <cmath>
#include "SAE.hpp" 
#include "parameters.hpp"

class SAEdetector
{
public:
    // Constructor and initialisation
    SAEdetector(const Parameters &params);
    SAEdetector();
    ~SAEdetector();
    
    // Functions
    void addEvent(Eigen::Vector4d e);
    Eigen::Vector<double, 5> performDetection(Eigen::Vector4d e);


    // Access the image
    Eigen::MatrixXd getImage();

    //------------------------------------------//
    //-----        PUBLIC VARIABLES        -----//
    //------------------------------------------//
    double m_t_current = 0;

private:
    //------------------------------------------//
    //-----        PRIVATE FUNCTIONS       -----//
    //------------------------------------------//
    Eigen::MatrixXd getTimePatch(Eigen::Vector4d e);


    // Internal functions
    void getPatches(Eigen::Vector4d e);

    void computeLeastSquaresVelocity(Eigen::Vector4d e);
    Eigen::Vector<double, 5> classifyDetection(Eigen::Vector4d e);


    void computeDirectionRegression(Eigen::Vector4d e);
    Eigen::Vector<double, 5> compareDirectionRegression(Eigen::Vector4d e);

    //------------------------------------------//
    //-----       PRIVATE VARIABLES        -----//
    //------------------------------------------//
    SurfaceOfActiveEvents* SAE;

    //----- Parameters -----//
    // Set some default values
    double m_alpha = 1e-5;
    double m_min_active_pixels = 5;
    double m_min_contributions = 2;
    double m_detection_threshold = 0.7;
    int m_ksize;

    //----- Flags -----//
    bool m_f_patch_status = false;
    bool m_f_regression_status = false;
    bool m_f_verbose;

    int m_width, m_height;
    Eigen::MatrixXd initialised_events;
    Eigen::MatrixXd init_events_patch, t_patch;

    Eigen::MatrixXd initialised_directions, init_directions_patch;
    Eigen::MatrixXd lx, ly;
    Eigen::MatrixXd lx_patch, ly_patch;
    
    std::string m_method;
    std::string ALLOWED_METHODS[3] = {"all", "positive", "negative"};


    // Set up some patches that we will need each iteration
    // Eigen::VectorXd b;
    Eigen::VectorXd patch_offset;
    std::vector<std::pair<int, int>> patch_points;
    Eigen::Vector<double, 5> no_detection = Eigen::Vector<double, 5> ::Constant(-1);
    
};

#endif