#ifndef DATAASSOCIATION_HPP
#define DATAASSOCIATION_HPP

//---- Standard Headers ----//
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

//---- User Headers ----//
#include "Kalman.hpp"
#include "parameters.hpp"

class DataAssociation 
{
public:    
    //------------------------------------------//
    //-----        PUBLIC FUNCTIONS        -----//
    //------------------------------------------//
    // Constructor
    DataAssociation(const Parameters &params);

    // Perform Data 
    // std::pair<int, std::vector<int>> performDataAssociation(const Eigen::Vector4d e, std::vector<KalmanFilter*> output_tracks, std::vector<KalmanFilter*> candidate_tracks);
    std::tuple<int, std::vector<int>, int> performDataAssociation(const Eigen::Vector4d e, std::vector<KalmanFilter*> output_tracks, std::vector<KalmanFilter*> candidate_tracks);

private:


    //------------------------------------------//
    //-----        PRIVATE VARIABLES       -----//
    //------------------------------------------//
    // Arrays to store the positions of the blobs
    double m_alpha  = 0;
    double m_dist_threshold;
    double m_ts_last = 0;
    double m_global_association_distance = 500;

    // Define data association method.
    int m_f_use_mahal = 0;
    double m_mahal_dist_threshold = 0;
    int m_f_use_euc = 0;
};

#endif
