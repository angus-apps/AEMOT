#ifndef KALMAN_HPP
#define KALMAN_HPP

//---- Standard Headers ----//
#include <eigen3/Eigen/Dense>
#include <boost/circular_buffer.hpp>
#include <torch/script.h>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

//---- User Headers ----//
#include "parameters.hpp"
// #include "DataLogger.hpp"
#include "HighPassFilter.hpp"


class KalmanFilter
{

public:

    //------------------------------------------//
    //-----        PUBLIC FUNCTIONS        -----//
    //------------------------------------------//
    // Constructor functions
    KalmanFilter(const Parameters &params);
    KalmanFilter(const Parameters &params, Eigen::Vector<double,10> x0, double ts);
    KalmanFilter(const Parameters &params, Eigen::Vector<double,10> x0, double ts, int unique_ID);

    // Initialisation functions
    void initialiseFilterParameters(const Parameters &params);
    void updateStartPoint(Eigen::MatrixXd x0);


    void predict(Eigen::Vector4d e);
    void predictAndUpdate(Eigen::Vector4d e);

    // Functions to access private functions
    Eigen::Vector<double, 10> getState() { return x_hat; };
    Eigen::Matrix<double, 10, 10> getCovariance() { return P; };
    double getTime() { return ts_last; };
    void setTime(double ts) {ts_last = ts; };

    void addEventToIntensityPatch(Eigen::Vector4d e);
    torch::Tensor getIntensityPatchTensor();
    void resetEvaluationFlag();


    //------------------------------------------//
    //-----        PUBLIC VARIABLES        -----//
    //------------------------------------------//
    double distance_threshold = 10;
    int event_count = 0;
    int event_count_since_evaluation = 0;
    bool f_evaluate = 0;
    boost::circular_buffer<float> evaluation_buffer;
    int evaluation_buffer_size = 3;
    int ID = -1;
    double dt_average = -1;
    

    HighPassFilter* associated_events;


private:
    //------------------------------------------//
    //-----        PRIVATE FUNCTIONS       -----//
    //------------------------------------------//
    // Filter functions
    void prediction(Eigen::Vector4d e);
    void update(Eigen::Vector4d e);

    //------------------------------------------//
    //-----        PRIVATE VARIABLES       -----//
    //------------------------------------------//
    Parameters m_params;

    bool f_initialised = 0;
    bool f_data_logger_initialised = 0;

    double lambda_init = 30;
    double ts_last = 0;
    int ring_buffer_len = 10;

    // State Matrices
    Eigen::Vector<double, 10>    x_hat;
    Eigen::Matrix<double, 10, 10>   F;
    Eigen::Matrix<double, 10, 10>   Q;
    Eigen::Matrix<double, 3, 3>     R;
    Eigen::Matrix<double, 10, 10>   P;
    Eigen::Matrix<double, 10, 10>   I;

    double m_q_v = 200000;
    double m_q_v_update_event_count = 1000;
    double m_q_v_update_ratio = 0.9;    

    // Ring buffers
    boost::circular_buffer<std::tuple<double, double, double, double>> ring_buffer_e;
    boost::circular_buffer<std::pair<double, double>> ring_buffer_x;
    boost::circular_buffer<double> ring_buffer_theta;
    boost::circular_buffer<std::pair<double, double>> ring_buffer_delta;

    // Intensity patch and time array (to decay)
    double alpha = 0;
    int evaluation_event_count_threshold =0;
    double ts_last_intensity_patch = 0;
    bool f_intensity_patch_initialised = 0;

    cv::Mat intensity_patch = cv::Mat::zeros(28,28, CV_64F);
    cv::Mat ts_intensity_patch = cv::Mat::zeros(28,28, CV_64F);


};

#endif