#ifndef HIGHPASSFILTER_HPP
#define HIGHPASSFILTER_HPP

//---- Standard Headers ----//
#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>

//---- User Headers ----//
#include "parameters.hpp"



class HighPassFilter 
{ 
public:
    //----- Public Functions -----//
    HighPassFilter(Parameters &params);
    void addEvent(Eigen::Vector4d e);
    cv::Mat getFrame();
    double getTime(){return m_t_current; };
    void setTime(double ts){m_ts_array.setTo(ts);};



private:
    //----- Private Variables -----//
    int m_f_verbose = 0;
    
    // Image arrays
    cv::Mat m_log_intensity, m_ts_array;

    // HPF parameters
    double m_alpha;
    double m_contrast_threshold;
    double m_t_current = 0;
    bool m_f_ts_init = 0;


    //----- Private Functions -----//
    void applyGlobalHighPassFilter();


};

#endif // HIGHPASSFILTER_HPP