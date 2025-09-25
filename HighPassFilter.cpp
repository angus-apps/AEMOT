//---- Standard Headers ----//
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>

//---- User Headers ----//
#include "HighPassFilter.hpp"
#include "parameters.hpp"


//------------------------------------------//
//-----        PUBLIC FUNCTIONS        -----//
//------------------------------------------//
HighPassFilter::HighPassFilter(Parameters &params){
    //----- Update properties -----//
    m_f_verbose = params.f_verbose_hpf;
    m_alpha = params.alpha;
    m_contrast_threshold = params.contrast_threshold;

    if (m_f_verbose){   
        std::cout << "HPF Initialisation: \n\tm_alpha: " << m_alpha << "\n\tm_contrast_threshold: " << m_contrast_threshold << std::endl;   
    }
    
    //----- Initialised Image Arrays -----//
    m_log_intensity = cv::Mat::zeros(params.height, params.width, CV_64FC1);
    m_ts_array = cv::Mat::zeros(params.height, params.width, CV_64FC1);
    m_log_intensity.setTo(0);
    m_ts_array.setTo(0);
}

void HighPassFilter::addEvent(Eigen::Vector4d e){
    //----- Unpack Event -----//
    int x = e(0);
    int y = e(1);
    double ts = e(2);
    int p = e(3);

    // Update ts_array if it is the first eent
    if (m_f_ts_init == 0){
        m_ts_array.setTo(ts);
        m_f_ts_init = 1; 
    }

    m_t_current = ts;

    //----- Add event to HPF image -----//
    m_log_intensity.at<double>(y, x) += (p > 0) ? -m_contrast_threshold : m_contrast_threshold;
    // m_ts_array.at<double>(y, x) = ts;
}

cv::Mat HighPassFilter::getFrame(){
    // Apply global high pass filter
    applyGlobalHighPassFilter();
    //Return the image
    return m_log_intensity;
}



//------------------------------------------//
//-----       PRIVATE FUNCTIONS        -----//
//------------------------------------------//
void HighPassFilter::applyGlobalHighPassFilter(){
  cv::Mat beta;
  // Compute pixel-wise decay factor
  cv::exp(-m_alpha * (m_t_current - m_ts_array), beta);
  // Apply  decay
  m_log_intensity = m_log_intensity.mul(beta);
  // Update time array (time of last decay)
  m_ts_array.setTo(m_t_current);
}