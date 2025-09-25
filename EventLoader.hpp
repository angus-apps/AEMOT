#ifndef EVENTLOADER_HPP
#define EVENTLOADER_HPP

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/core.hpp>
#include <fstream>

#include "parameters.hpp"
#include "HighPassFilter.hpp"

class EventLoader
{
public:

    //----- Public Functions -----//
    // Constructor
    EventLoader(std::string filepath, Parameters &params);

    void preloadByEvent();
    std::vector<Eigen::Vector4d> getAllEvents();

    int getCount(){return m_event_counter; };

    // Get the next event
    Eigen::Vector4d getEvent();
    HighPassFilter* getTargetSelectObject();

    double m_ts = 0;    

    double start_index = -1;    
    double end_index = -1;    
    double start_time = -1;    
    double end_time = -1;   

private:
    //----- Private Functions -----//
    std::string readEventLine();
    Eigen::Vector4d decodeLine(std::string line);

    //----- Private Variables -----//
    std::ifstream m_input_file;
    int m_f_verbose = 0;
    int m_event_counter = 0;
    int m_event_format = 1;
    
    int m_f_process_start_time = 0;
    int m_f_process_start_events = 0;

    double t0 = -1;
        
    // Have a HPF pointer ready if we need a selectTarget window
    HighPassFilter* m_hpf_initialiser = NULL;
};

#endif //EVENTLOADER_HPP