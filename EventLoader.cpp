#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/core.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>


#include "EventLoader.hpp"
#include "parameters.hpp"


//------------------------------------//
//-----     PUBLIC FUNCTIONS     -----//
//------------------------------------//
// Constructor


//************************************************************************************//
//**********                         CONSTRUCTOR                            **********//
//************************************************************************************//
EventLoader::EventLoader(std::string filename, Parameters &params){
    // Store the filepath
    m_event_format = params.f_event_format;  
    m_f_verbose = params.f_verbose_eventloader;

    // Check if the file suffix is included
    std::string m_filename;
    if (filename.length() >= 4) {
        std::string suffix = filename.substr(filename.length() - 4);
        if (suffix == ".csv" || suffix == ".txt"){
            m_filename = filename;
        }
        else {
            m_filename = filename + ".csv";
        }
    }

    // Open the file
    m_input_file.open(m_filename, std::ifstream::in);
    if (!m_input_file.is_open()){
        throw std::runtime_error("Could not open file");
    }
    if (m_f_verbose){
        std::cout << "File " << m_filename << " opened." << std::endl;
    }

    //----- Preload up to our starting point ----//
    // If we need to click to select targets, we will initialised the HPF
    m_hpf_initialiser = new HighPassFilter(params);

    // Iterate until the start time
    if (params.f_process_start_time){
        m_f_process_start_time = 1;
        start_time = params.process_ts_start;
        end_time = params.process_ts_end;
        if (m_f_verbose){
            std::cout << "Preloading events until " << params.process_ts_start << "\nStart time: " << start_time << "\nEnd time: " << end_time << std::endl;
        }
    }

    // Iterate until the start event number
    else if (params.f_process_start_events){
        m_f_process_start_events = 1;
        start_index = params.event_num_start;
        end_index = params.event_num_end;

        if (m_f_verbose){
            std::cout << "Preloading events until " << params.event_num_start << "\nStart index: " << start_index << "\nEnd index: " << end_index << std::endl;
        }
    }

}



//************************************************************************************//
//**********                     INITIALISE BY EVENT                        **********//
//************************************************************************************//
void EventLoader::preloadByEvent(){
    int event_counter_threshold = 20000;

    int event_count = 0;
    Eigen::Vector4d e;
    // Iterate until the start time
    if (m_f_process_start_time){
        if (m_f_verbose){
            std::cout << "Preloading events until " << start_time << std::endl;
        }

        while (m_ts < start_time){
            e = getEvent();
            m_ts = e(2);
            if (start_time - m_ts < 0.25){
                m_hpf_initialiser->addEvent(e);
                event_count++;
            }
        }
    }

    // Iterate until the start event number
    else if (m_f_process_start_events){
        if (m_f_verbose){
            std::cout << "Preloading events until " << start_index << std::endl;
        }

        while (m_event_counter < start_index){
            e = getEvent();
            m_ts = e(2);

            if (start_index - m_event_counter < event_counter_threshold){
                m_hpf_initialiser->addEvent(e);    
            }
        }
    }

}



//************************************************************************************//
//**********                    INITIALISE ALL EVENTS                       **********//
//************************************************************************************//

std::vector<Eigen::Vector4d> EventLoader::getAllEvents(){
    // Vector to store all events
    std::vector<Eigen::Vector4d> all_events; 
    int event_counter = 0;
    // Iterate through file and load all events
    while (true) {
        Eigen::Vector4d e = getEvent();
        m_ts = e(2);

        if (e(0) == -1){ // end of file indicator
            break;
        }

        if (m_f_process_start_time){
            if ((start_index == -1) && (e(2) >= start_time)){
                start_index = event_counter;
            }

            if ((end_index == -1) && (e(2) >= end_time)){
                end_index = event_counter;
            }
        }
        else if (m_f_process_start_events){
            if ((start_index == -1) && (event_counter >= start_index)){
                start_index = event_counter;
            }
            if ((end_index == -1) && (event_counter >= end_index)){
                end_index = event_counter;
            }
        }
        
        all_events.push_back(e);
        event_counter++;
    }

    // If the end index has not been found, set as index of last event
    if (end_index == -1){
        end_index = event_counter;
    }

    // Initialise the HPF with some events
    int num_events= 10000;
    int hpf_start_index = (start_index-num_events >= 0) ? start_index-num_events : 0;
    int hpf_end_index = (hpf_start_index+num_events <= end_index) ? hpf_start_index+num_events : end_index;

    for (int i = hpf_start_index; i < hpf_end_index; i++){
        Eigen::Vector4d e = all_events[i];
        m_hpf_initialiser->addEvent(e);    
    }
    return all_events;
}



//************************************************************************************//
//**********                          FUNCTIONS                             **********//
//************************************************************************************//

HighPassFilter* EventLoader::getTargetSelectObject(){
    // Make the flag is set, otherwise HPF object won't be initialised
    m_hpf_initialiser->setTime(m_ts);
    return m_hpf_initialiser;
}


Eigen::Vector4d EventLoader::getEvent(){   
    // Read line from text file
    std::string line;
    std::getline(m_input_file, line);
    std::stringstream ss(line);

    // Create event variables
    Eigen::Vector4d e;
    int x, y, p;
    double ts;

    // Check if we are at the end of the file
    if (m_input_file.eof() || m_input_file.fail()){
        e.setConstant(-1);
        return e;
    }

    // Unpack line
    switch (m_event_format){
        case 0: // e = {ts,x,y,p}
            ss >> ts;
            ss.get();
            ss >> x;
            ss.get();
            ss >> y;
            ss.get();
            ss >> p;
        break;

        case 1: // e = {x,y,p,t}
            ss >> x;
            ss.get();
            ss >> y;
            ss.get();
            ss >> p;
            ss.get();
            ss >> ts;
        break;

    default:
      throw std::runtime_error("Error: unknown event_format in read_events().");
      break;
    }   

    // Store the first timestamp to offset by
    if (t0 == -1){
        t0 = ts;
    }
    
    m_event_counter++;
    // Combine into a vector and return
    e << x, y, (ts)*1e-6, p;
    return e;
}

