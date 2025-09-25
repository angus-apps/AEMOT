//---- Standard Headers ----//
#include <algorithm>
#include <vector>
#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include "yaml-cpp/yaml.h"
#include <torch/script.h>

//---- User Headers ----//
#include "parameters.hpp"
#include "EventLoader.hpp"
#include "Visualiser.hpp"
#include "DataAssociation.hpp"
#include "SAEdetector.hpp"
#include "Kalman.hpp"
#include "TrackManager.hpp"



//---------------------------------------------------------------//
//-----                 FUNCTION PROTOTYPES                 -----//
//---------------------------------------------------------------//

void check_parameters(Parameters &params);
void print_help();

//---------------------------------------------------------------//
//-----                        MAIN                         -----//
//---------------------------------------------------------------//
int main(int argc, char *argv[]){
    //---------------------------//
    //-----   Parse Inputs  -----//
    //---------------------------//    
    if (argc < 2 | argc > 3) { // Force there to be an input 
        print_help();
        return -1;
    }
    if (std::strcmp(argv[1], "-i") != 0) {   // Check if the second argument is '-i' -> throw error if it isn't
        print_help();
        return -1;
    }

    std::string data_name = argv[2];    // Declare the supported options.

    //---------------------------//
    //----- Load Parameters -----//
    //---------------------------//
    Parameters params;
    params = loadParametersFromYAML("../configs/" + data_name + ".yaml");
    check_parameters(params);
    std::string input_event_path = params.input_folder_path + params.input_data_name;

    //-------------------------------------------//
    //----       Set Up System Objects       ----//
    //-------------------------------------------//
    // Create EventLoader object
    std::string filename = input_event_path + ".csv";
    EventLoader event_loader(filename, params);
    std::vector<Eigen::Vector4d> all_events;

    if (params.f_load_all_events){
        all_events = event_loader.getAllEvents();
    }
    else {
        event_loader.preloadByEvent();
    }

    DataAssociation data_assocation(params);
    TrackManager track_manager(params);
    Visualiser visualiser(params);
    visualiser.initialiseAllEventsHPF(event_loader.getTargetSelectObject());


    //-------------------------------------------//
    //----       Set Up Target Select        ----//
    //-------------------------------------------//   
    // Pointer for SAE to initialise if we need to
    SAEdetector* blob_detector = NULL;
    
    switch (params.f_select_target){
        //----- Read starting points from config file ----//
        case 0: {
            // Make sure that the postion coordinates are the same dimension
            if (params.position_x_init.size() != params.position_y_init.size()){
                throw std::runtime_error("Error: params.position_x_init and params.position_y_init are different dimensions.");
            }
            // Load coordinates from config file and initialise trackers
            std::vector<cv::Point> selected_points;
            for (int i = 0; i < params.position_x_init.size(); i++){
                selected_points.push_back(cv::Point(params.position_x_init[i], params.position_y_init[i]));
            }
            // Create trackers for the selected points
            track_manager.addSelectedPoints(selected_points, event_loader.m_ts);
            break;
        }

        //----- Select targets on the screen ----//
        case 1: {
            // Transfer target select data from event loader to visualer.
            visualiser.displayForSelect();
            // Create trackers for the selected points
            track_manager.addSelectedPoints(visualiser.getSelectedPoints(), event_loader.m_ts);
            break;
        }

        //----- Use object detector ----//
        case 2:{
            // Initialise the blob tracker
            blob_detector = new SAEdetector(params);
            break;
        }
        default:
            throw std::runtime_error("main.cpp: Unknown params.f_select_target in target setup.");
    }

    //-------------------------------------------//
    //----      Iterate through Events       ----//
    //-------------------------------------------//
    double ts_last = -1;

    int event_counter = 0;
    int detection_event_count = 0;

    int associated_events = 0;

    // Event data indexing (for load all events)
    int current_index = event_loader.start_index;
    
    while (true) {
        //----------------------------------//
        //----        Load Event        ----//
        //----------------------------------//
        Eigen::Vector4d e;

        if (params.f_load_all_events){ 
            e = all_events[current_index];
            if ((params.f_process_start_time) & (e(2) > event_loader.end_time) || (params.f_process_start_events) & (e(2) > event_loader.end_index)){
                break;
            }
            current_index++;
        }
        else {
            e = event_loader.getEvent();
            if ((params.f_process_start_time) & (e(2) > event_loader.end_time) || (params.f_process_start_events) & (event_loader.getCount() > event_loader.end_index)){
                break;
            }
        }

        if (e(0) == -1){ // end of file indicator
            break;
        }
        if (ts_last == -1){
            ts_last = e(2);
        }
        event_counter ++;

        // ----------------------------------//
        // ----     Data Association     ----//
        // ----------------------------------//
        // Perform data association
        std::tuple<int, std::vector<int>, int> association_output = data_assocation.performDataAssociation(e, track_manager.getOutputTracks(), track_manager.getCandidateTracks());

        // If the event is associated to either an output or candidate track
        if ((std::get<0>(association_output) == 1) || (std::get<0>(association_output) == 2)){
            std::pair<int, std::vector<int>> association = {std::get<0>(association_output), std::get<1>(association_output)};
            track_manager.updateTracks(e, association);
            associated_events++;

        }
        // If the event is not associated to an existing track, and the target detector is being used
        else if ((std::get<0>(association_output) == 3) && (params.f_select_target == 2)){
            // Add event to detector
            blob_detector->addEvent(e);

            // Perform detection for the new event every 10 events
            detection_event_count++;

            if ((detection_event_count > params.SAE_operation_rate) && (track_manager.f_new_tracks_allowed)){
                Eigen::Vector<double, 5> detector_output = blob_detector->performDetection(e);
                if((detector_output(0) != -1)){
                    track_manager.createNewCandidate(detector_output);
                }
                detection_event_count = 0; 
           }
        }

        if (std::get<2>(association_output) > 0){
            track_manager.getOutputTrack(std::get<2>(association_output)-1)->setTime(e(2));
        }
        else if (std::get<2>(association_output) < 0) {
            track_manager.getCandidateTrack(abs(std::get<2>(association_output))-1)->setTime(e(2));
        }

        //----------------------------------//
        //----     Track Management     ----//
        //----------------------------------//
        // Evaluate existing tracks, check which tracks should be deleted or promoted.
        track_manager.updateTime(e(2));
        
        // Perform track management after a number of events
        if ((event_counter % params.TM_evaluation_event_rate == 0) & (params.f_evaluate_tracks)){
            track_manager.evaluateOutputTracks();
            track_manager.evaluateCandidateTracks();
        }
        // Delete tracks if they have gone out of frame or haven't received enough events
        else if (event_counter % params.TM_evaluation_event_rate == 0){
            track_manager.evaluateTracksBasic();
        }

        //----------------------------------//
        //----      Visualisation       ----//
        //----------------------------------//
        // Update the visualiser
        visualiser.addEvent(e, std::get<0>(association_output));

        // Display if timer flag is set
        if (visualiser.f_display == 1){
            int visualiser_status = visualiser.display(track_manager.getOutputTracks(), track_manager.getCandidateTracks());
            visualiser.save(track_manager.getOutputTracks(), track_manager.getCandidateTracks());

            // "q" has been pressed 
            if (visualiser_status == -1){
                break;
            }
            // New targets have been selected, we need to reinitialise the trackers
            if (visualiser_status == 2){
                track_manager.addSelectedPoints(visualiser.getSelectedPoints(), e(2));
            }
        }   
    }

    // Close all windows
    std::cout << "Clean up \n";
    visualiser.clean_up();    
    track_manager.clean_up();
}







//---------------------------------------------------------------//
//-----                FUNCTION DEFINITIONS                 -----//
//---------------------------------------------------------------//

void print_help(){
  std::cout << "Usage: aeb_tracker\n"
            << "  -i [Required]  : Specify the name of the input configuration file\n"
            << "                        - bees        (for bees.yaml) \n";
}

void check_parameters(Parameters &params){
    // Define some allowed values
    std::set ALLOWED_F_SELECT_TARGET = {0, 1, 2};

    // Check values
    if (auto search = ALLOWED_F_SELECT_TARGET.find(params.f_select_target); search == ALLOWED_F_SELECT_TARGET.end()){
        throw std::runtime_error("Unknown Parameter: f_select_target");
    }
}