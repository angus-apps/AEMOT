
#include <vector>
#include <fstream>
#include <filesystem>
#include <torch/script.h>

#include "TrackManager.hpp"
#include "Kalman.hpp"
#include "DataAssociation.hpp"
#include "parameters.hpp"
#include "Classifier.hpp"


//------------------------------------------//
//-----        PUBLIC FUNCTIONS        -----//
//------------------------------------------//

//************************************************************************************//
//**********                          CONSTRUCTOR                           **********//
//************************************************************************************//
TrackManager::TrackManager(const Parameters &params){
    //---- Store parameters ----//
    m_params = params;

    // Instantiate the classifier
    if (m_params.f_use_classifier){
        blob_classifier = new Classifier(m_params);
    }
};


//************************************************************************************//
//**********                         ACCESS TRACKS                          **********//
//************************************************************************************//
// Get the vector of all output/candidate tracks
std::vector<KalmanFilter*> TrackManager::getOutputTracks(){
    return output_tracks;
}

std::vector<KalmanFilter*> TrackManager::getCandidateTracks(){
    return candidate_tracks;
}

// Access individual tracks
KalmanFilter* TrackManager::getOutputTrack(int id){
    if (id >= output_tracks.size() || id < 0){
        throw std::runtime_error("Requested ID out of range of Output Tracks");
    }
    return output_tracks[id];
}

KalmanFilter* TrackManager::getCandidateTrack(int id){
    if (id >= candidate_tracks.size() || id < 0){
        throw std::runtime_error("Requested ID out of range of Candidate Tracks");
    }
    return candidate_tracks[id];
}


//************************************************************************************//
//**********                       ADD MANUAL POINTS                        **********//
//************************************************************************************//
// Create trackers from the selected points
void TrackManager::addSelectedPoints(std::vector<cv::Point> selected_points, double ts){
    // Delete the existing trackers
    std::cout << "Initialising new tracks" << std::endl;
    for (int id = 0; id < output_tracks.size(); id++){
        std::cout << "Tracks are deleted" << std::endl;
        deleteTrack(output_tracks, id);
    }
    output_tracks.clear();

    // Initialise the blobs
    for (int i = 0; i < selected_points.size(); i++){
        Eigen::Vector<double, 10> x0 = Eigen::Vector<double, 10>::Zero();
        x0(0) = selected_points[i].x;
        x0(1) = selected_points[i].y;
        x0(4) = m_params.lambda_init;
        x0(5) = m_params.lambda_init;
        KalmanFilter* new_kf = new KalmanFilter(m_params, x0, ts, unique_ID);
        
        output_tracks.push_back(new_kf);
        unique_ID++;
    }   
}


//************************************************************************************//
//**********                        SPAWN CANDIDATES                        **********//
//************************************************************************************//
// Create new candidate tracker
void TrackManager::createNewCandidate(Eigen::Vector<double,5> new_point){
    Eigen::Vector<double, 10> x0_new_track = Eigen::Vector<double, 10>::Zero();
    x0_new_track(0) = new_point(0);
    x0_new_track(1) = new_point(1);
    x0_new_track(2) = new_point(2);
    x0_new_track(3) = new_point(3);
    x0_new_track(4) = m_params.lambda_init;
    x0_new_track(5) = m_params.lambda_init;
    KalmanFilter* new_kf = new KalmanFilter(m_params, x0_new_track, new_point(4), unique_ID);
    unique_ID++;
    candidate_tracks.push_back(new_kf);
}


//************************************************************************************//
//**********                         UPDATE TRACKS                          **********//
//************************************************************************************//
// Update the tracks associated with the new event
void TrackManager::updateTracks(Eigen::Vector4d e, std::pair<int, std::vector<int>> association){
    // Keep track of how many tracks we currently have
    m_total_tracks = candidate_tracks.size() + output_tracks.size();
    if (m_total_tracks > m_params.MAX_NUMBER_OF_TRACKS){
        f_new_tracks_allowed = 0;
        std::cout << "Track limit reached \t Output: " << output_tracks.size() << "\t Candidates: " << candidate_tracks.size() << std::endl;
    }
    else {
        f_new_tracks_allowed = 1; 
    }

    switch (association.first){
        //----- Associated to Output Track -----//
        case 1: {
            std::vector<int> associated_indexes = association.second;
            // If it is only associated to one blob, we will update that blob
            if (associated_indexes.size() == 1){
                int idx = associated_indexes[0];
                output_tracks[idx]->predictAndUpdate(e);
            }
            // Handle the case where an event is assigned to more than one blob
            if (associated_indexes.size() > 1){
                // Perform prediction for associated blobs with no measurement (forward integrate through collision)
                for (int idx = 0; idx < associated_indexes.size(); idx++){
                    output_tracks[idx]->predict(e);
                }
            }
            m_ts_now = e(2);
            break;  
        }

        //----- Associated to Candidate Track -----//
        case 2: {
            std::vector<int> associated_indexes = association.second;
            // If it is only associated to one blob, we will update that blob
            if (associated_indexes.size() == 1){
                int idx = associated_indexes[0];
                candidate_tracks[idx]->predictAndUpdate(e);
            }
            // Handle the case where an event is assigned to more than one blob
            if (associated_indexes.size() > 1){
                // Perform prediction for associated blobs with no measurement (forward integrate through collision)
                for (int idx = 0; idx < associated_indexes.size(); idx++){
                    candidate_tracks[idx]->predict(e);
                }
            }
            m_ts_now = e(2);
            break;  
        }
    }
}


//************************************************************************************//
//**********                          EVALUATION                            **********//
//************************************************************************************//
// Evaluate the output tracks
void TrackManager::evaluateOutputTracks(){
    // Evaluate each output tracker and see if it should be deleted
    for (int id = output_tracks.size()-1; id >= 0; id--){
        Eigen::Vector<double, 10> x_hat_i = output_tracks[id]->getState();
        Eigen::Vector<double, 10> cov_i = output_tracks[id]->getCovariance().diagonal();

        // Check basic deletion criteria every iteration (out of frame or long time between events)
        bool out_of_frame = ((x_hat_i(0) < 1) || (x_hat_i(0) > m_params.width-1)) || ((x_hat_i(1) < 1) || (x_hat_i(1) > m_params.height-1));
        bool inactive_1 = (abs(m_ts_now - output_tracks[id]->getTime()) > m_params.candidate_dt_termination) && (output_tracks[id]->event_count>10);
        bool inactive_2 = (abs(m_ts_now - output_tracks[id]->getTime()) > 50*output_tracks[id]->dt_average) && (output_tracks[id]->event_count>10);
        bool size_imbalance = (x_hat_i(4) > 10* x_hat_i(5)) || (x_hat_i(5) > 10* x_hat_i(4));

        //-------------------------------------------------------//
        //-----                 CLASSIFIER                  -----//
        //-------------------------------------------------------//
        if (m_params.f_use_classifier){

            // Evaluate the classifier if the flag is set
            bool classifier_delete = 0;

            if (output_tracks[id]->f_evaluate){
                // Classify the track and add to evaluation buffer
                float prediction = blob_classifier->runClassifier(output_tracks[id]->getIntensityPatchTensor());
                int pred_int = (prediction > m_params.classifier_threshold) ? 1 : 0;
                output_tracks[id]->evaluation_buffer.push_back(pred_int);

                // Sum the elements in the buffer
                float evaluation_buffer_sum = std::accumulate(output_tracks[id]->evaluation_buffer.begin(), output_tracks[id]->evaluation_buffer.end(), 0);

                // Delete track if buffered predictions are all 0
                if ((evaluation_buffer_sum == 0 && (output_tracks[id]->evaluation_buffer.size() == output_tracks[id]->evaluation_buffer_size))){
                    classifier_delete = 1;
                }

                // Reset evaluation flag
                output_tracks[id]->resetEvaluationFlag();
            }

            if (out_of_frame || inactive_1 || inactive_2 || size_imbalance || classifier_delete){
                deleteTrack(output_tracks, id);
                continue;
            }
        }

        //-------------------------------------------------------//
        //-----                NO CLASSIFIER                -----//
        //-------------------------------------------------------//       
        else {
            // Delete if necessary - adjust these to suit scenario
            if (out_of_frame || inactive_1 || inactive_2 || size_imbalance){
                deleteTrack(output_tracks, id);
            }
        }

    }
}


//---- Evaluate the Candidate tracks ----//
void TrackManager::evaluateCandidateTracks(){
    // Evaluate each candidate tracker to see if it should be promoted or deleted
    for (int id = candidate_tracks.size()-1; id >= 0; id--){
        Eigen::Vector<double, 10> x_hat_i = candidate_tracks[id]->getState();
        Eigen::Vector<double, 10> cov_i = candidate_tracks[id]->getCovariance().diagonal();


        //-------------------------------------------------------//
        //-----                 CLASSIFIER                  -----//
        //-------------------------------------------------------//
        if (m_params.f_use_classifier){
            // Check some basic criteria for deletion ---> out of frame, long time between events, or the lambda ration is large
            // If we have reached the maximum number of tracks, we will make the criteria harsher to clear way for new, real tracks.
            bool out_of_frame, inactive_1, inactive_2, size_imbalance;

            out_of_frame = ((x_hat_i(0) < 1) || (x_hat_i(0) > m_params.width-1)) || ((x_hat_i(1) < 1) || (x_hat_i(1) > m_params.height-1));
            inactive_1 = (m_ts_now - candidate_tracks[id]->getTime() > 0.5*m_params.candidate_dt_termination);
            inactive_2 = (abs(m_ts_now - candidate_tracks[id]->getTime()) > 25*candidate_tracks[id]->dt_average) && (candidate_tracks[id]->event_count>10);
            size_imbalance = (x_hat_i(4) > 10* x_hat_i(5)) || (x_hat_i(5) > 10* x_hat_i(4));

            // Evaluate the classifier if the flag is set
            bool classifier_delete = 0;
            bool classifier_promote = 0;

            //----- Run classifier if flag is set -----//
            int pred_int = 0;
            if (candidate_tracks[id]->f_evaluate){
                float prediction = blob_classifier->runClassifier(candidate_tracks[id]->getIntensityPatchTensor());
                pred_int = (prediction >= m_params.classifier_threshold) ? 1 : 0;
                candidate_tracks[id]->evaluation_buffer.push_back(pred_int);

                // Sum the elements in the buffer
                float evaluation_buffer_sum = std::accumulate(candidate_tracks[id]->evaluation_buffer.begin(), candidate_tracks[id]->evaluation_buffer.end(), 0);

                // Delete track if not enough predictions are 1
                if ((evaluation_buffer_sum <= 0.5*candidate_tracks[id]->evaluation_buffer.size() && (candidate_tracks[id]->evaluation_buffer.size() == candidate_tracks[id]->evaluation_buffer_size))){
                    classifier_delete = 1;
                }

                // Set promote flag if all predictions in the buffer are 1
                if (((evaluation_buffer_sum == candidate_tracks[id]->evaluation_buffer.size() && (candidate_tracks[id]->evaluation_buffer.size() == candidate_tracks[id]->evaluation_buffer_size)))){
                    classifier_promote = 1;
                }

                // Reset evaluation flag
                candidate_tracks[id]->resetEvaluationFlag();
            }

            // Apply deletion or promotion
            if (out_of_frame || inactive_1 || inactive_2 || size_imbalance || classifier_delete){
                deleteTrack(candidate_tracks, id);
                continue;
            }

            if (classifier_promote){
                promoteCandidate(id);
            }

        }

        //-------------------------------------------------------//
        //-----               NO CLASSIFIER                 -----//
        //-------------------------------------------------------//
        else {
            //---- Check for Delete ----//
            // Check some conditions
            bool out_of_frame = ((x_hat_i(0) < 1) || (x_hat_i(0) > m_params.width-1)) || ((x_hat_i(1) < 1) || (x_hat_i(1) > m_params.height-1));
            bool inactive = (m_ts_now - candidate_tracks[id]->getTime() > m_params.candidate_dt_termination);

            if (out_of_frame || inactive){
                deleteTrack(candidate_tracks, id);
                continue;
            }

            //---- Check for promotion ----//
            int cond_1 = (cov_i(0) < m_params.candidate_pos_cov_threshold && cov_i(1) < m_params.candidate_pos_cov_threshold); // confidence in position
            int cond_2 = (x_hat_i(2) >= m_params.candidate_vel_threshold && x_hat_i(3) >= m_params.candidate_vel_threshold); // velocity value
            int cond_3 = (cov_i(4) < m_params.candidate_lamb_cov_threshold && cov_i(5) < m_params.candidate_lamb_cov_threshold); // confidence in size
            int cond_5 = (x_hat_i(4) < m_params.candidate_lamb_size_threshold) && (x_hat_i(5) < m_params.candidate_lamb_size_threshold);

            // if (cond_1 & cond_3){
            if (cond_1 && 1 && 1 && cond_5){
                promoteCandidate(id);
            }
        }
    } 
}

void TrackManager::evaluateTracksBasic(){
    int border_threshold = 10; // pixels from the edge
    double dt_threshold = 0.1; // time without an event

    // Delete tracks that are out of the frame
    for (int id = output_tracks.size()-1; id >= 0; id--){
        Eigen::Vector<double, 10> x_hat_i = output_tracks[id]->getState();
        bool out_of_frame = ((x_hat_i(0) < border_threshold) || (x_hat_i(0) > m_params.width-border_threshold)) || ((x_hat_i(1) < border_threshold) || (x_hat_i(1) > m_params.height-border_threshold));
        bool inactive = (m_ts_now - output_tracks[id]->getTime() > dt_threshold);
        if (out_of_frame || inactive){
            deleteTrack(output_tracks, id);
        }
    }

    for (int id = candidate_tracks.size()-1; id >= 0; id--){
        Eigen::Vector<double, 10> x_hat_i = candidate_tracks[id]->getState();
        bool out_of_frame = ((x_hat_i(0) < border_threshold) || (x_hat_i(0) > m_params.width-border_threshold)) || ((x_hat_i(1) < border_threshold) || (x_hat_i(1) > m_params.height-border_threshold));
        bool inactive = (m_ts_now - candidate_tracks[id]->getTime() > dt_threshold);
        if (out_of_frame){
            deleteTrack(candidate_tracks, id);
        }
    }
}




void TrackManager::clean_up(){
}


//------------------------------------------//
//-----        PRIVATE FUNCTIONS       -----//
//------------------------------------------//

void TrackManager::deleteTrack(std::vector<KalmanFilter*> &track_array, int track_id){
    KalmanFilter* ptr_to_delete = track_array[track_id];
    delete ptr_to_delete;
    track_array.erase(track_array.begin() + track_id);
}

void TrackManager::promoteCandidate(int id){
    // Move tracker object from candidate vector to output vector
    output_tracks.push_back(candidate_tracks[id]);
    // Remove tracker pointer from candidate list
    candidate_tracks.erase(candidate_tracks.begin() + id);
}


void TrackManager::demoteCandidate(int id){
    // Move tracker object from output vector to candidate vector
    candidate_tracks.push_back(output_tracks[id]);
    // Remove tracker pointer from output list
    output_tracks.erase(output_tracks.begin() + id);
}




