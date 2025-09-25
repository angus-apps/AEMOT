#ifndef TRACKMANAGER_HPP
#define TRACKMANAGER_HPP

#include <vector>
#include <opencv4/opencv2/core/core.hpp>
#include <eigen3/Eigen/Dense>
#include <fstream>

#include "parameters.hpp"
#include "Kalman.hpp"
#include "Classifier.hpp"

class TrackManager
{

public:
    //------------------------------------------//
    //-----        PUBLIC FUNCTIONS        -----//
    //------------------------------------------//
    // Constructor Functions
    TrackManager(const Parameters &params);

    // Add track to output
    void addSelectedPoints(std::vector<cv::Point> selected_points, double ts);
    void createNewCandidate(Eigen::Vector<double,5> new_point);

    // Perform Kalman Filter operations
    void updateTracks(Eigen::Vector4d e, std::pair<int, std::vector<int>> association);
    void updateTime(double ts){ m_ts_now = ts; };

    // Evaluate tracks for deletion/promotion
    void evaluateOutputTracks();
    void evaluateCandidateTracks();
    void evaluateTracksBasic();


    // Access the track objects
    std::vector<KalmanFilter*> getOutputTracks();
    KalmanFilter* getOutputTrack(int id);
    std::vector<KalmanFilter*> getCandidateTracks();
    KalmanFilter* getCandidateTrack(int id);

    void clean_up();

    //------------------------------------------//
    //-----        PUBLIC VARIABLES        -----//
    //------------------------------------------//
    bool f_new_tracks_allowed = 1;

private:
    //------------------------------------------//
    //-----        PRIVATE FUNCTIONS       -----//
    //------------------------------------------//
    void addTrack(std::vector<cv::Point> );
    void deleteTrack(std::vector<KalmanFilter*> &track_array, int track_id);
    void promoteCandidate(int id);
    void demoteCandidate(int id);

    //------------------------------------------//
    //-----        PRIVATE VARIABLES       -----//
    //------------------------------------------//
    // Vector to store the filters
    Parameters m_params;
    std::vector<KalmanFilter*> output_tracks, candidate_tracks; 
    int m_total_tracks = 0;

    // Evaluation parameters
    double m_ts_now = 0;
    int unique_ID = 1;

    // Classifier object
    Classifier* blob_classifier = NULL;

    // Data writer
    std::ofstream output_file;
};



#endif