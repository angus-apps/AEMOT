#ifndef VISUALISER_HPP
#define VISUALISER_HPP

    // Eigen::Matrix<double, 10, 1>    x_hat;
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include <vector>

//---- User Headers ----//
#include "parameters.hpp"
#include "HighPassFilter.hpp"
#include "Kalman.hpp"

class Visualiser
{
public:
    Visualiser(Parameters &params);
    ~Visualiser();
    void addEvent(Eigen::Vector4d e);
    void addEvent(Eigen::Vector4d e, int data_association);
    void displayForSelect();
    int display();
    int display(const std::vector<KalmanFilter*> output_tracks, const std::vector<KalmanFilter*> candidate_tracks);
    void save(const std::vector<KalmanFilter*> output_tracks, const std::vector<KalmanFilter*> candidate_tracks);
    void initialiseAllEventsHPF(HighPassFilter* hpf_target_select);
    std::vector<cv::Point> getSelectedPoints();
    void clean_up();
    double getTime(){ return m_t_current; };
    bool f_display = 0;



private:

    void setUpCVDisplay();
    void setUpCVSaving();
    void setUpHPFimages(Parameters &params);
    int displayWait();
    // static void onMouse(int event, int x, int y, int flags, void *userdata);

    void display_with_tracks(cv::String window_name, cv::Mat image_array, const std::vector<KalmanFilter*> tracks);
    void display_without_tracks(cv::String window_name, cv::Mat image_array);
    

    cv::Mat generateImage(cv::Mat image_array);
    cv::Mat generateImageWithTracks(cv::Mat image_array, std::vector<KalmanFilter*> tracks);


    //----- Private Variables -----// 
    int m_f_verbose = 0;
    std::vector<cv::Point> m_selected_points;

    // Create pointers to HPF images
    HighPassFilter* m_hpf_all_events;
    HighPassFilter* m_hpf_track_events;
    HighPassFilter* m_hpf_residual_events;

    std::string m_filepath;
    int m_height, m_width;
    double m_t_last_display,
           m_t_current,
           m_dt_publish,
           m_publish_framerate,
           m_alpha,
           m_contrast_threshold;

    std::chrono::time_point<std::chrono::high_resolution_clock> m_chrono_time_last = std::chrono::high_resolution_clock::now();


    //----- Save Flags -----//
    bool m_f_target_select;
    bool m_f_hpf_all_events_master;

    bool m_f_save_blob_tracks, 
         m_f_save_candidate_tracks, 
         m_f_save_all_events, 
         m_f_save_track_events, 
         m_f_save_residual_events;
    bool m_f_show_blob_tracks, 
         m_f_show_candidate_tracks, 
         m_f_show_all_events, 
         m_f_show_track_events, 
         m_f_show_residual_events;


    //----- Window Names -----//
    cv::String m_tracks_window          = "tracks", 
               m_candidate_window       = "candidates",
               m_all_events_window      = "all",
               m_track_event_window     = "track_events",
               m_residual_event_window  = "residual_events",
               m_target_select_window   = "Target Select Window";

    //----- Video File Paths -----//
    std::string m_tracks_video_path, 
                m_output_candidates_path, 
                m_output_all_video_path, 
                m_output_track_video_path, 
                m_output_residual_video_path;
    
    cv::VideoWriter m_writer_tracks,
                    m_writer_candidates,
                    m_writer_all_events, 
                    m_writer_track_events,
                    m_writer_residual_events;

};


#endif //VISUALISER_HPP