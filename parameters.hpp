#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "yaml-cpp/yaml.h"
#include <string>


struct Parameters
{
  int width;
  int height;
  double dist_threshold;
  int publish_framerate;
  double contrast_threshold;
  double alpha;
  double alpha_dist;
  int save_image_flag;
  int f_disp_covariance;

  //---- Classifier -----//
  int f_use_classifier; 
  std::string classifier_model;
  std::string classifer_directory;
  int events_between_classification;
  int classifier_evaluation_buffer_size;
  double classifier_threshold;

  //---- Preloadind Data -----//
  int f_load_all_events; 
  int f_process_start_time;
  int f_process_start_events;
  double process_ts_start;
  double process_ts_end;
  double event_num_start;
  double event_num_end;

  //---- Verbose Flags ----//
  int f_verbose_eventloader;
  int f_verbose_visualiser;
  int f_verbose_hpf;
  int f_verbose_detector;
  int f_verbose_SAE;  
  int f_verbose_kf;
  int f_verbose_classifier;


  //---- Display Flags ----//
  int f_show_blob_tracks;
  int f_show_candidate_tracks;
  int f_show_all_events;
  int f_show_track_events;
  int f_show_residual_events;
  int f_show_intensity_patch;

  //---- Save Flags ----//
  int f_save_blob_tracks;
  int f_save_candidate_tracks;
  int f_save_all_events;
  int f_save_track_events;
  int f_save_residual_events;
  int f_save_event_csv;
  int f_save_image;

  int f_log_data;
  
  //---- Detector ----//
  std::string SAE_method;
  int SAE_operation_rate; 
  int SAE_ksize;
  double SAE_alpha;
  double SAE_angle_thresh;
  double SAE_min_contributions;
  double SAE_detection_threshold;
  double SAE_min_active_pixels;  

  //---- Pretracker ----//
  int MAX_NUMBER_OF_TRACKS;
  double candidate_pos_cov_threshold;
  double candidate_vel_cov_threshold;
  double candidate_lamb_cov_threshold;
  
  double candidate_theta_cov_threshold;
  double candidate_q_cov_threshold;
  double candidate_lamb_size_threshold;

  double candidate_vel_threshold;
  double candidate_dt_termination;
  double candidate_age_termination;
  double candidate_evol_age;

  //---- Data Association ----// 
  int f_mahal_distance;               
  double mahal_distance_threshold;       
  int f_euc_distance;
  
    
  //---- Track Manager ----//
  int f_evaluate_tracks;
  int TM_evaluation_event_rate;


  int f_save_track;
  int ring_buffer_len;
  std::string input_data_name;
  std::string input_folder_path;
  double var_x;
  double var_y;
  double var_vx;
  double var_vy;
  double var_lambda_1;
  double var_lambda_2;
  double var_theta;
  double var_q;
  double var_delta;
  double q_x;
  double q_y;
  double q_vx;
  double q_vy;
  double q_lambda_1;
  double q_lambda_2;
  double q_theta;
  double q_q;
  double q_delta;
  double q_vx_init;
  double q_vy_init;

  double q_v_update_event_count;  
  double q_v_update_ratio;

  std::vector<double> position_x_init;
  std::vector<double> position_y_init;
  double lambda_init;

  int f_select_target;
  int f_event_format;

};

Parameters loadParametersFromYAML(const std::string &yaml_file_path);

#endif // PARAMETERS_HPP
