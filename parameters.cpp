

#include "parameters.hpp"
#include "yaml-cpp/yaml.h"

Parameters loadParametersFromYAML(const std::string &yaml_file_path)
{
  const YAML::Node config = YAML::LoadFile(yaml_file_path);

  Parameters params;
  params.width = config["width"].as<int>();
  params.height = config["height"].as<int>();
  params.dist_threshold = config["dist_threshold"].as<double>();
  params.publish_framerate = config["publish_framerate"].as<int>();
  params.contrast_threshold = config["contrast_threshold"].as<double>();
  params.alpha = config["alpha"].as<double>();
  params.alpha_dist = config["alpha_dist"].as<double>();

  params.f_use_classifier = config["f_use_classifier"].as<int>();
  params.classifier_model = config["classifier_model"].as<std::string>();
  params.classifer_directory = config["classifer_directory"].as<std::string>();
  params.classifier_evaluation_buffer_size = config["classifier_evaluation_buffer_size"].as<int>();
  params.events_between_classification = config["events_between_classification"].as<int>();
  params.classifier_threshold = config["classifier_threshold"].as<double>();

  params.f_load_all_events = config["f_load_all_events"].as<int>();

  params.f_process_start_time = config["f_process_start_time"].as<int>();
  params.f_process_start_events = config["f_process_start_events"].as<int>();
  params.process_ts_start = config["process_ts_start"].as<double>();
  params.process_ts_end = config["process_ts_end"].as<double>();
  params.event_num_start = config["event_num_start"].as<double>();
  params.event_num_end = config["event_num_end"].as<double>();

  params.f_verbose_eventloader = config["f_verbose_eventloader"].as<int>();
  params.f_verbose_visualiser = config["f_verbose_visualiser"].as<int>();
  params.f_verbose_hpf = config["f_verbose_hpf"].as<int>();
  params.f_verbose_detector = config["f_verbose_detector"].as<int>();
  params.f_verbose_SAE = config["f_verbose_SAE"].as<int>();
  params.f_verbose_kf = config["f_verbose_kf"].as<int>();
  params.f_verbose_classifier = config["f_verbose_classifier"].as<int>();

  params.SAE_method = config["SAE_method"].as<std::string>();
  params.SAE_operation_rate = config["SAE_operation_rate"].as<int>();
  params.SAE_ksize = config["SAE_ksize"].as<int>();
  params.SAE_alpha = config["SAE_alpha"].as<double>();
  params.SAE_min_contributions = config["SAE_min_contributions"].as<double>();  
  params.SAE_detection_threshold = config["SAE_detection_threshold"].as<double>();
  params.SAE_min_active_pixels = config["SAE_min_active_pixels"].as<double>(); 

  params.f_mahal_distance = config["f_mahal_distance"].as<int>();
  params.mahal_distance_threshold = config["mahal_distance_threshold"].as<double>();
  params.f_euc_distance = config["f_euc_distance"].as<int>();

  params.f_evaluate_tracks = config["f_evaluate_tracks"].as<int>();
  params.TM_evaluation_event_rate = config["TM_evaluation_event_rate"].as<int>();

  params.MAX_NUMBER_OF_TRACKS = config["MAX_NUMBER_OF_TRACKS"].as<int>();
  params.candidate_pos_cov_threshold = config["candidate_pos_cov_threshold"].as<double>();
  params.candidate_vel_cov_threshold = config["candidate_vel_cov_threshold"].as<double>();
  params.candidate_lamb_cov_threshold = config["candidate_lamb_cov_threshold"].as<double>();
  params.candidate_theta_cov_threshold = config["candidate_theta_cov_threshold"].as<double>();
  params.candidate_q_cov_threshold = config["candidate_q_cov_threshold"].as<double>();
  
  
  params.candidate_vel_threshold = config["candidate_vel_threshold"].as<double>();
  params.candidate_lamb_size_threshold = config["candidate_lamb_size_threshold"].as<double>();
  params.candidate_dt_termination = config["candidate_dt_termination"].as<double>();
  params.candidate_age_termination = config["candidate_age_termination"].as<double>();
  params.candidate_evol_age = config["candidate_evol_age"].as<double>();

  params.f_show_blob_tracks = config["f_show_blob_tracks"].as <int>();
  params.f_show_candidate_tracks = config["f_show_candidate_tracks"].as <int>();
  params.f_show_all_events = config["f_show_all_events"].as<int>();
  params.f_show_track_events = config["f_show_track_events"].as<int>();
  params.f_show_residual_events = config["f_show_residual_events"].as<int>();
  params.f_show_intensity_patch = config["f_show_intensity_patch"].as<int>();
  
  params.f_save_blob_tracks = config["f_save_blob_tracks"].as<int>();
  params.f_save_candidate_tracks = config["f_save_candidate_tracks"].as<int>();
  params.f_save_all_events = config["f_save_all_events"].as<int>();
  params.f_save_track_events = config["f_save_track_events"].as<int>();
  params.f_save_residual_events = config["f_save_residual_events"].as<int>();
  params.f_save_event_csv = config["f_save_event_csv"].as<int>();
  params.f_save_image = config["f_save_image"].as<int>();
  
  params.f_log_data = config["f_log_data"].as<int>();

  params.ring_buffer_len = config["ring_buffer_len"].as<int>();

  params.input_data_name = config["input_data_name"].as<std::string>();

  params.f_save_track = config["f_save_track"].as<int>();
  params.f_select_target = config["f_select_target"].as<int>();
  params.input_folder_path = config["input_folder_path"].as<std::string>();
  params.f_event_format = config["f_event_format"].as<int>();

  params.var_x = config["var_x"].as<double>();
  params.var_y = config["var_y"].as<double>();
  params.var_vx = config["var_vx"].as<double>();
  params.var_vy = config["var_vy"].as<double>();
  params.var_lambda_1 = config["var_lambda_1"].as<double>();
  params.var_lambda_2 = config["var_lambda_2"].as<double>();
  params.var_theta = config["var_theta"].as<double>();
  params.var_q = config["var_q"].as<double>();
  params.var_delta = config["var_delta"].as<double>();
  params.q_x = config["q_x"].as<double>();
  params.position_x_init = config["position_x_init"].as<std::vector<double>>();
  params.position_y_init = config["position_y_init"].as<std::vector<double>>();
  params.lambda_init = config["lambda_init"].as<double>();

  params.q_y = config["q_y"].as<double>();
  params.q_vx = config["q_vx"].as<double>();
  params.q_vy = config["q_vy"].as<double>();
  params.q_lambda_1 = config["q_lambda_1"].as<double>();
  params.q_lambda_2 = config["q_lambda_2"].as<double>();
  params.q_theta = config["q_theta"].as<double>();
  params.q_q = config["q_q"].as<double>();
  params.q_delta = config["q_delta"].as<double>();
  params.q_vx_init = config["q_vx_init"].as<double>();
  params.q_vy_init = config["q_vy_init"].as<double>();

  params.q_v_update_event_count = config["q_v_update_event_count"].as<double>();
  params.q_v_update_ratio = config["q_v_update_ratio"].as<double>();


  return params;
}



