//---- Standard Headers ----//
#include <eigen3/Eigen/Dense>
#include <boost/circular_buffer.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <torch/script.h>

//---- User Headers ----//
#include "Kalman.hpp"
#include "parameters.hpp"
#include "HighPassFilter.hpp"


//************************************************************************************//
//**********                        INITIALISATION                          **********//
//************************************************************************************//
// Object constructors
KalmanFilter::KalmanFilter(const Parameters &params){
  initialiseFilterParameters(params);
  alpha = params.alpha;
  evaluation_event_count_threshold = params.events_between_classification;
  m_params = params;
}

KalmanFilter::KalmanFilter(const Parameters &params, Eigen::Vector<double,10> x0, double ts){
  initialiseFilterParameters(params);
  updateStartPoint(x0);
  ts_last = ts;
  alpha = params.alpha;
  evaluation_event_count_threshold = params.events_between_classification;
  m_params = params;
}

KalmanFilter::KalmanFilter(const Parameters &params, Eigen::Vector<double,10> x0, double ts, int unique_ID){
  initialiseFilterParameters(params);
  updateStartPoint(x0);
  ts_last = ts;
  ID = unique_ID;
  alpha = params.alpha;
  evaluation_event_count_threshold = params.events_between_classification;
  m_params = params;
}


// Update the starting point of the filter
void KalmanFilter::updateStartPoint(Eigen::MatrixXd x0){
  // Ensure initial lambda is correct
  if (x0(4) != lambda_init){
    x0(4) = lambda_init;
  } 
  if (x0(5) != lambda_init){
    x0(5) = lambda_init;
  } 
  x_hat = x0;
}

// Initialise the state variables
void KalmanFilter::initialiseFilterParameters(const Parameters &params) {

  ts_last = 0;

  distance_threshold = params.dist_threshold;  

  //---- Define State Matrices ----//
  I.setIdentity();
  
  // Set constant of F, dt is added each iteration
  F.setIdentity();

  lambda_init = params.lambda_init;
  x_hat.setZero();
  x_hat(4) = params.lambda_init;
  x_hat(5) = params.lambda_init;

  P.setZero();
  P.diagonal() << params.var_x, params.var_y, params.var_vx,
                  params.var_vy, params.var_lambda_1, params.var_lambda_2, params.var_theta,
                  params.var_q, params.var_delta, params.var_delta; 

  m_q_v = params.q_vx;
  m_q_v_update_event_count = params.q_v_update_event_count;
  m_q_v_update_ratio = params.q_v_update_ratio;

  Q.setZero();
  Q.diagonal() << params.q_x, params.q_y, params.q_vx_init, params.q_vy_init,
                  params.q_lambda_1, params.q_lambda_2, params.q_theta, params.q_q, params.q_delta, params.q_delta;

  R.setIdentity();

  //---- Initialise the ring buffers ----//
  ring_buffer_len = params.ring_buffer_len;

  ring_buffer_e.set_capacity(ring_buffer_len);
  ring_buffer_x.set_capacity(ring_buffer_len);
  ring_buffer_theta.set_capacity(ring_buffer_len);
  ring_buffer_delta.set_capacity(ring_buffer_len);
  
  ring_buffer_e.clear();
  ring_buffer_x.clear();
  ring_buffer_theta.clear();
  ring_buffer_delta.clear();

  evaluation_buffer_size = params.classifier_evaluation_buffer_size;
  evaluation_buffer.set_capacity(evaluation_buffer_size);
  evaluation_buffer.clear();

  f_initialised = 1;

}


//************************************************************************************//
//**********                    INTERACTION FUNCTIONS                       **********//
//************************************************************************************//
void KalmanFilter::predict(Eigen::Vector4d e){
  // Perform Kalman prediction
  prediction(e);
  // Update intensity patch
  addEventToIntensityPatch(e);
  // Update counter
  event_count++;
  event_count_since_evaluation++;
  // Check and set evaluation flag;
  if (event_count_since_evaluation > evaluation_event_count_threshold){
    f_evaluate = 1;
  }
}

void KalmanFilter::predictAndUpdate(Eigen::Vector4d e){
  // Perform Kalman prediction and update
  prediction(e);
  update(e);
  // Update intensity patch
  addEventToIntensityPatch(e);
  // Update counter
  event_count++;
  event_count_since_evaluation++;
  // Check and set evaluation flag;
  if (event_count_since_evaluation > evaluation_event_count_threshold){
    f_evaluate = 1;
  }
}




//************************************************************************************//
//**********                        PREDICTION STEP                         **********//
//************************************************************************************//
// Perform the EKF prediction step
void KalmanFilter::prediction(Eigen::Vector4d e){

  if (!f_initialised)
    throw std::runtime_error("Filter is not f_initialised!");

  //---- Update timing (dt) ----//
  double dt = (e(2) - ts_last);
  ts_last = e(2);

  double dt_alpha = 0.999;

  if (dt_average == -1){
    dt_average = dt;
  }
  else {
    dt_average = dt_alpha * dt_average + (1-dt_alpha) * dt;
  }
                                     
  //---- Update State Transition Matrix ----//
  F(0, 2) = dt;
  F(1, 3) = dt;
  F(6, 7) = dt;


  //----- Update Q for velocity -----//
  if (event_count > m_q_v_update_event_count){
    Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
    Eigen::Vector2d v = {x_hat(2), x_hat(3)};
    Eigen::Matrix2d v_outer_norm = (v*v.transpose())/v.norm();
    Eigen::Matrix2d Q_v = m_q_v * (m_q_v_update_ratio*v_outer_norm + (1-m_q_v_update_ratio)*(I2 - v_outer_norm));
    Q.block(2, 2, 2, 2) = Q_v;
  }

  //---- Apply Prediction ----//
  x_hat = (F*x_hat);
  P = F * P * F.transpose() + Q * dt;

}

//************************************************************************************//
//**********                         UPDATE STEP                            **********//
//************************************************************************************//
// Perform the EKF update step
void KalmanFilter::update(Eigen::Vector4d e){

  if (!f_initialised)
    throw std::runtime_error("Filter is not f_initialised!");
  
  // Define the measurement and Kalman gain matrices
  Eigen::Matrix<double, 3, 10> C = Eigen::Matrix<double, 3, 10>::Zero();
  Eigen::Matrix<double, 10, 3> K = Eigen::Matrix<double, 10, 3>::Zero();
  Eigen::Matrix2d Omega; 
  Omega << 0, -1, 1, 0;

  // Define a signed polarity variable (incase negative polarity is given by 0)
  int pol = (e(3) > 0 ) ? 1 : -1;
  int p = (e(3) > 0 ) ? 1 : -1;

  //----------------------------------------------------------//
  //-----            Compute Measurement Terms           -----//
  //----------------------------------------------------------//
  Eigen::Matrix2d rotation_m, A, A_rotation, Omiga, C_lambda, rotation_m_buf, temp;
  Eigen::Vector2d e_tilda, y_hat, y_hat_sum, C_lambda_diag, y_lambda, y_lambda_sum, e_tilda_buffer, y_hat_buffer, temp_diag;
  double y_square_sum, theta_buf;

  Omiga << 0, -1, 1, 0;


  rotation_m << cos(x_hat(6)), -sin(x_hat(6)), sin(x_hat(6)), cos(x_hat(6));
  A << 1 / x_hat(4), 0, 0, 1 / x_hat(5);
  A_rotation = rotation_m * A * rotation_m.transpose();

  if (p <= 0)  {
    e_tilda << e(0) + x_hat(8) - x_hat(0), e(1) + x_hat(9) - x_hat(1);
    ring_buffer_delta.push_back({x_hat(8), x_hat(9)});
  }
  else  {
    e_tilda << e(0) - x_hat(8) - x_hat(0), e(1) - x_hat(9) - x_hat(1);
    ring_buffer_delta.push_back({-x_hat(8), -x_hat(9)});
  }

  y_hat = A_rotation * e_tilda;
  y_hat_sum << 0, 0;
  y_square_sum = 0;

  C.block(0, 0, 2, 2) << -A_rotation;
  C.block(0, 2, 2, 2) << 0, 0, 0, 0;
  C_lambda_diag = A * A * rotation_m.transpose() * e_tilda;
  C_lambda << C_lambda_diag(0), 0, 0, C_lambda_diag(1);
  C.block(0, 4, 2, 2) << -rotation_m * C_lambda;
  C.block(0, 6, 2, 1) << rotation_m * (Omiga * A - A * Omiga) * rotation_m.transpose() * e_tilda;
  C.block(0, 7, 2, 1) << 0, 0;

  if (p <= 0){
    C.block(0, 8, 2, 2) << A_rotation;
  }
  else {
    C.block(0, 8, 2, 2) << -A_rotation;
  }

  C.block(2, 0, 1, 10) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  ring_buffer_e.push_back({e(0), e(1), e(2), p});
  ring_buffer_x.push_back({x_hat(0), x_hat(1)});
  ring_buffer_theta.push_back(x_hat(6));

  if (ring_buffer_e.size() >= 2)
  {
    // t1, t2, t3, ..., buf_dt[last] is the latest for buffer e, x and m
    // here use t1, t2, t3, ..., buf_dt[last]-1
    y_lambda << 0, 0;
    y_lambda_sum << 0, 0;
    for (int i = (ring_buffer_e.size() - 2); i >= 0; --i)
    {
      // Use the same time states and events
      e_tilda_buffer << std::get<0>(ring_buffer_e[i]) + ring_buffer_delta[i].first - ring_buffer_x[i].first,
          std::get<1>(ring_buffer_e[i]) + ring_buffer_delta[i].second - ring_buffer_x[i].second;

      theta_buf = ring_buffer_theta[i];
      rotation_m_buf << cos(theta_buf), -sin(theta_buf), sin(theta_buf), cos(theta_buf);
      y_hat_buffer = rotation_m_buf * A * rotation_m_buf.transpose() * e_tilda_buffer;
      y_hat_sum += y_hat_buffer;
      y_square_sum += pow(y_hat_buffer(0), 2) + pow(y_hat_buffer(1), 2);

      temp_diag = A * A * rotation_m_buf.transpose() * e_tilda_buffer;
      temp << temp_diag(0), 0, 0, temp_diag(1);

      y_lambda = -2 * y_hat_buffer.transpose() * rotation_m_buf * temp;
      y_lambda_sum += y_lambda;
    }
    C.block(2, 4, 1, 2) += y_lambda_sum.transpose();
  }

  Eigen::Matrix3d S;
  Eigen::Vector3d y_hat_fuse = Eigen::Vector3d::Zero();
  Eigen::Vector3d y_true = Eigen::Vector3d::Zero();

  // Innovation covariance
  R(2, 2) = 4 * ring_buffer_e.size();
  S = C * P * C.transpose() + R;
  K = P * C.transpose() * S.inverse();

  y_hat_fuse << y_hat, y_square_sum;
  y_true.coeffRef(2, 0) = 2 * ring_buffer_e.size();

  x_hat += (K * (y_true - y_hat_fuse)).transpose(); // y = {0, 0}
  P = (I - K * C) * P;

  // Put lower bounds on the shape parameters
  if (x_hat(4) < 1){ 
    x_hat(4) = 1;
  }
  if (x_hat(5) < 1){
    x_hat(5) = 1;
  }
}


void KalmanFilter::addEventToIntensityPatch(Eigen::Vector4d e){   
  // Shift the event by blob position (so patch is centered at (0,0))
  int x_index = std::round((e(0) - x_hat(0)) + (28/2));
  int y_index = std::round((e(1) - x_hat(1)) + (28/2));

  // If out of bound of patch, we won't add it
  if ((x_index < 0 || x_index >= 28) || (y_index < 0 || y_index >= 28)){
    return;
  }

  // Initialise timestamp array if needed
  if (f_intensity_patch_initialised == 0){
    ts_intensity_patch.setTo(e(2));
    f_intensity_patch_initialised = 1;
  } 

  //----- Add event to HPF image -----//
  int pol = (e(3) > 0) ? 1 : -1;
  intensity_patch.at<double>(y_index, x_index) += pol;
}



torch::Tensor KalmanFilter::getIntensityPatchTensor(){
  // Compute pixel-wise decay factor
  cv::Mat beta;
  cv::exp(-alpha * (ts_last - ts_intensity_patch), beta);
  intensity_patch = intensity_patch.mul(beta);

  // Update time array (time of last decay)
  ts_intensity_patch.setTo(ts_last);
  
  cv::Mat image = intensity_patch.clone();
  double min, max;
  cv::minMaxLoc(image, &min, &max);

  double scale = std::max(abs(min), max);

  image = image / (2*scale) + 0.5;
  cv::minMaxLoc(image, &min, &max);
  image.convertTo(image, CV_32F);//, 1.0 / 255);

  // Convert OpenCV image (H x W x C) to libtorch tensor (C x H x W)
  torch::Tensor input_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
  return input_tensor;
}


void KalmanFilter::resetEvaluationFlag(){
  // Reset flag and counter
  f_evaluate = 0;
  event_count_since_evaluation = 0;
}
