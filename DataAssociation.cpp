//---- Standard Headers ----//
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iomanip>
#include <iostream>

//---- User Headers ----//
#include "DataAssociation.hpp"
#include "parameters.hpp"


//------------------------------------//
//-----     PUBLIC FUNCTIONS     -----//
//------------------------------------//
// Constructor
DataAssociation::DataAssociation(const Parameters &params){
m_alpha = params.alpha_dist;
m_dist_threshold = params.dist_threshold;

// Define data association method.
m_f_use_mahal = params.f_mahal_distance;
m_mahal_dist_threshold = params.mahal_distance_threshold;

m_f_use_euc = params.f_euc_distance;

// Ensure 1 (and only 1) data association method is set
if (m_f_use_mahal == m_f_use_euc){
	throw std::runtime_error("DataAssociation.cpp: Undefined data association method.");
}

}

// std::pair<int, std::vector<int>, int> DataAssociation::performDataAssociation(const Eigen::Vector4d e, std::vector<KalmanFilter*> output_tracks, std::vector<KalmanFilter*> candidate_tracks){
std::tuple<int, std::vector<int>, int> DataAssociation::performDataAssociation(const Eigen::Vector4d e, std::vector<KalmanFilter*> output_tracks, std::vector<KalmanFilter*> candidate_tracks){

	// Unpack the input event
	int x = e(0);
	int y = e(1);
	double ts = e(2);
	int pol = (e(3) > 0 ) ? 1 : -1;

	std::vector<KalmanFilter*> current_output_tracks = output_tracks;
	std::vector<KalmanFilter*> current_candidate_tracks = candidate_tracks;

	// Set up some counters and flags  
	int f_associated_to_output_track = 0;
	int f_associated_to_candidate_track = 0;

	// Vectors to store any associated indexes
	std::vector<int> output_tracks_associated_indexes;
	std::vector<int> candidate_tracks_associated_indexes;


	double output_min_dist = 10000;
	double output_min_index = -1;


	//----- Output Tracks -----//
	// First iterate through the output tracks and check the distance
	for (int i = 0; i < current_output_tracks.size(); i++){
	
		// Compute distance beween event and blob center
		Eigen::Vector<double, 10> x_hat = current_output_tracks[i]->getState();
 
	    // Compute position error, including the delta offset (e - pol*Delta - p)
		Eigen::Vector2d e_tilde = {e(0) - pol*x_hat(8) - x_hat(0), 
								   e(1) - pol*x_hat(9) - x_hat(1)};	

		// Eigen::Vector2d e_tilde = {e(0) - x_hat(0), 
		// 						   e(1) - x_hat(1)};	


		// Perform simple data association to avoid calculations for events that are very far away
		double distance = sqrt(pow(e_tilde(0), 2) + pow(e_tilde(1), 2));

		if (distance < output_min_dist){
			output_min_dist = distance;
			output_min_index = i;
		}

		if (distance > 100){
			continue;
		}

		if (m_f_use_mahal){				   
			// Define covariance
			Eigen::Matrix2d R_theta;
			R_theta << cos(x_hat(6)), -sin(x_hat(6)), 
					sin(x_hat(6)), cos(x_hat(6));

			Eigen::Matrix2d A_inv = Eigen::Matrix2d::Zero();
			A_inv(0,0) = 1/x_hat(4);
			A_inv(1,1) = 1/x_hat(5);
			Eigen::Matrix2d Lamb_inv = R_theta * A_inv * R_theta.transpose();

			// Compute the Mahalanobis distance
			double mahal_dist = e_tilde.transpose() * Lamb_inv * e_tilde;

			// Update the number of blobs tracked
			if (mahal_dist < m_mahal_dist_threshold){
				output_tracks_associated_indexes.push_back(i);
				f_associated_to_output_track = 1;
			}
		}

		else if (m_f_use_euc){

			double gamma = std::exp(-m_alpha * (ts - current_output_tracks[i]->getTime()));
			double dist_threshold_track = gamma * current_output_tracks[i]->distance_threshold + (1 - gamma) * 2.5 * std::max(x_hat(4), x_hat(5));

			// Force lower limit for distance threshold
			current_output_tracks[i]->distance_threshold = std::max(dist_threshold_track, m_dist_threshold);
			

			if (distance < current_output_tracks[i]->distance_threshold){
				output_tracks_associated_indexes.push_back(i);
				f_associated_to_output_track = 1;
			}
		}
    }

	// If is associate with an output, we will return that
	if (f_associated_to_output_track){
		// std::pair<int, std::vector<int>> association_output;
		// association_output.first = 1; // we will associate '1' with an output track
		// association_output.second = output_tracks_associated_indexes;
		// // Update time
		// m_ts_last = ts;

		std::tuple<int, std::vector<int>, int> association_output;
		std::get<0>(association_output) = 1; // we will associate '1' with an output track
		std::get<1>(association_output) = output_tracks_associated_indexes;
		std::get<2>(association_output) = output_min_index + 1; 
		// Update time
		m_ts_last = ts;

		return association_output;
	}   


	double candidate_min_dist = 10000;
	double candidate_min_index = -1;

	//----- Candidate Tracks -----//
	// If not associated to an output, we will check the candidates
	for (int i = 0; i < current_candidate_tracks.size(); i++){
		// Compute distance beween event and blob center
		Eigen::Vector<double, 10> x_hat = current_candidate_tracks[i]->getState();

		Eigen::Vector2d e_tilde = {e(0) - pol*x_hat(8) - x_hat(0), 
								e(1) - pol*x_hat(9) - x_hat(1)};	
		// Eigen::Vector2d e_tilde = {e(0) - x_hat(0), 
		// 						   e(1) - x_hat(1)};	

		// Initially perform nearest neighbour will the filter initialises
		int f_euc_dist = 0;
		int f_mahal_dist = 0;

		double distance = sqrt(pow(e_tilde(0), 2) + pow(e_tilde(1), 2));
		
		if (distance < candidate_min_dist){
			candidate_min_dist = distance;
			candidate_min_index = i;
		}
		
		
		if (distance > 100){
			continue;
		}
		
		if (current_candidate_tracks[i]->event_count > 50){				
			// Define covariance
			Eigen::Matrix2d R_theta;
			R_theta << cos(x_hat(6)), -sin(x_hat(6)), 
					sin(x_hat(6)), cos(x_hat(6));

			Eigen::Matrix2d A_inv = Eigen::Matrix2d::Zero();
			A_inv(0,0) = 1/x_hat(4);
			A_inv(1,1) = 1/x_hat(5);
			Eigen::Matrix2d Lamb_inv = R_theta * A_inv * R_theta.transpose();

			distance = e_tilde.transpose() * Lamb_inv * e_tilde;
			f_mahal_dist = 1;
		}
		else {
			f_euc_dist = 1;
		}

		// Update the number of blobs tracked

		if (((distance < 13.82) & (f_mahal_dist == 1)) || ((distance < current_candidate_tracks[i]->distance_threshold) & (f_euc_dist == 1))){
			candidate_tracks_associated_indexes.push_back(i);
			f_associated_to_candidate_track = 1;
		}
	}



	if (f_associated_to_candidate_track){
		// // Update output pair
		// std::pair<int, std::vector<int>> association_output;
		// association_output.first = 2; // we will associate '1' with an output track
		// association_output.second = candidate_tracks_associated_indexes;
		// // Update time
		// m_ts_last = ts;

		std::tuple<int, std::vector<int>, int> association_output;
		std::get<0>(association_output) = 2; // we will associate '2' with a candidate track
		std::get<1>(association_output) = candidate_tracks_associated_indexes;
		std::get<2>(association_output) = -candidate_min_index - 1; 
		// Update time
		m_ts_last = ts;


		return association_output;
	}   

	//----- Not Associated -----//
	// Update ts last
	// m_ts_last = ts;
	// // Assign output values we can associate with no detection
	// std::pair<int, std::vector<int>> association_output;
	// association_output.first = 3;
	// association_output.second = {-1};

	// if (output_min_dist <= candidate_min_dist){
	// 	association_output.third = output_min_index;
	// }
	// else {
	// 	association_output.third = candidate_min_index;
	// }


	std::tuple<int, std::vector<int>, int> association_output;
	std::get<0>(association_output) = 3; // we will associate '3' for no association
	std::get<1>(association_output) = output_tracks_associated_indexes;

	if (output_min_dist <= candidate_min_dist){
		std::get<2>(association_output) = output_min_index + 1;
	}
	else {
		std::get<2>(association_output) = -candidate_min_index - 1;
	}

	// Update time
	m_ts_last = ts;


	return association_output;
}