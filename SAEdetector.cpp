#include <eigen3/Eigen/Dense>
#include <iostream>
#include <numeric>
#include <utility>

#include "SAEdetector.hpp"

SAEdetector::SAEdetector(const Parameters &params)
{
    // Update properties from configuration file
    m_f_verbose = params.f_verbose_detector;
    m_width = params.width;
    m_height = params.height;
    m_ksize = params.SAE_ksize;
    m_method = params.SAE_method;
    m_alpha = params.SAE_alpha;
    m_min_contributions = params.SAE_min_contributions;
    m_detection_threshold = params.SAE_detection_threshold;
    m_min_active_pixels = params.SAE_min_active_pixels;

    // Initialise the SAE 
    SAE = new SurfaceOfActiveEvents(params);

    // Setup up storage images
    initialised_events.resize(m_height, m_width);
    initialised_events.setZero();
    init_events_patch.resize(m_ksize, m_ksize);
    init_events_patch.setZero();

    lx.resize(m_height, m_width);
    lx.setZero();
    lx_patch.resize(m_ksize, m_ksize);
    lx_patch.setZero();

    ly.resize(m_height, m_width);
    ly.setZero();
    ly_patch.resize(m_ksize, m_ksize);
    ly_patch.setZero();

    initialised_directions.resize(m_height, m_width);
    initialised_directions.setZero();
    init_directions_patch.resize(m_ksize, m_ksize);
    init_directions_patch.setZero();

    t_patch.resize(m_ksize, m_ksize);
    t_patch.setZero();


    // Set up patch points/pairs
    patch_offset = Eigen::VectorXd::LinSpaced(m_ksize, -m_ksize/2, m_ksize/2);

    for (int i = 0; i < m_ksize; i++){
        for (int j = 0; j < m_ksize; j++){
            patch_points.push_back(std::pair(i, j));
       }
    }

    // Print some stuff if we want to
    if (m_f_verbose){
        std::cout << "SAE Detector intialised \nalpha: " << m_alpha << "\ndetection_threshold: " << m_detection_threshold << "\nmin_contributions:  " << m_min_contributions << std::endl;
    }
}

SAEdetector::~SAEdetector(){
    delete SAE;
}


//-----------------------------------//
//-----     Public Functions    -----//
//-----------------------------------//
// Main function to interface with
//-----    Add event to SAE    -----//
void SAEdetector::addEvent(Eigen::Vector4d e){
    SAE->addEvent(e);
    initialised_events(int(e(1)), int(e(0))) = 1;
    m_t_current = e(2);
}


Eigen::Vector<double, 5> SAEdetector::performDetection(Eigen::Vector4d e){
    // addEvent(e);
    getPatches(e);
    computeDirectionRegression(e);
    Eigen::VectorXd detection_result = compareDirectionRegression(e);

    return detection_result;
}


Eigen::MatrixXd SAEdetector::getImage(){
    return SAE->getImage();
}

//-----------------------------------//
//-----    Private Functions    -----//
//-----------------------------------//

// Get the patches used for least squares regression
void SAEdetector::getPatches(Eigen::Vector4d e){
    
    // Create index values
    Eigen::VectorXd x_indexes = Eigen::VectorXd::Constant(m_ksize, e(0)) + patch_offset;
    Eigen::VectorXd y_indexes = Eigen::VectorXd::Constant(m_ksize, e(1)) + patch_offset;

    // Set the patch flag to false if the requested indexes are out of the frame
    if (x_indexes[0] < 0 | x_indexes[x_indexes.size()-1] >= m_width |
        y_indexes[0] < 0 | y_indexes[y_indexes.size()-1] >= m_height){
        m_f_patch_status = false;
        return;
    }

    // First get the initialised pixel patch to see if we should get the  others
    init_events_patch = initialised_events.block(y_indexes[0], x_indexes[0], m_ksize, m_ksize);
    
    if (init_events_patch.sum() < m_min_contributions){
        m_f_patch_status = false;
        return;
    }    

    t_patch = SAE->getImage().block(y_indexes[0], x_indexes[0], m_ksize, m_ksize);
    init_directions_patch = initialised_directions.block(y_indexes[0], x_indexes[0], m_ksize, m_ksize);
    lx_patch = lx.block(y_indexes[0], x_indexes[0], m_ksize, m_ksize);
    ly_patch = ly.block(y_indexes[0], x_indexes[0], m_ksize, m_ksize);

    m_f_patch_status = true;
}


//-----    Compute LSQ Regression    -----//
void SAEdetector::computeDirectionRegression(Eigen::Vector4d e){
    // If we weren't able to get the patches, we do not perform least squares
    if (m_f_patch_status == false){
        m_f_regression_status = false;
        return;
    }

    int num_init_points = init_events_patch.sum() - 1;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_init_points, 2);
    Eigen::MatrixXd W_inv = Eigen::MatrixXd::Zero(num_init_points, num_init_points);

    int idx = 0;
    for (int i = 0; i < patch_points.size(); i++){   
        // Get the indexes in the patch
        int x = patch_points[i].first;
        int y = patch_points[i].second;

        // If the points is not initialised or it is the center element (pixel we are looking at), skip it 
        if ((init_events_patch(y,x) == 0) || (patch_offset(x) == 0 && patch_offset(y) == 0)){
            continue;
        }

        // Add distance values to A
        A(idx,0) = patch_offset(x);
        A(idx,1) = patch_offset(y);

        // Compute dt term
        double exp_term = std::exp(2 * m_alpha * (m_t_current - t_patch(y, x)));
        W_inv(idx, idx) = 1/exp_term;

        idx++;
    }

    Eigen::Matrix2d res = A.transpose() * W_inv * A;

    // Compute eigenvector of smallest eigenvalue for l_perp, rotate 90deg 
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(res); 
    Eigen::Vector2d l_perp =  eigen_solver.eigenvectors().col(0);
    Eigen::Vector2d l = {-l_perp(1), l_perp(0)};

    //----- Store values -----//
    // Store direction estimate in whole array and patch
    lx(int(e(1)), int(e(0))) = l(0);
    ly(int(e(1)), int(e(0))) = l(1);

    lx_patch(int(m_ksize/2), int(m_ksize/2)) = l(0);
    ly_patch(int(m_ksize/2), int(m_ksize/2)) = l(1);

    // Recorded the pixel as having an initialised direction estimate
    initialised_directions(int(e(1)), int(e(0))) = 1;
    init_directions_patch(int(m_ksize/2), int(m_ksize/2)) = 1;
   
    m_f_regression_status = true;

}


Eigen::Vector<double, 5> SAEdetector::compareDirectionRegression(Eigen::Vector4d e){
    // If flags for previous operations are invalid, we will not return a detection
    if ((m_f_regression_status == false) || (m_f_patch_status == false) || (init_directions_patch.sum()-1 < m_min_active_pixels)){
        return no_detection;
    }
    
    int num_init_points = init_directions_patch.sum() - 1;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_init_points, 2);
    Eigen::MatrixXd W_inv = Eigen::MatrixXd::Zero(num_init_points, num_init_points);

    int idx = 0;
    for (int i = 0; i < patch_points.size(); i++){   
        // Get the indexes in the patch
        int x = patch_points[i].first;
        int y = patch_points[i].second;

        // If the points is not initialised or it is the center element (pixel we are looking at), skip it 
        if ((init_directions_patch(y,x) == 0) || (patch_offset(x) == 0 && patch_offset(y) == 0)){
            continue;
        }

        // Add distance values to A
        A(idx,0) = lx_patch(y,x);
        A(idx,1) = ly_patch(y,x);

        // Compute dt term
        double exp_term = std::exp(2 * m_alpha * (m_t_current - t_patch(y, x)));
        W_inv(idx, idx) = 1/exp_term;

        idx++;
    }

    Eigen::Matrix2d res = A.transpose() * W_inv * A;

    // Compute eigenvector of smallest eigenvalue for l_perp, rotate 90deg 
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(res); 
    Eigen::Vector2d l_perp =  eigen_solver.eigenvectors().col(0);
    Eigen::Vector2d l_est = {-l_perp(1), l_perp(0)};

    Eigen::Vector2d l_event = {lx_patch(int(m_ksize/2), int(m_ksize/2)), ly_patch(int(m_ksize/2), int(m_ksize/2))};

    double l_inner = std::abs(l_event.transpose() * l_est);

        // Check if the magnitude of the difference is below our threshold
    if (l_inner >= m_detection_threshold){
        Eigen::Vector<double, 5> detection;
        detection << e(0), e(1), 0, 0, e(2);
        return detection;
    } 
    else {
        return no_detection;
    }
}