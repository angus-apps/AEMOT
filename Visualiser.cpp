//---- Standard Headers ----//
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>

//---- User Headers ----//
#include "Visualiser.hpp"
#include "parameters.hpp"
#include "Kalman.hpp"



std::vector<cv::Point> selected_points;
static void onMouse(int event, int x, int y, int flags, void* userdata);

//------------------------------------------//
//-----        PUBLIC FUNCTIONS        -----//
//------------------------------------------//

Visualiser::Visualiser(Parameters &params){
    //----- Update member values ----//
    m_f_verbose = params.f_verbose_visualiser;
    m_filepath = params.input_folder_path + params.input_data_name;
    m_height = params.height;
    m_width = params.width;
    m_publish_framerate = params.publish_framerate;
    if (m_publish_framerate > 0){
        m_dt_publish = 1.0/m_publish_framerate;
    }
    m_t_last_display = 0;

    //----- Update Save Flags -----//
    m_f_save_blob_tracks      = params.f_save_blob_tracks;
    m_f_save_candidate_tracks = params.f_save_candidate_tracks;
    m_f_save_all_events       = params.f_save_all_events;
    m_f_save_track_events     = params.f_save_track_events;
    m_f_save_residual_events  = params.f_save_residual_events;

    //----- Update Display flags -----//
    m_f_target_select         = (params.f_select_target == 1) ? 1 : 0;
    m_f_show_blob_tracks      = params.f_show_blob_tracks;
    m_f_show_candidate_tracks = params.f_show_candidate_tracks;
    m_f_show_all_events       = params.f_show_all_events;
    m_f_show_track_events     = params.f_show_track_events;
    m_f_show_residual_events  = params.f_show_residual_events;

    // Check if the any of the all event images are required and update if needed
    m_f_hpf_all_events_master = (m_f_show_blob_tracks || m_f_save_blob_tracks) ||
                                (m_f_show_candidate_tracks || m_f_save_candidate_tracks) ||
                                (m_f_show_all_events || m_f_save_all_events) ||
                                (m_f_target_select);

    //----- Update File Paths -----//
    m_tracks_video_path             = m_filepath + "_" + m_tracks_window + ".avi"    ;//"_outputs_video.avi";
    m_output_candidates_path        = m_filepath + "_" + m_candidate_window + ".avi"    ;//"_candidates_video.avi";
    m_output_all_video_path         = m_filepath + "_" + m_all_events_window + ".avi"    ;//"_all_video.avi";
    m_output_track_video_path       = m_filepath + "_" + m_track_event_window + ".avi"    ;//"_tracked_video.avi";
    m_output_residual_video_path    = m_filepath + "_" + m_residual_event_window + ".avi"    ;//"_residual_video.avi";
    
    //----- Setup Display and Saving -----//
    setUpCVDisplay();
    setUpCVSaving();

    //----- Setup HighPassFilter objects for each frame -----//
    setUpHPFimages(params);

    if (m_f_verbose){
        std::cout << "Visualiser: \n\tFrame Rate: " << m_publish_framerate << "\n\tm_dt_publish: " << m_dt_publish << "\n\tMaster Display Flag: " << m_f_hpf_all_events_master << std::endl;
    }
}

void Visualiser::addEvent(Eigen::Vector4d e){
    // Add event to HPF images
    if (m_f_hpf_all_events_master){
        m_hpf_all_events->addEvent(e);
    }
    if (m_f_show_track_events || m_f_save_track_events){
        m_hpf_track_events->addEvent(e);
    }
    if (m_f_show_residual_events || m_f_save_residual_events){
        m_hpf_residual_events->addEvent(e);
    }

    m_t_current = e(2);

    // Check if it is time to display, and set flag
    if ((m_t_current - m_t_last_display) > m_dt_publish){
        f_display = 1;
        m_t_last_display = m_t_current;
    }
}

void Visualiser::addEvent(Eigen::Vector4d e, int data_assocation){
    // Add event to HPF images

    if (m_f_hpf_all_events_master){
        m_hpf_all_events->addEvent(e);
    }
    if ((m_f_show_track_events || m_f_save_track_events) && (data_assocation == 1)){
        m_hpf_track_events->addEvent(e);
    }
    if ((m_f_show_residual_events || m_f_save_residual_events) && ((data_assocation == 2)||(data_assocation == 3))){
        m_hpf_residual_events->addEvent(e);
    }

    m_t_current = e(2);

    // Check if it is time to display, and set flag
    if ((m_t_current - m_t_last_display) > m_dt_publish){
        f_display = 1;
        m_t_last_display = m_t_current;
    }
}

int Visualiser::display(){
    if (m_f_show_all_events){
        display_without_tracks(m_all_events_window, m_hpf_all_events->getFrame());
    } 

    int status = displayWait();
    
    if (status == 2){
        std::cout << "Display for select" << std::endl;
    }
    return status; 
}

int Visualiser::display(const std::vector<KalmanFilter*> output_tracks, const std::vector<KalmanFilter*> candidate_tracks){
    if (m_f_show_blob_tracks){
        display_with_tracks(m_tracks_window, m_hpf_all_events->getFrame(), output_tracks);
    }
    if (m_f_show_candidate_tracks){
        display_with_tracks(m_candidate_window, m_hpf_all_events->getFrame(), candidate_tracks);
    }    
    if (m_f_show_all_events){
        display_without_tracks(m_all_events_window, m_hpf_all_events->getFrame());
    }    
    if (m_f_show_track_events){
        display_without_tracks(m_track_event_window, m_hpf_track_events->getFrame());
    }
    if (m_f_show_residual_events){
        display_without_tracks(m_residual_event_window, m_hpf_residual_events->getFrame());
    }
    
    int status = displayWait();
    if (status == 2){
        std::cout << "Display for select" << std::endl;
    }
    return status;
}

void Visualiser::save(const std::vector<KalmanFilter*> output_tracks, const std::vector<KalmanFilter*> candidate_tracks){
    if (m_f_save_blob_tracks){
        cv::Mat frame = generateImageWithTracks(m_hpf_all_events->getFrame(), output_tracks);
        m_writer_tracks.write(frame);
    }
    if (m_f_save_candidate_tracks){
        cv::Mat frame = generateImageWithTracks(m_hpf_all_events->getFrame(), candidate_tracks);
        m_writer_candidates.write(frame);
    }
    if (m_f_save_all_events){
        cv::Mat frame = generateImage(m_hpf_all_events->getFrame());
        m_writer_all_events.write(frame);
    }
    if (m_f_save_track_events){
        cv::Mat frame = generateImage(m_hpf_track_events->getFrame());
        m_writer_track_events.write(frame);
    }
    if (m_f_save_residual_events){
        cv::Mat frame = generateImage(m_hpf_residual_events->getFrame());
        m_writer_residual_events.write(frame);
    }
}

void Visualiser::displayForSelect(){
    std::string select_instructions = "\nDisplay for Select: \nClick objects to select them, press ENTER to finish. \nPress ESC to reselect objects. \nPress 'q' to quit.";
    std::cout << select_instructions << std::endl;
    selected_points.clear();
    
    // Close all windows
    cv::destroyAllWindows();

    // Display select window
    display_without_tracks(m_target_select_window, m_hpf_all_events->getFrame());
    cv::setMouseCallback(m_target_select_window, onMouse, 0);

    // Wait while the points are selected
    int key = 0;
    int status = 1;

    while (key != 13 && key != 'q'){
        key = cv::waitKey(30) & 0xFF;
        if (key == 'q'){
            status = -1;
        }
    }
    // Print diagnostics and close window
    std::cout << selected_points.size() << " targets selected." << std::endl;
    cv::destroyAllWindows();   
    
    // Save and clear values
    m_selected_points = selected_points;
    selected_points.clear();

    // Update display values to get time for filters to evolve
    m_t_last_display = m_t_current;
}

void Visualiser::initialiseAllEventsHPF(HighPassFilter* hpf_object){
    // Replace the class object with the existing data
    m_hpf_all_events = hpf_object;  
    m_t_current = hpf_object->getTime();
}

std::vector<cv::Point> Visualiser::getSelectedPoints(){
    std::vector<cv::Point> points = m_selected_points;
    m_selected_points.clear();
    return points;
}

void Visualiser::clean_up(){
    // Destroy all windows
    cv::destroyAllWindows();
    if (m_f_save_blob_tracks){
        m_writer_tracks.release();
    }
    if (m_f_save_candidate_tracks){
        m_writer_candidates.release();
    }
    if (m_f_save_all_events){
        m_writer_all_events.release();
    }
    if (m_f_save_track_events){
        m_writer_track_events.release();
    }
    if (m_f_save_residual_events){
        m_writer_residual_events.release();
    }
}


//------------------------------------------//
//-----       PRIVATE FUNCTIONS        -----//
//------------------------------------------//
Visualiser::~Visualiser(){
    // Delet the HPF objects when the Visualiser becomes out of scope
    delete m_hpf_all_events,
           m_hpf_track_events,
           m_hpf_residual_events;
}

//----- Setup Functions -----//
void Visualiser::setUpCVDisplay(){
    if (m_f_show_blob_tracks){
        cv::namedWindow(m_tracks_window, cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty(m_tracks_window, cv::WND_PROP_TOPMOST, 1);
    }
    if (m_f_show_candidate_tracks){
        cv::namedWindow(m_candidate_window, cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty(m_candidate_window, cv::WND_PROP_TOPMOST, 1);
    }    
    if (m_f_show_all_events){
        cv::namedWindow(m_all_events_window, cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty(m_all_events_window, cv::WND_PROP_TOPMOST, 1);
    }    
    if (m_f_show_track_events){
        cv::namedWindow(m_track_event_window, cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty(m_track_event_window, cv::WND_PROP_TOPMOST, 1);
    }
    if (m_f_show_residual_events){
        cv::namedWindow(m_residual_event_window, cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty(m_residual_event_window, cv::WND_PROP_TOPMOST, 1);
    }
    if (m_f_target_select){
        cv::namedWindow(m_target_select_window, cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty(m_target_select_window, cv::WND_PROP_TOPMOST, 1);
    }
}

void Visualiser::setUpCVSaving(){

    if (m_f_save_blob_tracks){
        m_writer_tracks.open(m_tracks_video_path,
                             cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 40.0,
                             cv::Size(m_width, m_height));
    }
    if (m_f_save_candidate_tracks){
        m_writer_candidates.open(m_output_candidates_path,
                                 cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 40.0,
                                 cv::Size(m_width, m_height));
    }
    if (m_f_save_all_events){
        m_writer_all_events.open(m_output_all_video_path,
                                 cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 40.0,
                                 cv::Size(m_width, m_height));
    }
    if (m_f_save_track_events){
        m_writer_track_events.open(m_output_track_video_path,
                                   cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 40.0,
                                   cv::Size(m_width, m_height));
    }
    if (m_f_save_residual_events){
        m_writer_residual_events.open(m_output_residual_video_path,
                                      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 40.0,
                                      cv::Size(m_width, m_height));
    }


}

void Visualiser::setUpHPFimages(Parameters &params){
    // Create instances of the HPF for frames that we want to display or save
    if (m_f_hpf_all_events_master){
        m_hpf_all_events = new HighPassFilter(params);
    }
    if (m_f_show_track_events || m_f_save_track_events){
        m_hpf_track_events = new HighPassFilter(params);
    }
    if (m_f_show_residual_events || m_f_save_residual_events){
        m_hpf_residual_events = new HighPassFilter(params);
    }
}

//----- Display Functions -----//
cv::Mat Visualiser::generateImage(cv::Mat image_array){
    cv::Mat image;
    cv::exp(image_array, image);
    double minVal = 1.4;
    double maxVal = 0.4;
    image = ((image - minVal) / (maxVal - minVal));
    cv::Mat img(image.rows, image.cols, CV_64FC1, (char *)image.data);
    img.convertTo(img, CV_8U, 255.0 / 1.0);
    cv::Mat cimg;
    // cv::cvtColor(img, cimg, cv::COLOR_GRAY2RGB);
    cv::applyColorMap(img, cimg, cv::COLORMAP_TWILIGHT_SHIFTED);

    // Add timestamp
    std::ostringstream strs;
    strs << std::setprecision(3) << std::fixed;
    strs << m_t_current;
    std::string timestamp = strs.str();
    cv::Point text_position(10, 700);
    float font_size = 0.75;
    cv::Scalar font_Color(0,0,0);
    float font_weight = 2;
    cv::putText(cimg, timestamp, text_position,cv::FONT_HERSHEY_SIMPLEX, font_size,font_Color, font_weight);
    return cimg;
}


cv::Mat Visualiser::generateImageWithTracks(cv::Mat image_array, std::vector<KalmanFilter*> tracks){
    // Get formatted image
    cv::Mat cv_image = generateImage(image_array);

    // Add tracks to the image
    for (int i = 0; i < tracks.size(); i++){
        Eigen::MatrixXd x_hat_current = tracks[i]->getState();   
        double rotationAngle = x_hat_current(6) * 57.295;
        rotationAngle = std::fmod(rotationAngle, 360.0);

        if (rotationAngle < 0.0){
            rotationAngle += 360.0;
        }

        if (std::isnan(x_hat_current(4)) || std::isnan(x_hat_current(5))){
            std::cout << "Visualiser: invalid lamda value \t" << x_hat_current(4) << "\t" << x_hat_current(5) << std::endl;
            continue;
        }
        cv::ellipse(cv_image, cv::Point(x_hat_current(0), x_hat_current(1)), cv::Size(x_hat_current(4), x_hat_current(5)), rotationAngle, 0, 360, cv::Scalar(0,0,255), 2, cv::LINE_AA);
    }
    
    return cv_image;
}


void Visualiser::display_without_tracks(cv::String window_name, cv::Mat image_array){
    // Get formatted image
    cv::Mat cv_image = generateImage(image_array);
    cv::imshow(window_name, cv_image);
}


void Visualiser::display_with_tracks(cv::String window_name, cv::Mat image_array, const std::vector<KalmanFilter*> tracks){
    // Get formatted image
    cv::Mat cv_image = generateImageWithTracks(image_array, tracks);
    cv::imshow(window_name, cv_image);
}


int Visualiser::displayWait(){
    // Define window display time
    // Find the actual run time since the last display and find the remaining
    auto m_chrono_time_now = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(m_chrono_time_now - m_chrono_time_last).count();
    int display_time = (m_dt_publish*1e3) - runtime;
    display_time = (display_time > 1) ? display_time : 1;

    // Set the wait key
    int key = cv::waitKey(display_time) & 0xFF;
    if (key == 'q'){ // 'q' key
        return -1;
    }
    if ((key == 27) && m_f_target_select){ // "ESC" key
        displayForSelect();
        return 2;
    }

    m_chrono_time_last = std::chrono::high_resolution_clock::now();
    f_display = 0; 
    return 1;
}

// Static member function as the callback
static void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        selected_points.push_back(cv::Point(x, y));
        std::cout << "\tPoint selected: (" << x << ", " << y << ")" << std::endl;
    }
}

