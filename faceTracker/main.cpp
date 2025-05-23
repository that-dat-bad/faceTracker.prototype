#include <iostream>
#include <vector>
#include <string>
#include <numeric> // std::iota用
#include <algorithm> // std::find用
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/face.hpp>

// --- グローバル変数 ---
enum TrackingLevel { LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH };
TrackingLevel g_current_tracking_level = LEVEL_MEDIUM;

std::vector<int> g_landmark_indices_low = { 0, 4, 8, 12, 16, 17, 21, 22, 26, 30, 36, 39, 42, 45, 48, 54, 57 };
std::vector<int> g_landmark_indices_medium = { 0,2,4,6,8,10,12,14,16,17,19,21,22,24,26,27,30,31,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,51,54,57 };
std::vector<int> g_target_landmark_indices_to_use;

std::vector<int> g_correction_landmark_indices = { 36, 45, 30, 48, 54 }; // 補正対象の主要5点
std::vector<std::string> g_correction_landmark_names = { "Left Eye Outer", "Right Eye Outer", "Nose Tip", "Mouth Left Corner", "Mouth Right Corner" };

bool g_prompt_for_correction_check = false; // 自動検出後、補正のY/Nをユーザーに尋ねる状態か
bool g_correction_phase_active = false;
int g_collected_correction_clicks = 0;
int g_current_correction_guide_idx = 0;

std::vector<cv::Point2f> g_all_68_auto_landmarks;
std::vector<cv::Point2f> g_initial_landmark_positions;
std::vector<cv::Point2f> g_points_for_optical_flow;
bool g_landmarks_initialized = false;
bool g_face_detected_in_current_frame = false; // 現在のフレームで顔が検出されたか
int g_num_landmarks_to_track = 0;

cv::CascadeClassifier g_face_cascade;
cv::Ptr<cv::face::FacemarkLBF> g_facemark;

std::vector<int> g_detailed_landmark_status;
std::vector<int> g_consecutive_failure_counts;
cv::Mat g_current_frame_for_callback_ref;

// --- 関数プロトタイプ宣言 ---
void update_tracking_parameters_and_reset();
void reset_initialization_flags();
void finalize_landmark_setup(const std::vector<cv::Point2f>& source_landmarks);

// --- 関数定義 ---
void update_tracking_parameters_and_reset() {
    g_target_landmark_indices_to_use.clear();
    if (g_current_tracking_level == LEVEL_LOW) {
        g_target_landmark_indices_to_use = g_landmark_indices_low;
        std::cout << "追跡レベル: 低 (" << g_landmark_indices_low.size() << "点)" << std::endl;
    }
    else if (g_current_tracking_level == LEVEL_MEDIUM) {
        g_target_landmark_indices_to_use = g_landmark_indices_medium;
        std::cout << "追跡レベル: 中 (" << g_landmark_indices_medium.size() << "点)" << std::endl;
    }
    else { // LEVEL_HIGH
        g_target_landmark_indices_to_use.resize(68);
        std::iota(g_target_landmark_indices_to_use.begin(), g_target_landmark_indices_to_use.end(), 0);
        std::cout << "追跡レベル: 高 (68点)" << std::endl;
    }
    g_num_landmarks_to_track = g_target_landmark_indices_to_use.size();
    reset_initialization_flags();
}

void reset_initialization_flags() {
    g_landmarks_initialized = false;
    g_initial_landmark_positions.clear();
    g_points_for_optical_flow.clear();
    g_detailed_landmark_status.clear();
    g_consecutive_failure_counts.clear();
    g_face_detected_in_current_frame = false;
    g_all_68_auto_landmarks.clear();
    g_prompt_for_correction_check = false;
    g_correction_phase_active = false;
    g_collected_correction_clicks = 0;
    g_current_correction_guide_idx = 0;
    std::cout << "初期化状態をリセット。顔検出後、指示に従ってください。" << std::endl;
}

void onMouseCorrectLandmarks(int event, int x, int y, int flags, void* userdata) {
    if (!g_correction_phase_active || g_landmarks_initialized) return;

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (g_collected_correction_clicks < g_correction_landmark_indices.size()) {
            cv::Point2f clicked_point(static_cast<float>(x), static_cast<float>(y));
            int current_correction_target_idx_68 = g_correction_landmark_indices[g_current_correction_guide_idx];

            // 自動検出された全68点のランドマークのうち、補正対象の点を更新
            if (static_cast<size_t>(current_correction_target_idx_68) < g_all_68_auto_landmarks.size()) {
                g_all_68_auto_landmarks[current_correction_target_idx_68] = clicked_point;
            }

            std::cout << g_correction_landmark_names[g_current_correction_guide_idx] << " を補正: (" << x << ", " << y << ")" << std::endl;
            if (!g_current_frame_for_callback_ref.empty()) {
                cv::circle(g_current_frame_for_callback_ref, clicked_point, 5, cv::Scalar(0, 255, 0), -1); // 補正した点を緑で
            }

            g_collected_correction_clicks++;
            g_current_correction_guide_idx++;

            if (g_collected_correction_clicks == g_correction_landmark_indices.size()) {
                std::cout << "全補正ポイントのクリック完了。" << std::endl;
                g_correction_phase_active = false;
                finalize_landmark_setup(g_all_68_auto_landmarks); // 補正後の全68点から最終セットアップ
            }
        }
    }
}

void finalize_landmark_setup(const std::vector<cv::Point2f>& source_68_landmarks) {
    g_initial_landmark_positions.clear();
    g_points_for_optical_flow.clear();

    for (int target_idx_from_68_model : g_target_landmark_indices_to_use) {
        if (target_idx_from_68_model >= 0 && static_cast<size_t>(target_idx_from_68_model) < source_68_landmarks.size()) {
            g_initial_landmark_positions.push_back(source_68_landmarks[target_idx_from_68_model]);
            g_points_for_optical_flow.push_back(source_68_landmarks[target_idx_from_68_model]);
        }
    }

    if (g_points_for_optical_flow.size() == g_target_landmark_indices_to_use.size()) {
        g_num_landmarks_to_track = g_points_for_optical_flow.size();
        g_detailed_landmark_status.assign(g_num_landmarks_to_track, 1);
        g_consecutive_failure_counts.assign(g_num_landmarks_to_track, 0);
        g_landmarks_initialized = true;
        g_prompt_for_correction_check = false; // Y/Nプロンプトは不要に
        g_correction_phase_active = false;    // 補正フェーズも終了
        std::cout << g_num_landmarks_to_track << "点のランドマークでトラッキングを開始します。" << std::endl;
    }
    else {
        std::cout << "警告: finalize_landmark_setupで要求数のランドマークを取得できませんでした。" << std::endl;
        reset_initialization_flags(); // 問題があれば最初からやり直し
    }
}


int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "エラー: Webカメラを開けませんでした。" << std::endl; return -1; }

    if (!g_face_cascade.load("haarcascade_frontalface_alt.xml")) { std::cerr << "エラー: haarcascade_frontalface_alt.xml をロードできませんでした。" << std::endl; return -1; }
    g_facemark = cv::face::FacemarkLBF::create();
    try {
        g_facemark->loadModel("lbfmodel.yaml");
    }
    catch (const cv::Exception& e) { std::cerr << "エラー: lbfmodel.yaml のロード中に例外: " << e.what() << std::endl; return -1; }

    update_tracking_parameters_and_reset();

    std::string window_name = "フェイシャルトラッキング (1:低, 2:中, 3:高, R:リセット)";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::setMouseCallback(window_name, onMouseCorrectLandmarks, nullptr);

    cv::Mat frame, prev_gray_frame;
    std::vector<uchar> uchar_landmark_tracking_status;
    const float MAX_OPTICAL_FLOW_ERROR = 20.0f;
    const int MAX_CONSECUTIVE_FAILURES = 5;

    std::cout << "カメラに顔を向け、指示に従ってください。" << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) { std::cerr << "エラー: フレームが空です。" << std::endl; break; }
        frame.copyTo(g_current_frame_for_callback_ref);
        cv::Mat frame_display = frame.clone();
        g_face_detected_in_current_frame = false; // フレームごとにリセット

        if (!g_landmarks_initialized) {
            cv::Mat gray_frame_for_detection;
            cv::cvtColor(frame, gray_frame_for_detection, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray_frame_for_detection, gray_frame_for_detection);
            std::vector<cv::Rect> faces;
            g_face_cascade.detectMultiScale(gray_frame_for_detection, faces, 1.15, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(80, 80));

            if (!faces.empty()) {
                cv::Rect face_roi = faces[0];
                cv::rectangle(frame_display, face_roi, cv::Scalar(255, 0, 0), 2);
                g_face_detected_in_current_frame = true;

                std::vector<std::vector<cv::Point2f>> landmarks_all_faces_detected_in_fit;
                bool fit_success = false;
                // g_all_68_auto_landmarks が空（＝まだY/Nプロンプト前、またはリセット直後）の場合のみfitを実行
                if (g_all_68_auto_landmarks.empty()) {
                    try {
                        fit_success = g_facemark->fit(gray_frame_for_detection, faces, landmarks_all_faces_detected_in_fit);
                        if (fit_success && !landmarks_all_faces_detected_in_fit.empty()) {
                            g_all_68_auto_landmarks = landmarks_all_faces_detected_in_fit[0];
                            g_prompt_for_correction_check = true; // fit成功後、Y/Nプロンプト状態へ
                        }
                        else {
                            g_all_68_auto_landmarks.clear(); // fit失敗ならクリア
                        }
                    }
                    catch (const cv::Exception& e) { std::cerr << "fit中に例外: " << e.what() << std::endl; g_all_68_auto_landmarks.clear(); }
                }


                if (!g_all_68_auto_landmarks.empty()) { // 自動検出されたランドマークがある場合
                    // まず現在の追跡レベルのランドマークを一時的に表示（補正前の状態）
                    for (int target_idx : g_target_landmark_indices_to_use) {
                        if (target_idx >= 0 && static_cast<size_t>(target_idx) < g_all_68_auto_landmarks.size()) {
                            cv::circle(frame_display, g_all_68_auto_landmarks[target_idx], 3, cv::Scalar(255, 100, 0), -1); // 自動検出点を青っぽく表示
                        }
                    }

                    if (g_prompt_for_correction_check) {
                        cv::putText(frame_display, "Correct 주요 LMs? (Y/N)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                    }
                    else if (g_correction_phase_active) {
                        if (g_current_correction_guide_idx < g_correction_landmark_indices.size()) {
                            int target_68_idx = g_correction_landmark_indices[g_current_correction_guide_idx];
                            if (static_cast<size_t>(target_68_idx) < g_all_68_auto_landmarks.size()) {
                                cv::Point2f guide_pt = g_all_68_auto_landmarks[target_68_idx]; // ガイドは常に最新のg_all_68_auto_landmarksから
                                cv::circle(frame_display, guide_pt, 7, cv::Scalar(0, 255, 255), 2); // 黄色でガイド
                                std::string text = "Click: " + g_correction_landmark_names[g_current_correction_guide_idx] +
                                    " (" + std::to_string(g_collected_correction_clicks + 1) + "/" +
                                    std::to_string(g_correction_landmark_indices.size()) + ")";
                                cv::putText(frame_display, text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
                            }
                        }
                        // 既に補正された点を強調表示
                        for (int k = 0; k < g_collected_correction_clicks; ++k) {
                            int corrected_idx_68 = g_correction_landmark_indices[k];
                            if (static_cast<size_t>(corrected_idx_68) < g_all_68_auto_landmarks.size()) {
                                cv::circle(frame_display, g_all_68_auto_landmarks[corrected_idx_68], 5, cv::Scalar(0, 128, 255), -1); // 補正試行中の点をオレンジで
                            }
                        }
                    }
                }
                else if (g_face_detected_in_current_frame) { // 顔は検出されたがfitに失敗した場合
                    cv::putText(frame_display, "Landmark fit failed.", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                }
            }
            else { // 顔が検出されなかった場合
                cv::putText(frame_display, "Face not detected.", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                if (g_correction_phase_active || g_prompt_for_correction_check) { // 補正中やY/N待ち中に顔を見失ったらリセット
                    std::cout << "顔を見失いました。初期化プロセスをリセットします。" << std::endl;
                    reset_initialization_flags();
                }
            }
        }
        else { // g_landmarks_initialized == true
            // --- トラッキング処理フェーズ ---
            // (この部分は前回のコードとほぼ同じ。g_initial_landmark_positions を基準にする)
            cv::Mat current_gray_frame;
            cv::cvtColor(frame_display, current_gray_frame, cv::COLOR_BGR2GRAY);

            if (prev_gray_frame.empty()) { current_gray_frame.copyTo(prev_gray_frame); }

            if (g_points_for_optical_flow.empty() || g_points_for_optical_flow.size() != static_cast<size_t>(g_num_landmarks_to_track)) {
                if (g_num_landmarks_to_track > 0) reset_initialization_flags();
            }
            else {
                std::vector<cv::Point2f> next_points_candidate; std::vector<float> error_metric;
                cv::calcOpticalFlowPyrLK(prev_gray_frame, current_gray_frame, g_points_for_optical_flow, next_points_candidate, uchar_landmark_tracking_status, error_metric, cv::Size(31, 31), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
                std::vector<cv::Point2f> final_points_for_this_frame = g_points_for_optical_flow;
                std::vector<cv::Point2f> successfully_tracked_initial_coords, successfully_tracked_current_coords;
                if (g_detailed_landmark_status.size() != g_num_landmarks_to_track) {
                    g_detailed_landmark_status.assign(g_num_landmarks_to_track, 1);
                    g_consecutive_failure_counts.assign(g_num_landmarks_to_track, 0);
                }
                for (size_t i = 0; i < uchar_landmark_tracking_status.size(); ++i) {
                    if (i >= g_detailed_landmark_status.size() || i >= g_initial_landmark_positions.size() || i >= next_points_candidate.size()) continue;
                    bool tracked = (uchar_landmark_tracking_status[i] == 1 && error_metric[i] < MAX_OPTICAL_FLOW_ERROR);
                    if (tracked) {
                        final_points_for_this_frame[i] = next_points_candidate[i];
                        g_detailed_landmark_status[i] = 1; g_consecutive_failure_counts[i] = 0;
                        successfully_tracked_initial_coords.push_back(g_initial_landmark_positions[i]);
                        successfully_tracked_current_coords.push_back(next_points_candidate[i]);
                    }
                    else {
                        g_consecutive_failure_counts[i]++;
                        if (g_consecutive_failure_counts[i] > MAX_CONSECUTIVE_FAILURES) g_detailed_landmark_status[i] = 0;
                    }
                }
                if (successfully_tracked_current_coords.size() >= std::max(2, (int)(g_num_landmarks_to_track / 2.0)) &&
                    successfully_tracked_current_coords.size() == successfully_tracked_initial_coords.size() && !successfully_tracked_initial_coords.empty()) {
                    cv::Mat transform_matrix = cv::estimateAffinePartial2D(successfully_tracked_initial_coords, successfully_tracked_current_coords, cv::noArray(), cv::RANSAC, 3.0);
                    if (!transform_matrix.empty()) {
                        for (int i = 0; i < g_num_landmarks_to_track; ++i) {
                            if (i >= g_detailed_landmark_status.size() || i >= g_initial_landmark_positions.size()) continue;
                            if (g_detailed_landmark_status[i] != 1) {
                                std::vector<cv::Point2f> pt_vec(1, g_initial_landmark_positions[i]), transformed_pt_vec;
                                cv::transform(pt_vec, transformed_pt_vec, transform_matrix);
                                if (!transformed_pt_vec.empty()) {
                                    final_points_for_this_frame[i] = transformed_pt_vec[0];
                                    g_detailed_landmark_status[i] = 2; g_consecutive_failure_counts[i] = 0;
                                }
                                else {
                                    if (g_detailed_landmark_status[i] != 0 && g_consecutive_failure_counts[i] > MAX_CONSECUTIVE_FAILURES) g_detailed_landmark_status[i] = 0;
                                }
                            }
                        }
                    }
                }
                else {
                    for (int i = 0; i < g_num_landmarks_to_track; ++i) {
                        if (i >= g_detailed_landmark_status.size()) continue;
                        if (g_detailed_landmark_status[i] != 1 && g_detailed_landmark_status[i] != 0) {
                            if (g_consecutive_failure_counts[i] > MAX_CONSECUTIVE_FAILURES) g_detailed_landmark_status[i] = 0;
                        }
                    }
                }
                g_points_for_optical_flow = final_points_for_this_frame;
            }
            for (size_t i = 0; i < g_points_for_optical_flow.size(); ++i) {
                if (i >= g_detailed_landmark_status.size()) continue;
                cv::Scalar color; std::string status_text_brief;
                switch (g_detailed_landmark_status[i]) {
                case 1: color = cv::Scalar(0, 255, 0); status_text_brief = "T"; break;
                case 2: color = cv::Scalar(0, 165, 255); status_text_brief = "E"; break;
                default:color = cv::Scalar(0, 0, 255); status_text_brief = "F"; break;
                }
                if (g_points_for_optical_flow[i].x >= 0 && g_points_for_optical_flow[i].y >= 0 && g_points_for_optical_flow[i].x < frame_display.cols && g_points_for_optical_flow[i].y < frame_display.rows) {
                    cv::circle(frame_display, g_points_for_optical_flow[i], 3, color, -1);
                }
            }
            if (!current_gray_frame.empty()) current_gray_frame.copyTo(prev_gray_frame);
        }

        cv::imshow(window_name, frame_display);

        int key = cv::waitKey(1);
        if (key == 27) { break; }
        else if (key == '1') { g_current_tracking_level = LEVEL_LOW; update_tracking_parameters_and_reset(); }
        else if (key == '2') { g_current_tracking_level = LEVEL_MEDIUM; update_tracking_parameters_and_reset(); }
        else if (key == '3') { g_current_tracking_level = LEVEL_HIGH; update_tracking_parameters_and_reset(); }
        else if (key == 'r' || key == 'R') { reset_initialization_flags(); }
        else if ((key == 'y' || key == 'Y') && g_prompt_for_correction_check && g_face_detected_in_current_frame && !g_all_68_auto_landmarks.empty()) {
            g_prompt_for_correction_check = false; // Y/Nプロンプト終了
            finalize_landmark_setup(g_all_68_auto_landmarks); // 自動検出のままトラッキング開始
            std::cout << "補正なしでトラッキングを開始します。" << std::endl;
        }
        else if ((key == 'n' || key == 'N') && g_prompt_for_correction_check && g_face_detected_in_current_frame && !g_all_68_auto_landmarks.empty()) {
            g_prompt_for_correction_check = false; // Y/Nプロンプト終了
            g_correction_phase_active = true;      // 補正フェーズ開始
            g_collected_correction_clicks = 0;
            g_current_correction_guide_idx = 0;
            std::cout << "補正フェーズ開始。指示に従って主要ランドマークをクリックしてください。" << std::endl;
        }

        if (cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) < 1) { break; }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}