#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp> // トラッキング用ヘッダー

using namespace cv;
using namespace std;

// グローバル変数 (プロトタイプなので簡略化)
vector<Point> initial_landmarks;
bool collecting_clicks = false;
Rect current_face_rect;
int click_count = 0;
const int NUM_CLICKS_NEEDED = 4;

// マウスイベントコールバック関数
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (collecting_clicks && event == EVENT_LBUTTONDOWN) {
        if (current_face_rect.contains(Point(x, y))) {
            initial_landmarks.push_back(Point(x, y));
            click_count++;
            cout << "クリック " << click_count << ": (" << x << ", " << y << ")" << endl;
        }
    }
}

int main() {
    // Webカメラの初期化
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Webカメラを開けませんでした。" << endl;
        return -1;
    }

    // 顔検出器の準備
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Haar cascade ファイルをロードできませんでした。" << endl;
        return -1;
    }

    // ウィンドウの作成とマウスイベントの設定
    namedWindow("Face Tracking Prototype", WINDOW_NORMAL);
    setMouseCallback("Face Tracking Prototype", onMouse, nullptr);

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 顔検出
        vector<Rect> faces;
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // 最初の顔を検出したらクリック受付モードに入る
        Mat frame_with_detections = frame.clone();
        for (const auto& face : faces) {
            rectangle(frame_with_detections, face, Scalar(255, 0, 0), 2);
            if (!collecting_clicks && !faces.empty()) {
                collecting_clicks = true;
                current_face_rect = face;
                initial_landmarks.clear();
                click_count = 0;
                cout << "顔が検出されました。目、鼻、口をクリックしてください。" << endl;
            }
        }

        // クリック受付モード中の処理
        if (collecting_clicks) {
            rectangle(frame_with_detections, current_face_rect, Scalar(0, 255, 255), 2); // クリック対象の顔を強調
            putText(frame_with_detections, format("クリック %d/%d", click_count, NUM_CLICKS_NEEDED), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            for (const auto& p : initial_landmarks) {
                circle(frame_with_detections, p, 5, Scalar(0, 0, 255), -1);
            }
            if (click_count >= NUM_CLICKS_NEEDED) {
                collecting_clicks = false;
                cout << "初期ランドマークを取得しました。トラッキングを開始します。" << endl;
                // ここにトラッキング開始の処理を記述します
            }
        }

        // 結果の表示
        imshow("Face Tracking Prototype", frame_with_detections);

        // ESCキーで終了
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}