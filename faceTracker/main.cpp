#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

const char* windowName = "WebCamTracker";

int main() {
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "カメラが開けません\n";
		return -1;
	}

	// 正面向きの Haar分類器を読み込む
	cv::CascadeClassifier frontal_face_cascade;
	if (!frontal_face_cascade.load("./haarcascade_frontalface_default.xml")) {
		std::cerr << "正面顔検出器が読み込めません\n";
		return -1;
	}

	// 横向きの Haar分類器を読み込む
	cv::CascadeClassifier profile_face_cascade;
	if (!profile_face_cascade.load("./haarcascade_profileface.xml")) {
		std::cerr << "横顔検出器が読み込めません\n";
		return -1;
	}

	// 目の Haar分類器 (眼鏡対応版を試す)
	cv::CascadeClassifier eye_cascade;
	if (!eye_cascade.load("./haarcascade_eye_tree_eyeglasses.xml")) {
		std::cerr << "眼鏡対応の目検出器が読み込めません。通常の目検出器を試します。\n";
		eye_cascade.load("./haarcascade_eye.xml");
	}

	cv::Mat frame;
	while (true) {
		cap >> frame;
		if (frame.empty()) { break; }

		// 画像サイズの縮小 (処理速度向上)
		cv::Mat small_frame;
		cv::resize(frame, small_frame, cv::Size(), 0.5, 0.5);
		cv::Mat gray_frame;
		cv::cvtColor(small_frame, gray_frame, cv::COLOR_BGR2GRAY);

		// 正面顔検出
		std::vector<cv::Rect> frontal_faces;
		frontal_face_cascade.detectMultiScale(gray_frame, frontal_faces, 1.1, 3, 0, cv::Size(30, 30));

		// 横顔検出 (右向き)
		std::vector<cv::Rect> profile_faces;
		profile_face_cascade.detectMultiScale(gray_frame, profile_faces, 1.1, 3, 0, cv::Size(30, 30));

		// 横顔検出 (左向き - 画像を反転させて検出)
		cv::Mat flipped_gray_frame;
		cv::flip(gray_frame, flipped_gray_frame, 1); // 水平反転
		std::vector<cv::Rect> flipped_profile_faces;
		profile_face_cascade.detectMultiScale(flipped_gray_frame, flipped_profile_faces, 1.1, 3, 0, cv::Size(30, 30));
		// 反転して検出した顔の位置を元に戻す
		std::vector<cv::Rect> left_profile_faces;
		for (const auto& face : flipped_profile_faces) {
			cv::Rect original_pos;
			original_pos.x = static_cast<int>((small_frame.cols - face.x - face.width) * 2);
			original_pos.y = static_cast<int>(face.y * 2);
			original_pos.width = static_cast<int>(face.width * 2);
			original_pos.height = static_cast<int>(face.height * 2);
			left_profile_faces.push_back(original_pos);
		}

		// 検出された正面顔に青枠を描画し、目の検出を行う
		for (const auto& face : frontal_faces) {
			cv::Rect original_face;
			original_face.x = static_cast<int>(face.x * 2);
			original_face.y = static_cast<int>(face.y * 2);
			original_face.width = static_cast<int>(face.width * 2);
			original_face.height = static_cast<int>(face.height * 2);
			cv::rectangle(frame, original_face, cv::Scalar(255, 0, 0), 2); // 青枠

			// 顔領域内で目を検出
			cv::Mat face_roi = gray_frame(face);
			std::vector<cv::Rect> eyes;
			eye_cascade.detectMultiScale(face_roi, eyes, 1.05, 6, 0, cv::Size(20, 20));

			// 検出された目を元のフレームに緑色の枠で描画
			for (const auto& eye : eyes) {
				cv::Point eye_center(static_cast<int>((face.x + eye.x) * 2 + eye.width),
					static_cast<int>((face.y + eye.y) * 2 + eye.height / 2));
				int radius = static_cast<int>(eye.width * 0.25);
				cv::circle(frame, eye_center, radius, cv::Scalar(0, 255, 0), 2); // 緑色の円
			}
		}

		// 検出された右向き横顔に緑枠を描画
		for (const auto& face : profile_faces) {
			cv::Rect original_face;
			original_face.x = static_cast<int>(face.x * 2);
			original_face.y = static_cast<int>(face.y * 2);
			original_face.width = static_cast<int>(face.width * 2);
			original_face.height = static_cast<int>(face.height * 2);
			cv::rectangle(frame, original_face, cv::Scalar(0, 255, 0), 2); // 緑枠

			// 横顔領域内で目を検出 (精度向上のため、必要に応じて調整)
			cv::Mat face_roi = gray_frame(face);
			std::vector<cv::Rect> eyes;
			eye_cascade.detectMultiScale(face_roi, eyes, 1.1, 3, 0, cv::Size(15, 15));

			// 検出された目を元のフレームに緑色の枠で描画
			for (const auto& eye : eyes) {
				cv::Point eye_center(static_cast<int>((face.x + eye.x) * 2 + eye.width),
					static_cast<int>((face.y + eye.y) * 2 + eye.height / 2));
				int radius = static_cast<int>(eye.width * 0.25);
				cv::circle(frame, eye_center, radius, cv::Scalar(0, 255, 0), 2); // 緑色の円
			}
		}

		// 検出された左向き横顔に緑枠を描画し、目の検出を行う
		for (const auto& face : left_profile_faces) {
			cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2); // 緑枠

			// 左向き横顔領域内で目を検出
			cv::Rect small_face;
			small_face.x = static_cast<int>(face.x * 0.5);
			small_face.y = static_cast<int>(face.y * 0.5);
			small_face.width = static_cast<int>(face.width * 0.5);
			small_face.height = static_cast<int>(face.height * 0.5);
			cv::Mat face_roi = flipped_gray_frame(small_face);
			std::vector<cv::Rect> eyes;
			eye_cascade.detectMultiScale(face_roi, eyes, 1.1, 3, 0, cv::Size(15, 15));

			// 検出された目を元のフレームに緑色の枠で描画 (位置を補正)
			for (const auto& eye : eyes) {
				cv::Point original_eye_center;
				original_eye_center.x = static_cast<int>((small_frame.cols - (small_face.x + eye.x + eye.width)) * 2);
				original_eye_center.y = static_cast<int>((small_face.y + eye.y) * 2 + eye.height / 2);
				int radius = static_cast<int>(eye.width * 0.25 * 2);
				cv::circle(frame, original_eye_center, radius, cv::Scalar(0, 255, 0), 2); // 緑色の円
			}
		}

		cv::imshow(windowName, frame);
		if (cv::waitKey(1) == 27 ||
			cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();
	return 0;
}