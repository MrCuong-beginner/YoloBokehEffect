#include "ObjectDetector.h"
#include "ImageProcess.h"
#include <iostream>

using namespace std;
int main()
{
    // モデルと動画ファイルのパスを設定
    string modelPath = "Data\\yolov5s.onnx";        // YOLOv5モデルのパス
    string classFile = "Data\\coco.names";          // クラス名を記載したファイルのパス
    string videoPath = "Data\\PeopleVideo.mp4";     // 処理する動画ファイルのパス
    string cascadePath = "Data\\haarcascade_frontalface_default.xml";     // Haar Cascadeのパス

    // ObjectDetector と ImageProcess のインスタンスを作成
    unique_ptr<ObjectDetector> pDetector = make_unique<ObjectDetector>(modelPath, classFile, cascadePath);
    unique_ptr<ImageProcess> pImageProcess = make_unique<ImageProcess>();

    // 動画を読み込む
    cv::VideoCapture cap(videoPath);     // 動画ファイルを開く
    if (!cap.isOpened()) {        // 動画が開けなかった場合
        cout << "Cannot open video file!" << endl;
        return -1;
    }
    cv::UMat uFrame;        // 動画の各フレームを格納する変数
    cv::Mat peopleMask;
    cv::UMat outputFrame;
    // 動画のフレームを1つずつ処理
    while (cap.read(uFrame))
    {
        if (uFrame.empty()) {   // フレームが空の場合（動画が終了した場合）
            cerr << "Could not grab the first frame!" << endl;
            return -1;
        }

        // 人物検出
		peopleMask = pDetector->detectorPeople(uFrame);

        // 人物の顔の部分をぼかす
        outputFrame = pImageProcess->blurProcess(peopleMask, uFrame);

        // 結果を表示
        cv::imshow("Output", outputFrame);

        // 'q' キーが押されたら終了
		if (cv::waitKey(1) == 'q') {
			break;  // 'q' キーでループを終了
		}
    }

	return 0;   // プログラム終了
}
