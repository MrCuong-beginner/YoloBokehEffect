#ifndef _OBJECT_DETECTOR_H_
#define _OBJECT_DETECTOR_H_

#include "OpenCV.h"

// 定数の定義（YOLOモデルの設定値）
#define SCALE (1.0 / 255.0)             // 画素値を正規化するスケール
#define INPUT_WIDTH 640.0               // YOLOモデルの入力画像幅
#define INPUT_HEIGHT 640.0              // YOLOモデルの入力画像高さ
#define CONFIDENCE_THRESHOLD 0.5        // 物体を検出するための信頼度の閾値
#define SCORE_THRESHOLD 0.4             // クラス識別のスコア閾値
#define NMS_THRESHOLD 0.2               // 非最大抑制（NMS）の閾値
#define AREAS_THRESHOLD 3600            // 検出されたオブジェクトの最小面積閾値

using namespace std;
/**
* @class ObjectDetector
* @brief YOLOv5モデルを使用して物体検出を行うクラス。
* このクラスは、指定された画像に対してYOLOv5モデルを用いて物体検出を行い、
* 検出された物体に対してバウンディングボックスを描画します。また、非最大抑制
* (NMS)を使用して検出結果をフィルタリングします。
*/
class ObjectDetector
{
public:
    /**
    * @brief コンストラクタ（デフォルトコンストラクタ）。
    * デフォルトの設定でObjectDetectorオブジェクトを初期化します。
    */
    ObjectDetector();

    /**
    * @brief コンストラクタ（デフォルトコンストラクタ）。
    * デフォルトの設定でObjectDetectorオブジェクトを初期化します。
    */
    ~ObjectDetector();

	/**
	* @brief コンストラクタ。
	* 指定されたモデルファイルとクラス名ファイルからObjectDetectorオブジェクトを初期化します。
	* @param modelPath モデルファイルのパス（ONNX形式）
	* @param classFile クラス名が記載されたファイルのパス
	*/
    explicit ObjectDetector(const string& modelPath, const string& classFile, const string& cascadePath);

    /**
    * @brief 画像から物体を検出し、バウンディングボックスを返す。
    * 入力画像に対して物体検出を行い、検出された物体のバウンディングボックスを返します。
    * @param uImage 入力画像（cv::UMat形式）。
    * @return 検出された物体のリスト（Predicted型のベクトル）。
    */
    vector<Predicted> boundingBoxDetector(const cv::UMat& uImage);

    /**
    * @brief 画像から人を検出する。
    * 入力画像に対して人を検出し、その結果を返します。
    * @param uImage 入力画像（cv::UMat形式）。
    * @return 人を検出した画像（cv::Mat形式）。
    */
    cv::Mat detectorPeople(const cv::UMat& uImage);

    /**
    * @brief バウンディングボックスを画像に描画する。
    * 検出された物体のバウンディングボックスを入力画像に描画します。
    * @param image 入力画像（cv::Mat形式）。
    * @param detected 検出された物体のリスト（Predicted型のベクトル）。
    * @return バウンディングボックスを描画した画像（cv::Mat形式）。
    */
    cv::Mat drawBoundingBoxes(const cv::Mat& image, const vector<Predicted>& detected);

private:
    /**
    * @brief YOLOモデルの出力を取得する。
    * 入力画像に対してYOLOモデルを実行し、出力を取得します。
    * @param uImage 前処理された入力画像（cv::UMat形式）。
    * @return YOLOネットワークの出力（cv::Mat型のベクトル）。   
    */
    vector<cv::Mat> getOutputs(const cv::UMat& uImage);

    /**
    * @brief クラス名リストをファイルから読み込む。
    * 指定されたパスからクラス名リストを読み込みます。
    * @param path クラス名リストファイルのパス。
    * @return クラス名のリスト（文字列型のベクトル）。
    */
    vector<string> loadClasses(const string& path);

    /**
    * @brief 非最大抑制（NMS）を適用して予測結果をフィルタリングする。
    *
    * NMSを使用して、重複するバウンディングボックスを排除します。
    *
    * @param predictions 検出された物体の予測結果のリスト（Predicted型のベクトル）。
    * @return NMS後の予測結果リスト（Predicted型のベクトル）。
    */
    vector<Predicted> applyNMS(const vector<Predicted>& predictions);

    cv::dnn::Net net;           // YOLOv5モデルのニューラルネットワーク。
    vector<string> classList;   // クラス名リスト。
    cv::CascadeClassifier face_cascade;     //Haar Cascade（顔の検出のライブラリ）
};
#endif // !
