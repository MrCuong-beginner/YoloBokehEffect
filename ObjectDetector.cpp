/**************************************************************************//**
@クラス名         Detector
@概要             YOLOモデルを使用した物体検出クラス
@詳細             指定されたONNX形式のYOLOモデルを使用して画像中の物体を検出する。
                  検出結果は信頼度閾値とNMS処理を経て返される。
******************************************************************************/
#include <iostream>
#include <fstream>
#include "ObjectDetector.h"

/**************************************************************************//**
@関数名           ObjectDetector::ObjectDetector
@概要             ONNX形式のYOLOモデルを読み込み、DNNネットワークを初期化する
@パラメータ[in]   modelPath: モデルファイルのパス（ONNX形式）
                  classFile: クラス名が記載されたファイルのパス
@パラメータ[out]  なし
@戻り値           なし
@詳細             指定されたONNX形式のモデルを読み込んで、DNNネットワークを初期化します。
                  モデルのバックエンド（処理方法）として、OpenCLが使用可能であればOpenCLを選択し、
                  使用できない場合はCPUを選択します。また、クラス名は指定されたファイルからロードされます。
                  使用するハードウェア（OpenCLまたはCPU）は、OpenCLのサポート状況によって動的に決定されます。
******************************************************************************/
ObjectDetector::ObjectDetector(const string& modelPath, const string& classFile, const string& cascadePath)
{
    // ONNX形式のモデルを読み込み、DNNネットワークを作成
    net = cv::dnn::readNetFromONNX(modelPath);
    
    // OpenCLが利用可能な場合はOpenCLを使用、それ以外はCPUを使用
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);  // バックエンドをOpenCVに設定
    if (cv::ocl::haveOpenCL())  // OpenCLが利用可能ならOpenCLを選択
    {
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);    // ターゲットをOpenCLに設定
    }
    else
    {
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);       // ターゲットをCPUに設定
    }
    if (!face_cascade.load(cv::samples::findFile(cascadePath))) {
        std::cerr << "Error loading Haar Cascade!" << std::endl;
        return;
    }
    // クラス名をファイルからロード
    classList = loadClasses(classFile);
}

/**************************************************************************//**
@関数名           ObjectDetector::~ObjectDetector
@概要             オブジェクト検出器のデストラクタ。YOLOモデルの読み込みとDNNネットワークのリソース解放を行う
@パラメータ[in]   なし
@パラメータ[out]  なし
@戻り値           なし
@詳細             - DNNネットワークのメモリを解放し、リソースを適切にクリーンアップする
******************************************************************************/
ObjectDetector::~ObjectDetector()
{
}

/**************************************************************************//**
@関数名           Detector::loadClasses
@概要             クラス名をファイルからロードする
@パラメータ[in]   path: クラス名が記載されたファイルのパス
@パラメータ[out]  なし
@戻り値           クラス名のリスト（vector<string>）
@詳細             ファイルから1行ずつ読み込んで、クラス名を格納する
******************************************************************************/
vector<string> ObjectDetector::loadClasses(const string& path)
{
    vector<string> classList;
    ifstream file(path);
    string line;
    while (getline(file, line))
    {
        classList.push_back(line);
    }
    return classList;
}

/**************************************************************************//**
@関数名           ObjectDetector::detectorPeople
@概要             入力画像から人を検出し、マスク画像を生成する
@パラメータ[in]   uImage: 入力画像（UMat型）
@パラメータ[out]  なし
@戻り値           人が存在する領域を示すマスク画像（cv::Mat 型, CV_8UC1）
@詳細             YOLOモデルを使用して物体を検出し、信頼度が閾値を超える人（クラスID 0）のみを選択する。
                  その後、非最大抑制（NMS）を適用し、最終的な結果をマスク画像として返す。
******************************************************************************/
cv::Mat ObjectDetector::detectorPeople(const cv::UMat& uImage)
{
    // 結果を格納するベクターの準備
    vector<int> indices;            // 非最大抑制（NMS）のためのインデックス
    vector<float> confidences;      // 各物体の信頼度を格納する
    vector<cv::Rect> boxes;         // 検出されたバウンディングボックスを格納する  
    // YOLOモデルからの出力結果を取得
    vector<cv::Mat> yoloOutputs = getOutputs(uImage);
    // 画像のサイズと入力サイズに基づきスケーリングファクターを計算
    float xFactor = static_cast<float>(uImage.cols) / INPUT_WIDTH;
    float yFactor = static_cast<float>(uImage.rows) / INPUT_HEIGHT;
    // データのサイズを取得
    const int rows = yoloOutputs[0].size[1];
    const int cols = yoloOutputs[0].size[2];
    // 出力データへのポインタを取得
    float* data = (float*)yoloOutputs[0].data;
    // メモリの使用を最適化するため、予め容量を予約しておく
    confidences.reserve(rows); 
	boxes.reserve(rows);
    indices.reserve(rows);
    for (int i = 0; i < rows; i++, data += cols)
    {
        float confidence = data[4];                                     // 信頼度を取得
        int classId = max_element(data + 5, data + cols) - (data + 5);  // クラスIDを取得（最大の予測確率を持つクラス）
        // 信頼度が閾値を超えていて、かつクラスIDが0（人）の場合に物体を検出
        if (confidence > CONFIDENCE_THRESHOLD && classId==0)
        {
            // バウンディングボックスの座標とサイズをスケールアップ
            float cx = data[0] * xFactor;
            float cy = data[1] * yFactor;
            float w = data[2] * xFactor;
            float h = data[3] * yFactor;
            
            // 検出された物体を格納
            boxes.emplace_back(cv::Rect(cx - w / 2, cy - h / 2, w, h));
			confidences.push_back(confidence);
        }
    }

    // 非最大抑制 (NMS) を適用
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    // 顔検出用のマスク画像を作成
    cv::Mat peopleMask = cv::Mat::zeros(uImage.size(), CV_8UC1);// 画像サイズと同じで、初期値が0（黒）のマスク画像を作成
    vector<cv::Rect> faces;                                     // 検出された顔のバウンディングボックスを格納するベクター
    // 入力画像をグレースケールに変換
    cv::Mat grayImage;
    cv::cvtColor(uImage, grayImage, cv::COLOR_BGR2GRAY);
    int imageWidth = grayImage.cols;
    int imageHeight = grayImage.rows;
    for (auto idx : indices)
    {
        if (idx < 0 || idx >= boxes.size()) continue;  // インデックスが不正な場合はスキップ

        // 検出された物体の領域を切り取る
        cv::Rect objectBox = boxes[idx];  // バウンディングボックスを取得

        // バウンディングボックスが画像の範囲を超えないように調整する
        objectBox.x = std::max(0, objectBox.x);
		objectBox.y = std::max(0, objectBox.y);
        objectBox.width = std::min(objectBox.width, imageWidth - objectBox.x);
		objectBox.height = std::min(objectBox.height, imageHeight - objectBox.y);

        // バウンディングボックスを使ってROI（関心領域）を画像から切り取る
        cv::Mat roi = grayImage(objectBox);  // バウンディングボックスで切り取る（ROI）

        // 顔検出を実行
        vector<cv::Rect> facesInRegion;
        face_cascade.detectMultiScale(roi, facesInRegion);

        // 顔の位置をマスク画像に描画
        for (const auto& face : facesInRegion)
        {
            // 顔の位置を元の画像の座標に戻す
            cv::Rect faceRect(objectBox.x + face.x, objectBox.y + face.y, face.width, face.height);
            cv::rectangle(peopleMask, faceRect, cv::Scalar(255), cv::FILLED);  // 白い長方形を描画
        }
    }
    return peopleMask;
}
/**************************************************************************//**
@関数名           ObjectDetector::boundingBoxDetector
@概要             入力画像から物体を検出し、物体のバウンディングボックスを取得する
@パラメータ[in]   uImage: 入力画像（UMat型）
@パラメータ[out]  なし
@戻り値           検出された物体の予測結果（vector<Predicted> 型）
@詳細             YOLOモデルを使用して物体を検出し、信頼度が指定された閾値を超える物体を選択する。
                　予測結果を元に、物体のクラスID、信頼度、バウンディングボックスを
                  格納したPredicted構造体を作成し、最終的に非最大抑制（NMS）を適用して冗長な検出を排除します。
******************************************************************************/
vector<Predicted> ObjectDetector::boundingBoxDetector(const cv::UMat& uImage)
{
    vector<Predicted> predictions;                                      // 物体の予測結果を格納するベクター
    vector<cv::Mat> dataOutputs = getOutputs(uImage);                   // YOLOモデルからの出力結果を取得
    // 画像のサイズと入力サイズに基づきスケーリングファクターを計算
    float xFactor = static_cast<float>(uImage.cols) / INPUT_WIDTH;
    float yFactor = static_cast<float>(uImage.rows) / INPUT_HEIGHT;
    // データのサイズを取得
    const int rows = dataOutputs[0].size[1];
    const int cols = dataOutputs[0].size[2];
    // 出力データへのポインタを取得
    float* data = (float*)dataOutputs[0].data;

    predictions.reserve(rows); // メモリの使用を最適化するため、予め容量を予約しておく
    for (int i = 0; i < rows; i++, data += cols)
    {
        float confidence = data[4];                     // 信頼度を取得
        // 信頼度が閾値を超えていれば物体を検出
        if (confidence > CONFIDENCE_THRESHOLD)
        {
            // バウンディングボックスの座標とサイズをスケールアップ
            float cx = data[0] * xFactor;
            float cy = data[1] * yFactor;
            float w = data[2] * xFactor;
            float h = data[3] * yFactor;
            // クラスIDを取得（最大の予測確率を持つクラス）
            int classId = max_element(data + 5, data + cols) - (data + 5);
            // 検出された物体をPredicted構造体に格納
            predictions.push_back({ classId, confidence, cv::Rect(cx - w / 2, cy - h / 2, w , h) });
        }
    }
    // 非最大抑制（NMS）を適用して冗長な検出を排除
    return applyNMS(predictions);
}

/**************************************************************************//**
@関数名           Detector::applyNMS
@概要             非最大抑制（NMS）を適用し、重複する検出を削除する
@パラメータ[in]   predictions: 検出された物体のリスト（予測結果）
@パラメータ[out]  なし
@戻り値           NMSが適用された後の物体のリスト（vector<Predicted>）
@詳細             NMSアルゴリズムを使用して、重複する物体検出を削除します。
                  信頼度が高い検出結果を残し、他の重複する結果を取り除きます。
                  具体的には、各物体検出に対して、他の物体との重複度合いを計算し、
                  重複が一定の閾値以上であればその検出を除外します。
                  最終的に、最も信頼性の高い検出結果が残ります。
******************************************************************************/
vector<Predicted> ObjectDetector::applyNMS(const vector<Predicted>& predictions)
{
    // 予測結果が空の場合、空のベクターを返す
    if (predictions.empty()) return {};

    // スコアとバウンディングボックスのためのベクターを準備
    vector<float> scores;
    vector<cv::Rect> boxes;

    scores.reserve(predictions.size()); // 予測結果の数に応じてメモリを予約
    boxes.reserve(predictions.size());  // 同様にバウンディングボックスのメモリを予約

    // 予測結果からスコアとボックスを抽出
    for (const auto& object : predictions)
    {
        scores.push_back(object.confidence);
        boxes.emplace_back(object.box);
    }

    // 非最大抑制 (NMS) を適用
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    // 結果を格納するためのベクター
    vector<Predicted> results;
    results.reserve(indices.size());
    for (auto idx : indices)
    {
        results.emplace_back(predictions[idx]);
    }
    return results;
}

/**************************************************************************//**
@関数名           Detector::drawBoundingBoxes
@概要             画像に物体の検出結果（バウンディングボックス）を描画する
@パラメータ[in]   image: 入力画像
                  detected: 検出された物体のリスト（予測結果）
@パラメータ[out]  なし
@戻り値           物体のバウンディングボックスが描画された画像（Mat）
@詳細             画像にバウンディングボックスとラベルを描画する
******************************************************************************/
cv::Mat ObjectDetector::drawBoundingBoxes(const cv::Mat& image, const vector<Predicted>& detected)
{
    int baseLine;
    cv::Mat result = image.clone();
    for (const auto& idx : detected)
    {
        //string label = classList[idx.classId] + ":" + to_string(idx.confidence);
        string label = classList[idx.classId];
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = idx.box.x;
        int y = idx.box.y - baseLine;
        rectangle(result, idx.box, cv::Scalar(0, 255, 0), 2);
        putText(result, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    return result;
}

/**************************************************************************//**
@関数名           ObjectDetector::getOutputs
@概要             入力画像から物体を検出するための出力を取得する
@パラメータ[in]   uImage: 入力画像（UMat型）
@パラメータ[out]  なし
@戻り値           モデルの出力結果（vector<cv::Mat> 型）
@詳細             YOLOモデルを実行し、入力画像に対して物体検出を行い、出力を取得する。
******************************************************************************/
vector<cv::Mat> ObjectDetector::getOutputs(const cv::UMat& uImage)
{
    cv::UMat uBlob;
    vector<cv::Mat> outputs;
    // 入力画像に対して前処理を実施（サイズ変更、スケーリング、チャネル順序変更）
    cv::dnn::blobFromImage(uImage, uBlob, SCALE, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    // モデルの入力として設定
    net.setInput(uBlob);
    // YOLOモデルを実行し、出力を取得
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    // 結果を返す（Mat型のベクター）
    return outputs;
}




