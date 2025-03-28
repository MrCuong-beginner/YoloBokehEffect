#ifndef _OPENCV_H_
#define _OPENCV_H_

#include <opencv2/opencv.hpp>       // OpenCVの基本的な機能を含むヘッダファイル
#include <opencv2/dnn.hpp>          // DNNモジュール（ディープラーニング）のヘッダファイル
#include <opencv2/core/ocl.hpp>     // OpenCLのサポートを含むヘッダファイル

/**
* @struct Predicted
* @brief 物体検出の予測結果を保持する構造体。
* この構造体は、YOLOなどの物体検出モデルによって返される
* 検出結果を格納します。各予測には、クラスID、信頼度、バウンディングボックスが含まれます。
*/
struct Predicted
{
    int classId;            //検出された物体のクラスID。
    float confidence;       //予測された物体の信頼度（0〜1の範囲）
    cv::Rect box;           //検出された物体のバウンディングボックス（物体の位置とサイズを定義）。
};

#endif