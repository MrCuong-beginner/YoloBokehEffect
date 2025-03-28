#ifndef _IMAGE_PROCESS_H_
#define _IMAGE_PROCESS_H_

#include "OpenCV.h"

using namespace std;

/**
* @class ImageProcess
* @brief 画像処理を行うクラス。
*
* このクラスは、指定されたマスクを用いて入力画像の一部をぼかす処理を提供します。
* OpenCLを利用して処理を高速化することが可能です。
*/
class ImageProcess
{
public:
	/**
	* @brief コンストラクタ。
	*
	* ImageProcessオブジェクトを初期化します。
	*/
	ImageProcess();

	/**
	* @brief コンストラクタ。
	*
	* ImageProcessオブジェクトのクリーンアップを行います。
	*/
	~ImageProcess();

	/**
	* @brief 指定されたマスクに基づいて画像の一部をぼかす。
	*
	* 入力されたマスクに基づいて、白い部分をぼかし、黒い部分は元の画像を保持します。
	* OpenCLが利用可能な場合は、GPUで処理を行い、高速化します。
	*
	* @param mask ぼかしを適用しない部分を指定するマスク画像。型はcv::Mat（8ビット）。
	* @param uFrame 入力画像。型はcv::UMat（OpenCL対応）。
	* @return ぼかし処理後の画像。型はcv::UMat。
	*/
	cv::UMat blurProcess(const cv::Mat& mask, const cv::UMat& uFrame);
};

#endif //!_IMAGE_PROCESS_H_