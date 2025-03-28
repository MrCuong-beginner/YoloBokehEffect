/**************************************************************************//**
@クラス名         ImageProcess
@概要             画像処理を行うクラス
@詳細             OpenCLを利用し、指定されたマスクを用いて画像の一部をぼかす処理を実装
******************************************************************************/
#include "ImageProcess.h"

ImageProcess::ImageProcess()
{
}

ImageProcess::~ImageProcess()
{
}

/**************************************************************************//**
@関数名           ImageProcess::blurProcess
@概要             指定されたマスクに基づいて画像の一部をぼかす
@パラメータ[in]   mask: ぼかしを適用しない部分を指定するマスク画像（cv::Mat 型, 8bit）
                  uFrame: 入力画像（cv::UMat 型）
@パラメータ[out]  なし
@戻り値           ぼかし処理後の画像（cv::UMat 型）
@詳細             - OpenCL が利用可能な場合、GPU で処理を行う
                  - マスクの白い部分をぼかし、黒い部分は元の画像を保持
                  - ガウシアンブラー（55×55）を適用
******************************************************************************/
cv::UMat ImageProcess::blurProcess(const cv::Mat& mask, const cv::UMat& uFrame)
{
    // OpenCLが利用可能かどうかを確認
    if (!cv::ocl::haveOpenCL())
    {
        std::cout << "OpenCLは利用できません！" << std::endl;
        return uFrame;  // 利用不可の場合は元の画像を返す
    }

    // CPUのマスクをGPUへ転送
    cv::UMat uMask = mask.getUMat(cv::ACCESS_READ);

    // マスクを反転 (GPUで処理)
    cv::UMat uInverseMask;
    cv::bitwise_not(uMask, uInverseMask);

    // 画像をぼかす (GPUで処理)
    cv::UMat uBlurredImage;
    cv::GaussianBlur(uFrame, uBlurredImage, cv::Size(55, 55), 0);

    // 出力画像を作成し、元画像をコピー
    cv::UMat uOutput;
    uFrame.copyTo(uOutput);

    // ぼかし画像を適用
    //uBlurredImage.copyTo(uOutput, uInverseMask);
    uBlurredImage.copyTo(uOutput, uMask);

    return uOutput;
}
