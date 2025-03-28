/**************************************************************************//**
@�N���X��         ImageProcess
@�T�v             �摜�������s���N���X
@�ڍ�             OpenCL�𗘗p���A�w�肳�ꂽ�}�X�N��p���ĉ摜�̈ꕔ���ڂ�������������
******************************************************************************/
#include "ImageProcess.h"

ImageProcess::ImageProcess()
{
}

ImageProcess::~ImageProcess()
{
}

/**************************************************************************//**
@�֐���           ImageProcess::blurProcess
@�T�v             �w�肳�ꂽ�}�X�N�Ɋ�Â��ĉ摜�̈ꕔ���ڂ���
@�p�����[�^[in]   mask: �ڂ�����K�p���Ȃ��������w�肷��}�X�N�摜�icv::Mat �^, 8bit�j
                  uFrame: ���͉摜�icv::UMat �^�j
@�p�����[�^[out]  �Ȃ�
@�߂�l           �ڂ���������̉摜�icv::UMat �^�j
@�ڍ�             - OpenCL �����p�\�ȏꍇ�AGPU �ŏ������s��
                  - �}�X�N�̔����������ڂ����A���������͌��̉摜��ێ�
                  - �K�E�V�A���u���[�i55�~55�j��K�p
******************************************************************************/
cv::UMat ImageProcess::blurProcess(const cv::Mat& mask, const cv::UMat& uFrame)
{
    // OpenCL�����p�\���ǂ������m�F
    if (!cv::ocl::haveOpenCL())
    {
        std::cout << "OpenCL�͗��p�ł��܂���I" << std::endl;
        return uFrame;  // ���p�s�̏ꍇ�͌��̉摜��Ԃ�
    }

    // CPU�̃}�X�N��GPU�֓]��
    cv::UMat uMask = mask.getUMat(cv::ACCESS_READ);

    // �}�X�N�𔽓] (GPU�ŏ���)
    cv::UMat uInverseMask;
    cv::bitwise_not(uMask, uInverseMask);

    // �摜���ڂ��� (GPU�ŏ���)
    cv::UMat uBlurredImage;
    cv::GaussianBlur(uFrame, uBlurredImage, cv::Size(55, 55), 0);

    // �o�͉摜���쐬���A���摜���R�s�[
    cv::UMat uOutput;
    uFrame.copyTo(uOutput);

    // �ڂ����摜��K�p
    //uBlurredImage.copyTo(uOutput, uInverseMask);
    uBlurredImage.copyTo(uOutput, uMask);

    return uOutput;
}
