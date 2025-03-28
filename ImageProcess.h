#ifndef _IMAGE_PROCESS_H_
#define _IMAGE_PROCESS_H_

#include "OpenCV.h"

using namespace std;

/**
* @class ImageProcess
* @brief �摜�������s���N���X�B
*
* ���̃N���X�́A�w�肳�ꂽ�}�X�N��p���ē��͉摜�̈ꕔ���ڂ���������񋟂��܂��B
* OpenCL�𗘗p���ď��������������邱�Ƃ��\�ł��B
*/
class ImageProcess
{
public:
	/**
	* @brief �R���X�g���N�^�B
	*
	* ImageProcess�I�u�W�F�N�g�����������܂��B
	*/
	ImageProcess();

	/**
	* @brief �R���X�g���N�^�B
	*
	* ImageProcess�I�u�W�F�N�g�̃N���[���A�b�v���s���܂��B
	*/
	~ImageProcess();

	/**
	* @brief �w�肳�ꂽ�}�X�N�Ɋ�Â��ĉ摜�̈ꕔ���ڂ����B
	*
	* ���͂��ꂽ�}�X�N�Ɋ�Â��āA�����������ڂ����A���������͌��̉摜��ێ����܂��B
	* OpenCL�����p�\�ȏꍇ�́AGPU�ŏ������s���A���������܂��B
	*
	* @param mask �ڂ�����K�p���Ȃ��������w�肷��}�X�N�摜�B�^��cv::Mat�i8�r�b�g�j�B
	* @param uFrame ���͉摜�B�^��cv::UMat�iOpenCL�Ή��j�B
	* @return �ڂ���������̉摜�B�^��cv::UMat�B
	*/
	cv::UMat blurProcess(const cv::Mat& mask, const cv::UMat& uFrame);
};

#endif //!_IMAGE_PROCESS_H_