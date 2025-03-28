#ifndef _OBJECT_DETECTOR_H_
#define _OBJECT_DETECTOR_H_

#include "OpenCV.h"

// �萔�̒�`�iYOLO���f���̐ݒ�l�j
#define SCALE (1.0 / 255.0)             // ��f�l�𐳋K������X�P�[��
#define INPUT_WIDTH 640.0               // YOLO���f���̓��͉摜��
#define INPUT_HEIGHT 640.0              // YOLO���f���̓��͉摜����
#define CONFIDENCE_THRESHOLD 0.5        // ���̂����o���邽�߂̐M���x��臒l
#define SCORE_THRESHOLD 0.4             // �N���X���ʂ̃X�R�A臒l
#define NMS_THRESHOLD 0.2               // ��ő�}���iNMS�j��臒l
#define AREAS_THRESHOLD 3600            // ���o���ꂽ�I�u�W�F�N�g�̍ŏ��ʐ�臒l

using namespace std;
/**
* @class ObjectDetector
* @brief YOLOv5���f�����g�p���ĕ��̌��o���s���N���X�B
* ���̃N���X�́A�w�肳�ꂽ�摜�ɑ΂���YOLOv5���f����p���ĕ��̌��o���s���A
* ���o���ꂽ���̂ɑ΂��ăo�E���f�B���O�{�b�N�X��`�悵�܂��B�܂��A��ő�}��
* (NMS)���g�p���Č��o���ʂ��t�B���^�����O���܂��B
*/
class ObjectDetector
{
public:
    /**
    * @brief �R���X�g���N�^�i�f�t�H���g�R���X�g���N�^�j�B
    * �f�t�H���g�̐ݒ��ObjectDetector�I�u�W�F�N�g�����������܂��B
    */
    ObjectDetector();

    /**
    * @brief �R���X�g���N�^�i�f�t�H���g�R���X�g���N�^�j�B
    * �f�t�H���g�̐ݒ��ObjectDetector�I�u�W�F�N�g�����������܂��B
    */
    ~ObjectDetector();

	/**
	* @brief �R���X�g���N�^�B
	* �w�肳�ꂽ���f���t�@�C���ƃN���X���t�@�C������ObjectDetector�I�u�W�F�N�g�����������܂��B
	* @param modelPath ���f���t�@�C���̃p�X�iONNX�`���j
	* @param classFile �N���X�����L�ڂ��ꂽ�t�@�C���̃p�X
	*/
    explicit ObjectDetector(const string& modelPath, const string& classFile, const string& cascadePath);

    /**
    * @brief �摜���畨�̂����o���A�o�E���f�B���O�{�b�N�X��Ԃ��B
    * ���͉摜�ɑ΂��ĕ��̌��o���s���A���o���ꂽ���̂̃o�E���f�B���O�{�b�N�X��Ԃ��܂��B
    * @param uImage ���͉摜�icv::UMat�`���j�B
    * @return ���o���ꂽ���̂̃��X�g�iPredicted�^�̃x�N�g���j�B
    */
    vector<Predicted> boundingBoxDetector(const cv::UMat& uImage);

    /**
    * @brief �摜����l�����o����B
    * ���͉摜�ɑ΂��Đl�����o���A���̌��ʂ�Ԃ��܂��B
    * @param uImage ���͉摜�icv::UMat�`���j�B
    * @return �l�����o�����摜�icv::Mat�`���j�B
    */
    cv::Mat detectorPeople(const cv::UMat& uImage);

    /**
    * @brief �o�E���f�B���O�{�b�N�X���摜�ɕ`�悷��B
    * ���o���ꂽ���̂̃o�E���f�B���O�{�b�N�X����͉摜�ɕ`�悵�܂��B
    * @param image ���͉摜�icv::Mat�`���j�B
    * @param detected ���o���ꂽ���̂̃��X�g�iPredicted�^�̃x�N�g���j�B
    * @return �o�E���f�B���O�{�b�N�X��`�悵���摜�icv::Mat�`���j�B
    */
    cv::Mat drawBoundingBoxes(const cv::Mat& image, const vector<Predicted>& detected);

private:
    /**
    * @brief YOLO���f���̏o�͂��擾����B
    * ���͉摜�ɑ΂���YOLO���f�������s���A�o�͂��擾���܂��B
    * @param uImage �O�������ꂽ���͉摜�icv::UMat�`���j�B
    * @return YOLO�l�b�g���[�N�̏o�́icv::Mat�^�̃x�N�g���j�B   
    */
    vector<cv::Mat> getOutputs(const cv::UMat& uImage);

    /**
    * @brief �N���X�����X�g���t�@�C������ǂݍ��ށB
    * �w�肳�ꂽ�p�X����N���X�����X�g��ǂݍ��݂܂��B
    * @param path �N���X�����X�g�t�@�C���̃p�X�B
    * @return �N���X���̃��X�g�i������^�̃x�N�g���j�B
    */
    vector<string> loadClasses(const string& path);

    /**
    * @brief ��ő�}���iNMS�j��K�p���ė\�����ʂ��t�B���^�����O����B
    *
    * NMS���g�p���āA�d������o�E���f�B���O�{�b�N�X��r�����܂��B
    *
    * @param predictions ���o���ꂽ���̗̂\�����ʂ̃��X�g�iPredicted�^�̃x�N�g���j�B
    * @return NMS��̗\�����ʃ��X�g�iPredicted�^�̃x�N�g���j�B
    */
    vector<Predicted> applyNMS(const vector<Predicted>& predictions);

    cv::dnn::Net net;           // YOLOv5���f���̃j���[�����l�b�g���[�N�B
    vector<string> classList;   // �N���X�����X�g�B
    cv::CascadeClassifier face_cascade;     //Haar Cascade�i��̌��o�̃��C�u�����j
};
#endif // !
