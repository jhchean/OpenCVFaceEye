#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <android/asset_manager_jni.h>
#include <android/log.h>



using namespace cv;
using namespace std;

//얼굴 알아보기
extern "C" {

// 안경좌표 임시 저장소
int glass_point_x =0;
int glass_point_y =0;

//이미지 중첩 함수 헤더 선언
void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location);

float resize(Mat img_src, Mat &img_resize, int resize_width){

    float scale = resize_width / (float)img_src.cols ;
    if (img_src.cols > resize_width) {
        int new_height = cvRound(img_src.rows * scale);
        resize(img_src, img_resize, Size(resize_width, new_height));
    }
    else {
        img_resize = img_src;
    }
    return scale;
}


JNIEXPORT jlong JNICALL
Java_com_mycompany_opencv_MainActivity_detect(JNIEnv *, jobject,
                                              jlong cascadeClassifier_face,
                                              jlong cascadeClassifier_eye,
                                              jlong addrInput,
                                              jlong addrResult
                                             , jlong glassImage                      //안경이미지 추가 받음..
) {

    jlong ret = 0;

    Mat &img_input = *(Mat *) addrInput;
    Mat &img_result = *(Mat *) addrResult;

    Mat &img_glass = *(Mat *) glassImage;   //안경 이미지 변수 연결

    img_result = img_input.clone();

    std::vector<Rect> faces;

    Mat img_gray;

    cvtColor(img_input, img_gray, COLOR_BGR2GRAY);
    equalizeHist(img_gray, img_gray);

    Mat img_resize;
    float resizeRatio = resize(img_gray, img_resize, 640);

    //-- Detect faces
       ((CascadeClassifier *) cascadeClassifier_face)->detectMultiScale( img_resize, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

      //로그
      //   __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ",                   (char *) "face %d found ", faces.size());


    for (int i = 0; i < faces.size(); i++) {
        
        double real_facesize_x = faces[i].x / resizeRatio;
        double real_facesize_y = faces[i].y / resizeRatio;
        double real_facesize_width = faces[i].width / resizeRatio;
        double real_facesize_height = faces[i].height / resizeRatio;


        //얼굴 중심 원그리기
        //Point center( real_facesize_x + real_facesize_width / 2, real_facesize_y + real_facesize_height/2);
        /*ellipse(img_result, center, Size( real_facesize_width / 2, real_facesize_height / 2), 0, 0, 360,
                Scalar(255, 0, 255), 2, 8, 0);*/
                                   // 2 두께  // 8 선스타일   // 0 옵셋

        //콧구멍 제외시키기위해 얼굴 아래를 잘라내면서 얼굴 영역 가저옴..
        Rect face_area(real_facesize_x, real_facesize_y, real_facesize_width,real_facesize_height/2);

        //얼굴 흑백변환 및 영역가져오기
        Mat faceROI = img_gray( face_area );

        //눈 영역 변수 선언
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        ((CascadeClassifier *) cascadeClassifier_eye)->detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        //눈의 갯수 가져오기
        ret = eyes.size();

        if(ret>0){


            
            //눈이 두개 잡혔을 때 .. 저장해두고 다음에 씀..
            if(ret == 2){
                //왼쪽 눈의 좌표를 구한다.
                if(eyes[0].x < eyes[1].x){
                    glass_point_x = real_facesize_x + eyes[0].x ;
                    glass_point_y = real_facesize_y + eyes[0].y ;
                }else{
                    glass_point_x = real_facesize_x + eyes[1].x;
                    glass_point_y = real_facesize_y + eyes[1].y;
                }
            }


            // Resize image
               //목표한 가로길이를 구한다.
               int tempx = (int)real_facesize_width;
               //가로길이의 비율을 세로에 적용하여 목표한 세로길이를 구한다.
               int tempy =  (int) real_facesize_width * img_glass.rows /img_glass.cols;
               //안경의 사이즈를 줄인다.
               cv::resize( img_glass, img_glass, cv::Size(tempx ,tempy ), 0, 0, CV_INTER_NN );

            //안경을 어굴에 오버레이 시킨다.
            overlayImage(img_result, img_glass, img_result, cv::Point((int) (glass_point_x-tempx/5),(int)(glass_point_y-tempy/5) ));

        }



        //눈 갯수만큼 동그라미 그리기
        /*for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( real_facesize_x + eyes[j].x + eyes[j].width/2, real_facesize_y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( img_result, eye_center, radius, Scalar( 255, 0, 0 ), 2, 8, 0 );

        }*/
        //눈 갯수만큼 동그라미 그리기
    }

    return  ret;
}


JNIEXPORT jlong JNICALL
Java_com_mycompany_opencv_MainActivity_loadCascade(JNIEnv *env, jobject, jstring cascadeFileName) {

    const char *nativeFileNameString = env->GetStringUTFChars(cascadeFileName, JNI_FALSE);

    string baseDir("/storage/emulated/0/");
    baseDir.append(nativeFileNameString);
    const char *pathDir = baseDir.c_str();

    jlong ret = 0;
    ret = (jlong) new CascadeClassifier(pathDir);
    if (((CascadeClassifier *) ret)->empty()) {
        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ",
                            "CascadeClassifier로 로딩 실패  %s", nativeFileNameString);
    }
    else
        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ",
                            "CascadeClassifier로 로딩 성공 %s", nativeFileNameString);

    return ret;
}



void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
                  cv::Mat &output, cv::Point2i location)
{
    background.copyTo(output);


    // start at the row indicated by location, or at row 0 if location.y is negative.
    for(int y = std::max(location.y , 0); y < background.rows; ++y)
    {
        int fY = y - location.y; // because of the translation

        // we are done of we have processed all rows of the foreground image.
        if(fY >= foreground.rows)
            break;

        // start at the column indicated by location,

        // or at column 0 if location.x is negative.
        for(int x = std::max(location.x, 0); x < background.cols; ++x)
        {
            int fX = x - location.x; // because of the translation.

            // we are done with this row if the column is outside of the foreground image.
            if(fX >= foreground.cols)
                break;

            // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
            double opacity =
                    ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

                    / 255.;


            // and now combine the background and foreground pixel, using the opacity,

            // but only if opacity > 0.
            for(int c = 0; opacity > 0 && c < output.channels(); ++c)
            {
                unsigned char foregroundPx =
                        foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx =
                        background.data[y * background.step + x * background.channels() + c];
                output.data[y*output.step + output.channels()*x + c] =
                        backgroundPx * (1.-opacity) + foregroundPx * opacity;
            }
        }
    }
}















}




//카메라 띄우기
/*extern "C" {
JNIEXPORT void JNICALL
Java_com_mycompany_opencv_MainActivity_ConvertRGBtoGray(JNIEnv *env, jobject instance,
                                                        jlong matAddrInput, jlong matAddrResult) {

    Mat &matInput = *(Mat *) matAddrInput;
    Mat &matResult = *(Mat *) matAddrResult;

    cvtColor(matInput, matResult, CV_RGBA2GRAY);

}


            JNIEXPORT jstring JNICALL
            Java_com_mycompany_opencv_MainActivity_stringFromJNI(
                    JNIEnv *env,
                    jobject *//* this *//*) {


                std::string hello = "Hello from C++";
                return env->NewStringUTF(hello.c_str());
            }
}

*/