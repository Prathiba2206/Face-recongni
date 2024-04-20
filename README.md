# Face-recongni
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the pre-trained face detection model
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_alt.xml")) {
        printf("Error loading face cascade!\n");
        return -1;
    }

    // Initialize camera or load video file
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Error opening camera!\n");
        return -1;
    }

    Mat frame;
    while (cap.read(frame)) {
        // Convert the frame to grayscale for face detection
        Mat frame_gray;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        // Detect faces in the frame
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Draw rectangles around the detected faces
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2);
        }

        // Display the frame with detected faces
        imshow("Face Detection", frame);

        // Wait for key press to exit
        if (waitKey(1) == 27) {
            break;
        }
    }

    // Release the camera and close OpenCV windows
    cap.release();
    destroyAllWindows();

    return 0;
}

