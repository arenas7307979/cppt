#include <queue>
#include <thread>
#include <condition_variable>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Imu.h>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include "ros_utility.h"
using namespace std;
using namespace message_filters;
using namespace sensor_msgs;

class Node {
public:
    using Measurements = vector<pair<pair<ImageConstPtr, ImageConstPtr>, vector<ImuConstPtr>>>;
    Node() {
        t_system = std::thread(&Node::SystemThread, this);
    }
    ~Node() {}

    void ReadFromNodeHandle(ros::NodeHandle& nh) {
        std::string config_file;
        config_file = readParam<std::string>(nh, "config_file");

        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        fs["imu_topic"] >> imu_topic;
        fs["image_topic"] >> img_topic[0];
        fs["image_r_topic"] >> img_topic[1];

        cv::Size image_size;
        image_size.height = fs["image_height"];
        image_size.width = fs["image_width"];

        cv::Mat Tbc0, Tbc1, Tbi;
        fs["T_BC0"] >> Tbc0;
        fs["T_BC1"] >> Tbc1;
        fs["T_BI"] >> Tbi;

        std::vector<double> intrinsics0, intrinsics1;
        std::vector<double> distortion_coefficients0, distortion_coefficients1;

        fs["intrinsics0"] >> intrinsics0;
        fs["distortion_coefficients0"] >> distortion_coefficients0;
        fs["intrinsics1"] >> intrinsics1;
        fs["distortion_coefficients1"] >> distortion_coefficients1;

        cv::Mat K0, K1, D0, D1;
        K0 = (cv::Mat_<double>(3, 3) << intrinsics0[0], 0, intrinsics0[2],
                                        0, intrinsics0[1], intrinsics0[3],
                                        0, 0, 1);
        K1 = (cv::Mat_<double>(3, 3) << intrinsics1[0], 0, intrinsics1[2],
                                        0, intrinsics1[1], intrinsics1[3],
                                        0, 0, 1);
        D0.create(distortion_coefficients0.size(), 1, CV_64F);
        D1.create(distortion_coefficients1.size(), 1, CV_64F);

        for(int i = 0, n = distortion_coefficients0.size(); i < n; ++i)
            D0.at<double>(i) = distortion_coefficients0[i];

        for(int i = 0, n = distortion_coefficients1.size(); i < n; ++i)
            D1.at<double>(i) = distortion_coefficients1[i];

        cv::Mat Tc1c0 = Tbc1.inv() * Tbc0;
        cv::Mat Rc1c0, tc1c0;
        Tc1c0.rowRange(0, 3).colRange(0, 3).copyTo(Rc1c0);
        Tc1c0.col(3).rowRange(0, 3).copyTo(tc1c0);
        cv::Mat R0, R1, P0, P1; // R0 = Rcp0c0, R1 = cp1c1
        cv::stereoRectify(K0, D0, K1, D1, image_size, Rc1c0, tc1c0, R0, R1, P0, P1, cv::noArray());

        double f, cx, cy;
        double b;
        f = P0.at<double>(0, 0);
        cx = P0.at<double>(0, 2);
        cy = P0.at<double>(1, 2);
        b = -P1.at<double>(0, 3) / f;

        cv::Mat M1l, M2l, M1r, M2r;
        cv::initUndistortRectifyMap(K0, D0, R0, P0, image_size, CV_32F, M1l, M2l);
        cv::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_32F, M1r, M2r);

        // fix entrinsics
        Eigen::Matrix4d temp_T;
        Eigen::Matrix3d temp_R;
        cv::cv2eigen(Tbc0, temp_T);
        Sophus::SE3d sTbc0(temp_T);

        cv::cv2eigen(Tbc1, temp_T);
        Sophus::SE3d sTbc1(temp_T);

        cv::cv2eigen(Tbi, temp_T);
        Sophus::SE3d sTbi(temp_T);

        cv::cv2eigen(R0, temp_R);
        Sophus::SE3d sTcp0c0;
        sTcp0c0.setRotationMatrix(temp_R);

        cv::cv2eigen(R1, temp_R);
        Sophus::SE3d sTcp1c1;
        sTcp1c1.setRotationMatrix(temp_R);

        Sophus::SE3d sTbcp0 = sTbc0 * sTcp0c0.inverse();
        Sophus::SE3d sTbcp1 = sTbc1 * sTcp1c1.inverse();
        fs.release();
    }

    void ImageCallback(const ImageConstPtr& img_msg, const ImageConstPtr& img_r_msg) {
        unique_lock<mutex> lock(m_buf);
        img_buf.emplace(img_msg, img_r_msg);
        cv_system.notify_one();
    }

    void ImuCallback(const ImuConstPtr& imu_msg) {
        unique_lock<mutex> lock(m_buf);
        imu_buf.emplace(imu_msg);
        cv_system.notify_one();
    }

    Measurements GetMeasurements() {
        // The buffer mutex is locked before this function be called.
        Measurements measurements;

        while (1) {
            if (imu_buf.empty() || img_buf.empty())
                return measurements;

            double img_ts = img_buf.front().first->header.stamp.toSec();
            // catch the imu data before image_timestamp
            // ---------------^-----------^ image
            //                f           f+1
            // --x--x--x--x--x--x--x--x--x- imu
            //   f                       b
            // --o--o--o--o--o^-?---------- collect data in frame f

            // if ts(imu(b)) < ts(img(f)), wait imu data
            if (imu_buf.back()->header.stamp.toSec() < img_ts) {
                return measurements;
            }
            // if ts(imu(f)) > ts(img(f)), img data faster than imu data, drop the img(f)
            if (imu_buf.front()->header.stamp.toSec() > img_ts) {
                img_buf.pop();
                continue;
            }

            pair<ImageConstPtr, ImageConstPtr> img_msg = img_buf.front();
            img_buf.pop();

            vector<ImuConstPtr> IMUs;
            while (imu_buf.front()->header.stamp.toSec() < img_ts) {
                IMUs.emplace_back(imu_buf.front());
                imu_buf.pop();
            }
            // IMUs.emplace_back(imu_buf.front()); // ??
            measurements.emplace_back(img_msg, IMUs);
        }
    }

    void SystemThread() {
        while(1) {
            Measurements measurements;
            std::unique_lock<std::mutex> lock(m_buf);
            cv_system.wait(lock, [&] {
                return (measurements = GetMeasurements()).size() != 0;
            });
            lock.unlock();

            // TODO
            for(auto& meas : measurements) {
                auto& img_msg = meas.first.first;
                auto& img_msg_right = meas.first.second;
                double timestamp = img_msg->header.stamp.toSec();

                cv::Mat img_left, img_right;
                img_left = cv_bridge::toCvCopy(img_msg, "mono8")->image;
                img_right = cv_bridge::toCvCopy(img_msg_right, "mono8")->image;
            }
        }
    }

    string imu_topic;
    string img_topic[2];

    mutex m_buf;
    queue<ImuConstPtr> imu_buf;
    queue<pair<ImageConstPtr, ImageConstPtr>> img_buf;

    condition_variable cv_system;
    thread t_system;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cppt_player");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    Node node;
    node.ReadFromNodeHandle(nh);

    message_filters::Subscriber<Image> sub_img[2] {{nh, node.img_topic[0], 100},
                                                   {nh, node.img_topic[1], 100}};
    TimeSynchronizer<Image, Image> sync(sub_img[0], sub_img[1], 100);
    sync.registerCallback(boost::bind(&Node::ImageCallback, &node, _1, _2));

    ros::Subscriber sub_imu = nh.subscribe(node.imu_topic, 2000, &Node::ImuCallback, &node,
                                           ros::TransportHints().tcpNoDelay());

    ROS_INFO_STREAM("Player is ready.");

    ros::spin();
    return 0;
}
