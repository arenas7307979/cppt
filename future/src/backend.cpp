#include "backend.h"
#include "ceres/local_parameterization_se3.h"
#include "ceres/projection_factor.h"
#include <ceres/ceres.h>
#include <glog/logging.h>

BackEnd::BackEnd(double focal_length_,
                 double gyr_n, double acc_n,
                 double gyr_w, double acc_w,
                 const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
                 const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
                 uint window_size_, double min_parallax_)
    : focal_length(focal_length_), p_rl(p_rl_), p_bc(p_bc_), q_rl(q_rl_), q_bc(q_bc_), window_size(window_size_),
      frame_count(0), next_frame_id(0), state(NEED_INIT), min_parallax(min_parallax_ / focal_length)
{
    gyr_noise_cov = gyr_n * gyr_n * Eigen::Matrix3d::Identity();
    acc_noise_cov = acc_n * acc_n * Eigen::Matrix3d::Identity();
    gyr_bias_cov = gyr_w * gyr_w * Eigen::Matrix3d::Identity();
    acc_bias_cov = acc_w * acc_w * Eigen::Matrix3d::Identity();

    para_pose = new double[(window_size + 1) * 7];
    para_speed_bias = new double[(window_size + 1) * 7];
    para_features = new double[para_features_capacity * 1];

    request_reset_flag = false;

    thread_ = std::thread(&BackEnd::Process, this);
}

BackEnd::~BackEnd() {
    delete [] para_pose;
    delete [] para_speed_bias;
    delete [] para_features;
}

void BackEnd::PushFrame(FramePtr frame) {
    m_buffer.lock();
    frame_buffer.emplace_back(frame);
    frame->id = next_frame_id++;
    m_buffer.unlock();
    cv_buffer.notify_one();
}

void BackEnd::Process() {
    while(1) {
        std::vector<FramePtr> measurements;
        std::unique_lock<std::mutex> lock(m_buffer);
        cv_buffer.wait(lock, [&] {
            measurements = std::vector<FramePtr>(frame_buffer.begin(), frame_buffer.end());
            frame_buffer.clear();
            return (!measurements.empty() || request_reset_flag);
        });

        if(request_reset_flag) {
            Reset();
            request_reset_flag = false;
            continue;
        }

        for(auto& frame : measurements) {
            ProcessFrame(frame);
        }
    }
}

void BackEnd::ProcessFrame(FramePtr frame) {
    // test graph
    d_frames.emplace_back(frame);

    // start the margin when the sliding window fill the frames
    marginalization_flag = AddFeaturesCheckParallax(frame);

    if(state == NEED_INIT) {
        frame->q_wb = Sophus::SO3d();
        frame->p_wb.setZero();
        int num_mps = Triangulate();

        if(num_mps > 50) {
            frame_count++;
            state = CV_ONLY;
        }
        else {
            LOG(INFO) << "Stereo init fail, number of mappoints less than 50: " << num_mps;
            Reset();
        }
    }
    else {
        if(state == CV_ONLY) {
            frame->q_wb = d_frames[frame_count-1]->q_wb;
            frame->p_wb = d_frames[frame_count-1]->p_wb;
            SolveBA();
            Triangulate();
            if(frame_count >= window_size) {
                SlidingWindow();
            }
            else {
                ++frame_count;
            }

            if(draw_pose) {
                draw_pose(frame->id, frame->timestamp, Sophus::SE3d(frame->q_wb, frame->p_wb));
            }

            if(draw_mps) {
                Eigen::VecVector3d mps;

                for(auto& it : m_features) {
                    auto& feat = it.second;
                    if(feat.inv_depth == -1)
                        continue;
                    Sophus::SE3d Twb(d_frames[feat.start_id]->q_wb, d_frames[feat.start_id]->p_wb);
                    Sophus::SE3d Tbc(q_bc, p_bc);
                    Eigen::Vector3d x3Dc = feat.inv_depth * feat.pt_n_per_frame[0];
                    Eigen::Vector3d x3Dw = Twb * Tbc * x3Dc;
                    mps.emplace_back(x3Dw);
                }

                draw_mps(mps);
            }
        }
        else { // TIGHTLY
            // TODO
        }
    }
}

BackEnd::MarginType BackEnd::AddFeaturesCheckParallax(FramePtr frame) {
    uint last_track_num = 0;
    for(int i = 0; i < frame->pt_id.size(); ++i) {
        auto it = m_features.find(frame->pt_id[i]);
        Feature* feat = nullptr;
        if(it == m_features.end()) {
            auto result = m_features.emplace(std::make_pair(frame->pt_id[i], Feature(frame->pt_id[i], frame_count)));
            if(!result.second)
                throw std::runtime_error("m_features insert fail?");
            feat = &result.first->second;
        }
        else {
            ++last_track_num;
            feat = &it->second;
        }

        feat->pt_n_per_frame.emplace_back(frame->pt_normal_plane[i]);
        feat->pt_r_n_per_frame.emplace_back(frame->pt_r_normal_plane[i]);
        if(frame->pt_r_normal_plane[i](0) == -1)
            feat->num_meas += 1;
        else
            feat->num_meas += 2;
    }

    if(frame_count < 2 || last_track_num < 20) {
        return MARGIN_OLD;
    }

    double parallax_sum = 0.0f;
    uint parallax_num = 0;

    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.start_id <= frame_count - 2 &&
           feat.start_id + feat.pt_n_per_frame.size() - 1 >= frame_count - 1) {
            size_t idx_i = frame_count - 2 - feat.start_id;
            size_t idx_j = frame_count - 1 - feat.start_id;

            parallax_sum += (feat.pt_n_per_frame[idx_i] - feat.pt_n_per_frame[idx_j]).norm();
            parallax_num++;
        }
    }

    if(parallax_num == 0) {
        return MARGIN_OLD;
    }
    else {
        return (parallax_sum / parallax_num >= min_parallax) ? MARGIN_OLD : MARGIN_SECOND_NEW;
    }
}

void BackEnd::SlidingWindow() {
    if(marginalization_flag == MARGIN_OLD) {
        // margin out old
        for(auto it = m_features.begin(); it != m_features.end();) {
            auto& feat = it->second;
            if(feat.start_id == 0) {
                // change parent
                if(feat.pt_r_n_per_frame[0](0) == -1)
                    feat.num_meas -= 1;
                else
                    feat.num_meas -= 2;

                feat.pt_n_per_frame.pop_front();
                feat.pt_r_n_per_frame.pop_front();

                if(feat.num_meas < 2) {
                    it = m_features.erase(it);
                }
                else
                    ++it;
            }
            else {
                feat.start_id--;
                ++it;
            }
        }

        d_frames.pop_front();
    }
    else if(marginalization_flag == MARGIN_SECOND_NEW) {
        for(auto it = m_features.begin(); it != m_features.end();) {
            auto& feat = it->second;

            if(feat.start_id == frame_count) {
                feat.start_id--;
            }
            else {
                int j = frame_count - 1 - feat.start_id;

                if(feat.pt_n_per_frame.size() <= j) {
                    ++it;
                    continue;
                }

                if(feat.pt_r_n_per_frame[j](0) == -1) {
                    feat.num_meas -= 1;
                }
                else {
                    feat.num_meas -= 2;
                }

                feat.pt_n_per_frame.erase(feat.pt_n_per_frame.begin() + j);
                feat.pt_r_n_per_frame.erase(feat.pt_r_n_per_frame.begin() + j);

                if(feat.pt_n_per_frame.empty()) {
                    it = m_features.erase(it);
                    continue;
                }
            }
            ++it;
        }
        d_frames.erase(d_frames.end() - 2);
    }
    else
        throw std::runtime_error("marginalization flag error!");
}

uint BackEnd::Triangulate() {
    uint num_triangulate = 0;
    for(auto& it : m_features) {
        auto& feat = it.second;

        if(feat.num_meas < 2)
            continue;

        if(feat.inv_depth > 0)
            continue;

        Eigen::MatrixXd A(2 * feat.num_meas, 4);
        Sophus::SE3d Tw0(d_frames[feat.start_id]->q_wb, d_frames[feat.start_id]->p_wb);

        int A_idx = 0;
        for(int i = 0, n = feat.pt_n_per_frame.size(); i < n; ++i) {
            int idx_i = feat.start_id + i;
            Sophus::SE3d Twi(d_frames[idx_i]->q_wb, d_frames[idx_i]->p_wb);
            Sophus::SE3d Ti0 = Twi.inverse() * Tw0;
            Eigen::Matrix<double, 3, 4> P;
            P << Ti0.rotationMatrix(), Ti0.translation();
            A.row(A_idx++) = P.row(0) - feat.pt_n_per_frame[i](0) * P.row(2);
            A.row(A_idx++) = P.row(1) - feat.pt_n_per_frame[i](1) * P.row(2);

            if(feat.pt_r_n_per_frame[i](0) != -1) {
                Sophus::SE3d Trl(q_rl, p_rl);
                Sophus::SE3d Tri_0 = Trl * Ti0;
                P << Tri_0.rotationMatrix(), Tri_0.translation();
                A.row(A_idx++) = P.row(0) - feat.pt_r_n_per_frame[i](0) * P.row(2);
                A.row(A_idx++) = P.row(1) - feat.pt_r_n_per_frame[i](1) * P.row(2);
            }
        }

        // Solve AX = 0
        Eigen::Vector4d X = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        X /= X(3);

        if(X(2) < 0.1)
            continue;

        feat.inv_depth = 1.0 / X(2);
        ++num_triangulate;
    }
    return num_triangulate;
}

void BackEnd::Reset() {
    LOG(WARNING) << "BackEnd Reset...";
    d_frames.clear();
    m_features.clear();
    frame_count = 0;
    next_frame_id = 0;
}

void BackEnd::data2double() {
    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        size_t idx_pose = i * 7;
        size_t idx_vb = i * 9;
        std::memcpy(para_pose + idx_pose, d_frames[i]->q_wb.data(), sizeof(double) * Sophus::SO3d::num_parameters);
        std::memcpy(para_pose + idx_pose + 4, d_frames[i]->p_wb.data(), sizeof(double) * 3);
        std::memcpy(para_speed_bias + idx_vb, d_frames[i]->v_wb.data(), sizeof(double) * 3);
        std::memcpy(para_speed_bias + idx_vb + 3, d_frames[i]->ba.data(), sizeof(double) * 3);
        std::memcpy(para_speed_bias + idx_vb + 6, d_frames[i]->bg.data(), sizeof(double) * 3);
    }

    // ensure the feature array size can be filled by track features.
    if(para_features_capacity <= m_features.size()) {
        para_features_capacity *= 2;
        delete [] para_features;
        para_features = new double[para_features_capacity * 1];
    }

    size_t num_mps = 0;
    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.num_meas < 2 || feat.inv_depth < 0) {
            continue;
        }
        para_features[num_mps++] = feat.inv_depth;
    }
}

void BackEnd::double2data() {
    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        size_t idx_pose = i * 7;
        size_t idx_vb = i * 9;
        std::memcpy(d_frames[i]->q_wb.data(), para_pose + idx_pose, sizeof(double) * Sophus::SO3d::num_parameters);
        std::memcpy(d_frames[i]->p_wb.data(), para_pose + idx_pose + 4, sizeof(double) * 3);
        std::memcpy(d_frames[i]->v_wb.data(), para_speed_bias + idx_vb , sizeof(double) * 3);
        std::memcpy(d_frames[i]->ba.data(), para_speed_bias + idx_vb + 3 , sizeof(double) * 3);
        std::memcpy(d_frames[i]->bg.data(), para_speed_bias + idx_vb + 6 , sizeof(double) * 3);
    }

    size_t num_mps = 0;
    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.num_meas < 2 || feat.inv_depth < 0) {
            continue;
        }
        feat.inv_depth = para_features[num_mps++];
    }
}

void BackEnd::SolveBA() {
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(5.991));
    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3();

    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        problem.AddParameterBlock(para_pose + i * 7, 7, local_para_se3);

        if(i == 0)
            problem.SetParameterBlockConstant(para_pose + i * 7);
    }

    size_t mp_idx = 0;
    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.num_meas < 2 || feat.inv_depth < 0) {
            continue;
        }
        size_t id_i = feat.start_id;
        Eigen::Vector3d pt_i = feat.pt_n_per_frame[0];
        for(int j = 0, n = feat.pt_n_per_frame.size(); j < n; ++j) {
            size_t id_j = id_i + j;
            Eigen::Vector3d pt_j = feat.pt_n_per_frame[j];
            if(feat.pt_r_n_per_frame[j](0) != -1) {
                if(j == 0)
                    continue;
                auto factor = new ProjectionFactor(pt_i, pt_j, q_bc, p_bc, focal_length);
                problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_features + mp_idx);
            }
            else {
                Eigen::Vector3d pt_jr = feat.pt_r_n_per_frame[j];
                if(j == 0) {
                    auto factor = new SelfProjectionFactor(pt_j, pt_jr, q_rl, p_rl, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_features + mp_idx);
                }
                else {
                    auto factor = new SlaveProjectionFactor(pt_i, pt_jr, q_rl, p_rl, q_bc, p_bc, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_features + mp_idx);
                }
            }
        }
        ++mp_idx;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.num_threads = 4;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport();

    double2data();
}