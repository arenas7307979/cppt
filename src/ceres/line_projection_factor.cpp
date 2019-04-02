#include "line_projection_factor.h"

LineProjectionFactor::LineProjectionFactor(const Eigen::Vector3d& spt_, const Eigen::Vector3d& ept_, double focal_length)
    : spt(spt_), ept(ept_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
}

bool LineProjectionFactor::Evaluate(double const* const* parameters_raw, double* residuals_raw, double** jacobians_raw) const {
    // parameters [0]: Twb
    //            [1]: Tbc
    //            [2]: Lw
    Eigen::Map<const Sophus::SE3d> Twb(parameters_raw[0]), Tbc(parameters_raw[1]);
    Eigen::Map<const Sophus::SO3d> so3(parameters_raw[2]);
    Eigen::Map<const Sophus::SO2d> so2(parameters_raw[2] + 4);
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Eigen::Matrix3d U = so3.matrix();
    double w1 = so2.unit_complex()(0), w2 = so2.unit_complex()(1);
    Sophus::SE3d Tbw = Twb.inverse(), Tcb = Tbc.inverse();
    Sophus::SE3d Tcw = (Twb * Tbc).inverse();
    auto hat = Sophus::SO3d::hat;

    Eigen::Vector3d mw = w1 * U.col(0), lw = w2 * U.col(1),
            mb = Tbw.so3() * mw + hat(Tbw.translation()) * (Tbw.so3() * lw),
            lb = Tbw.so3() * lw,
            mc = Tcb.so3() * mb + hat(Tcb.translation()) * (Tcb.so3() * lb);

    Eigen::Vector3d l = mc / mc.head<2>().norm(); // 2d line in normal plane equal to Lc normal vector
    residuals << l.dot(spt),
            l.dot(ept);
    residuals = sqrt_info * residuals;

    Eigen::Matrix<double, 6, 6> Tcw_plucker;
    Tcw_plucker.block<3,3>(0,0) = Tcw.so3().matrix();
    Tcw_plucker.block<3,3>(3,3) = Tcw.so3().matrix();
    Tcw_plucker.block<3,3>(0,3) = Sophus::SO3d::hat(Tcw.translation()) * (Tcw.so3().matrix());

    Eigen::Matrix<double, 6, 6> Tcb_plucker;
    Tcb_plucker.block<3,3>(0,0) = Tcb.so3().matrix();
    Tcb_plucker.block<3,3>(3,3) = Tcb.so3().matrix();
    Tcb_plucker.block<3,3>(0,3) = Sophus::SO3d::hat(Tcb.translation()) * (Tcb.so3().matrix());


    if(jacobians_raw) {
        Eigen::Matrix<double, 2, 3> reduce;
        double m01 = mc.head<2>().norm();
        double inv_m01 = 1.0f / m01, inv_m01_2 = inv_m01 * inv_m01, inv_m01_3 = inv_m01 * inv_m01_2;
        double mc_dot_spt = mc.dot(spt), mc_dot_ept = mc.dot(ept);
        reduce << spt(0)*inv_m01-mc(0)*mc_dot_spt*inv_m01_3, spt(1)*inv_m01-mc(1)*mc_dot_spt*inv_m01_3, inv_m01,
                ept(0)*inv_m01-mc(0)*mc_dot_ept*inv_m01_3, ept(1)*inv_m01-mc(1)*mc_dot_ept*inv_m01_3, inv_m01;
        reduce = sqrt_info * reduce;

        Sophus::SO3d q_cb = Tcb.so3(), q_bw = Tbw.so3(), q_cw = q_cb * q_bw;

        Eigen::Matrix3d Rcb = q_cb.matrix(),
                Rbw = q_bw.matrix(),
                Rcw = q_cw.matrix();

        const Eigen::Vector3d &p_bc = Tbc.translation(), &p_wb = Twb.translation();

        Eigen::Matrix3d lw_hat = hat(lw),
                mb_hat = hat(mb), lb_hat = hat(lb),
                mc_hat = hat(mc),
                tbc_hat = hat(p_bc), twb_hat = hat(p_wb);

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jwb(jacobians_raw[0]);
            Eigen::Matrix<double, 3, 6> J;
            Eigen::Matrix<double, 6, 3> dLb_pwb;
            dLb_pwb.setZero();
            dLb_pwb.topRows(3) =  Rbw * lw_hat;
            //Eigen::Matrix<double, 3, 3> K_dLc_pwb = Sophus::Upper_Transformation(Tcb, dLb_pwb);
            Eigen::Matrix<double, 3, 3> K_dLc_pwb = (Tcb_plucker * dLb_pwb).topRows(3);
            Eigen::Matrix<double, 6, 3> dLb_qwb;
            dLb_qwb.topRows(3) = mb_hat;
            dLb_qwb.bottomRows(3) = Sophus::SO3d::hat(Rbw * lw);
            Eigen::Matrix<double, 3, 3> K_dLc_qwb = (Tcb_plucker * dLb_qwb).topRows(3);
            J.leftCols<3>() = K_dLc_pwb;
            J.rightCols<3>() = K_dLc_qwb;
            Jwb.leftCols<6>() = reduce * J;
            Jwb.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jbc(jacobians_raw[1]);
            Eigen::Matrix<double, 3, 6> J;
            J.leftCols<3>() = -Rcb * lb_hat;
            J.rightCols<3>() = mc_hat;
            Jbc.leftCols<6>() = reduce * J;
            Jbc.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> Jline(jacobians_raw[2]);
            Eigen::Vector3d O3x1 = Eigen::Vector3d::Zero();
            Eigen::Matrix<double, 3, 4> dmw_dTh, dlw_dTh;
            Eigen::Matrix<double, 6, 4> dLw_dTh ;
            dLw_dTh.setZero();
            dmw_dTh << O3x1, -w1*U.col(2), w1*U.col(1), -w2*U.col(0);
            dlw_dTh << w2*U.col(2),  O3x1, -w2*U.col(0), w1*U.col(1);
            dLw_dTh.topRows(3) =  dmw_dTh;
            dLw_dTh.bottomRows(3) = dlw_dTh;
            Eigen::Matrix<double, 3, 4> K_dLc_Ow = (Tcw_plucker * dLw_dTh).topRows(3);
            Jline.leftCols<4>() = reduce * K_dLc_Ow;
            //Jline.leftCols<4>() = reduce * Rcb * ((Rbw * dmw_dTh - Rbw * twb_hat * dlw_dTh) - tbc_hat * (Rbw * dlw_dTh));
            Jline.rightCols<2>().setZero();
        }
    }
    return true;
}

LineSlaveProjectionFactor::LineSlaveProjectionFactor(const Eigen::Vector3d& spt_, const Eigen::Vector3d& ept_, double focal_length)
    : spt(spt_), ept(ept_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
}

bool LineSlaveProjectionFactor::Evaluate(double const* const* parameters_raw, double* residuals_raw, double** jacobians_raw) const {
    // parameters [0]: Twb
    //            [1]: Tbc
    //            [2]: Tsm
    //            [3]: Lw
    Eigen::Map<const Sophus::SE3d> Twb(parameters_raw[0]), Tbc(parameters_raw[1]), Tsm(parameters_raw[2]);
    Eigen::Map<const Sophus::SO3d> so3(parameters_raw[3]);
    Eigen::Map<const Sophus::SO2d> so2(parameters_raw[3] + 4);
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Sophus::SE3d Tbw = Twb.inverse(), Tcb = Tbc.inverse();
    Eigen::Matrix3d U = so3.matrix();
    double w1 = so2.unit_complex()(0), w2 = so2.unit_complex()(1);
    auto hat = Sophus::SO3d::hat;

    Eigen::Vector3d mw = w1 * U.col(0), lw = w2 * U.col(1),
            mb = Tbw.so3() * mw + hat(Tbw.translation()) * (Tbw.so3() * lw),
            lb = Tbw.so3() * lw,
            mc = Tcb.so3() * mb + hat(Tcb.translation()) * (Tcb.so3() * lb),
            lc = Tcb.so3() * lb,
            ms = Tsm.so3() * mc + hat(Tsm.translation()) * (Tsm.so3() * lc);

    Sophus::SE3d Tsb = Tsm * Tcb;
    Eigen::Matrix<double, 6, 6> Tsb_plucker;
    Tsb_plucker.block<3,3>(0,0) = Tsb.so3().matrix();
    Tsb_plucker.block<3,3>(3,3) = Tsb.so3().matrix();
    Tsb_plucker.block<3,3>(0,3) = Sophus::SO3d::hat(Tsb.translation()) * (Tsb.so3().matrix());

    Sophus::SE3d Tcw = (Twb * Tbc).inverse();
    Sophus::SE3d Tsw = Tsm * Tcw;
    Eigen::Matrix<double, 6, 6> Tsw_plucker;
    Tsw_plucker.block<3,3>(0,0) = Tsw.so3().matrix();
    Tsw_plucker.block<3,3>(3,3) = Tsw.so3().matrix();
    Tsw_plucker.block<3,3>(0,3) = Sophus::SO3d::hat(Tsw.translation()) * (Tsw.so3().matrix());

    Eigen::Matrix<double, 6, 6> Tsm_plucker;
    Tsm_plucker.block<3,3>(0,0) = Tsm.so3().matrix();
    Tsm_plucker.block<3,3>(3,3) = Tsm.so3().matrix();
    Tsm_plucker.block<3,3>(0,3) = Sophus::SO3d::hat(Tsm.translation()) * (Tsm.so3().matrix());


    Eigen::Vector3d l = ms / ms.head<2>().norm(); // 2d line in normal plane equal to Lc normal vector

    residuals << l.dot(spt),
            l.dot(ept);

    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix<double, 2, 3> reduce;
        double m01 = ms.head<2>().norm();
        double inv_m01 = 1.0f / m01, inv_m01_2 = inv_m01 * inv_m01, inv_m01_3 = inv_m01 * inv_m01_2;
        double mc_dot_spt = ms.dot(spt), mc_dot_ept = ms.dot(ept);
        reduce << spt(0)*inv_m01-ms(0)*mc_dot_spt*inv_m01_3, spt(1)*inv_m01-ms(1)*mc_dot_spt*inv_m01_3, inv_m01,
                ept(0)*inv_m01-ms(0)*mc_dot_ept*inv_m01_3, ept(1)*inv_m01-ms(1)*mc_dot_ept*inv_m01_3, inv_m01;
        reduce = sqrt_info * reduce;

        Sophus::SO3d q_sm = Tsm.so3(), q_cb = Tbc.so3().inverse(), q_bw = Twb.so3().inverse(),
                q_sb = q_sm * q_cb,
                q_sw = q_sb * q_bw;

        Eigen::Matrix3d Rsm = q_sm.matrix(), Rsb = q_sb.matrix(), Rsw = q_sw.matrix(), Rbw = q_bw.matrix();

        const Eigen::Vector3d &p_sm = Tsm.translation(),
                &p_bc = Tbc.translation(),
                &p_wb = Twb.translation();

        auto hat = Sophus::SO3d::hat;
        Eigen::Matrix3d lw_hat = hat(lw),
                mb_hat = hat(mb), lb_hat = hat(lb),
                mc_hat = hat(mc), lc_hat = hat(lc),
                twb_hat = hat(p_wb), tbc_hat = hat(p_bc), tsm_hat = hat(p_sm);



        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jwb(jacobians_raw[0]);
            Eigen::Matrix<double, 3, 6> J;
            Eigen::Matrix<double, 6, 3> dLb_pwb;
            dLb_pwb.setZero();
            dLb_pwb.topRows(3) =  Rbw * lw_hat;
            Eigen::Matrix<double, 3, 3> K_dLs_pwb = (Tsb_plucker * dLb_pwb).topRows(3);
            Eigen::Matrix<double, 6, 3> dLb_qwb;
            dLb_qwb.topRows(3) = mb_hat;
            dLb_qwb.bottomRows(3) = Sophus::SO3d::hat(Rbw * lw);
            Eigen::Matrix<double, 3, 3> K_dLs_qwb = (Tsb_plucker * dLb_qwb).topRows(3);
            J.leftCols<3>() = K_dLs_pwb;
            J.rightCols<3>() = K_dLs_qwb;
            //J.leftCols<3>() = Rsw * lw_hat;
            //J.rightCols<3>() = Rsb * (mb_hat - tbc_hat * lb_hat) + tsm_hat * Rsb * lb_hat;
            Jwb.leftCols<6>() = reduce * J;
            Jwb.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jbc(jacobians_raw[1]);
            Eigen::Matrix<double, 3, 6> J;

            Eigen::Matrix<double, 6, 3> dLc_pbc;
            dLc_pbc.setZero();
            dLc_pbc.topRows(3) =  q_cb.matrix() * lb_hat;
            Eigen::Matrix<double, 3, 3> K_dLs_pbc = (Tsm_plucker * dLc_pbc).topRows(3);

            Eigen::Matrix<double, 6, 3> dLc_qbc;
            dLc_qbc.topRows(3) = mc_hat;
            dLc_qbc.bottomRows(3) = lc_hat;
            Eigen::Matrix<double, 3, 3> K_dLs_qbc = (Tsm_plucker * dLc_qbc).topRows(3);

            //J.leftCols<3>() = Rsb * lb_hat;
            //J.rightCols<3>() = Rsm * mc_hat + tsm_hat * Rsm * lc_hat;
            J.leftCols<3>() = K_dLs_pbc;
            J.rightCols<3>() = K_dLs_qbc;
            Jbc.leftCols<6>() = reduce * J;
            Jbc.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jsm(jacobians_raw[2]);
            Eigen::Matrix<double, 3, 6> J;
            J.leftCols<3>() = -hat(q_sm * lc);
            J.rightCols<3>() = -(Rsm * mc_hat + tsm_hat * Rsm * lc_hat);
            Jsm.leftCols<6>() = reduce * J;
            Jsm.rightCols<1>().setZero();
        }

        if(jacobians_raw[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> Jline(jacobians_raw[3]);
            Eigen::Vector3d O3x1 = Eigen::Vector3d::Zero();
            Eigen::Matrix<double, 3, 4> dmw_dTh, dlw_dTh;
            Eigen::Matrix<double, 6, 4> dLw_dTh ;
            dLw_dTh.setZero();
            dmw_dTh << O3x1, -w1*U.col(2), w1*U.col(1), -w2*U.col(0);
            dlw_dTh << w2*U.col(2),  O3x1, -w2*U.col(0), w1*U.col(1);
            dLw_dTh.topRows(3) =  dmw_dTh;
            dLw_dTh.bottomRows(3) = dlw_dTh;
            Eigen::Matrix<double, 3, 4> K_dLs_Ow = (Tsw_plucker * dLw_dTh).topRows(3);
            Jline.leftCols<4>() = reduce * K_dLs_Ow;
            //Jline.leftCols<4>() = reduce * (Rsw * dmw_dTh + (tsm_hat * Rsw - Rsb * tbc_hat * Rbw - Rsw * twb_hat) * dlw_dTh);
            Jline.rightCols<2>().setZero();
        }
    }
    return true;
}
