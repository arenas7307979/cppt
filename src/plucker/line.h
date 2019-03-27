#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <iostream>
// ------------------------ref---------------------
// Plucker Coordinates for Lines in the Space
// Structure-From-Motion Using Lines: Representation, Triangulation and Bundle Adjustment

namespace Plucker {
template <class Scalar>
class Line3;
using Line3d = Line3<double>;
using Line3f = Line3<float>;

template <class Scalar>
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
template <class Scalar>
using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
};

namespace Eigen {
namespace internal {

template <class Scalar_>
struct traits<Plucker::Line3<Scalar_>> {
    using Scalar = Scalar_;
    using DirectionType = Plucker::Vector3<Scalar_>;
    using NormalType = Plucker::Vector3<Scalar_>;
};

template <class Scalar_>
struct traits<Map<Plucker::Line3<Scalar_>>> {
    using Scalar = Scalar_;
    using DirectionType = Eigen::Map<Plucker::Vector3<Scalar_>>;
    using NormalType = Eigen::Map<Plucker::Vector3<Scalar_>>;
};

template <class Scalar_>
struct traits<Map<const Plucker::Line3<Scalar_>>> {
    using Scalar = Scalar_;
    using DirectionType = Eigen::Map<const Plucker::Vector3<Scalar_>>;
    using NormalType = Eigen::Map<const Plucker::Vector3<Scalar_>>;
};
}
}

namespace Plucker {

enum LineInitMethod {
    POINT_DIR,
    PLUCKER_L_M,
    TWO_POINT,
    TWO_PLANE
};

enum LinesStatus {
    SKEW_LINES,
    PARALLEL_LINES,
    INTERSECT_LINES,
};

template <class Derived>
class Line3Base {
public:
    using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
    using DirectionType = typename Eigen::internal::traits<Derived>::DirectionType;
    using NormalType = typename Eigen::internal::traits<Derived>::NormalType;
    static int constexpr DoF = 4;
    static int constexpr num_parameters = 6;

    // copy assignment
    Line3Base& operator=(const Line3Base& rhs) = default;

    template<class OtherDerived>
    Line3Base& operator=(const Line3Base<OtherDerived>& rhs) {
        l() = rhs.l();
        m() = rhs.m();
        return *this;
    }

    // cast
    template <class NewScaleType>
    Line3<NewScaleType> cast() const {
        return Line3<NewScaleType>(l().template cast<NewScaleType>(), m().template cast<NewScaleType>(), PLUCKER_L_M);
    }

    // getter
    DirectionType& l() {
        return static_cast<Derived*>(this)->l();
    }

    const DirectionType& l() const {
        return static_cast<const Derived*>(this)->l();
    }

    NormalType& m() {
        return static_cast<Derived*>(this)->m();
    }

    const NormalType& m() const {
        return static_cast<const Derived*>(this)->m();
    }

    // method
    Plucker::Vector4<Scalar> Orthonormal() const {
        Eigen::Matrix<Scalar, 3, 2> C;
        C << m(), l();
        C.normalize();
        // C = U*Sigma (QR decomposition)
        Eigen::HouseholderQR<Eigen::Matrix<Scalar, 3, 2>> qr(C);
        Eigen::Matrix<Scalar, 3, 2> Sigma = qr.matrixQR().template triangularView<Eigen::Upper>(); // R
        Eigen::Matrix<Scalar, 3, 3> U = qr.householderQ();                                // Q

        Scalar theta = std::atan2(Sigma(1, 1), Sigma(0, 0));
        // theta_x theta_y theta_z from decomposition U
        Scalar theta_x = std::atan2(-U(1, 2), U(2, 2));
        Scalar theta_y = std::atan2(U(0, 2), U(2, 2) / cos(theta_x));
        Scalar theta_z = std::atan2(-U(0, 1), U(0, 0));
        Plucker::Vector4<Scalar> Theta(theta_x, theta_y, theta_z, theta);
        return Theta;
    }

    void FromOrthonormal(const Plucker::Vector4<Scalar>& Theta) {
        Scalar sx = sin(Theta(0)), sy = sin(Theta(1)), sz = sin(Theta(2)),
               cx = cos(Theta(0)), cy = cos(Theta(1)), cz = cos(Theta(2)),
               w1 = cos(Theta(3)), w2 = sin(Theta(3));
        Plucker::Vector3<Scalar> u1, u2, l , m;
        u1(0) =  cy * cz;
        u1(1) =  sx * sy * cz + cx * sz;
        u1(2) = -cx * sy * cz + sx * sz;

        u2(0) = -cy * sz;
        u2(1) = -sx * sy * sz + cx * cz;
        u2(2) =  cx * sy * sz + sx * cz;

        m = w1 * u1;
        l = w2 * u2;
        SetPlucker(l, m);
    }

    void SetPlucker(const Plucker::Vector3<Scalar>& l_, const Plucker::Vector3<Scalar>& m_) {
        Scalar l_norm = l_.norm();
        if(l_norm <= std::numeric_limits<double>::min())
            throw std::runtime_error("l close to zero vector!!!");
        l() = l_ / l_norm;
        m() = m_ / l_norm;
    }

    Scalar Distance() const {
        return m().norm();
    }

    Scalar Distance(const Plucker::Vector3<Scalar>& q) const {
        Plucker::Vector3<Scalar> mq = m() - q.cross(l());
        return mq.norm();
    }

    Plucker::Vector3<Scalar> ClosestPoint() const {
        // p_perpendicular
        return l().cross(m());
    }

    Plucker::Vector3<Scalar> ClosestPoint(const Plucker::Vector3<Scalar>& q) const {
        // q_perpendicular
        Plucker::Vector3<Scalar> mq = m() - q.cross(l());
        return q + l().cross(mq);
    }

    Line3<Scalar> operator*(const Plucker::Vector4<Scalar>& delta) {
        Plucker::Vector4<Scalar> Theta = Orthonormal();
        Eigen::Matrix<Scalar, 3, 3> U, dU, U_plus_dU;
        U =  Eigen::AngleAxis<Scalar>(Theta(0), Plucker::Vector3<Scalar>::UnitX()) *
             Eigen::AngleAxis<Scalar>(Theta(1), Plucker::Vector3<Scalar>::UnitY()) *
             Eigen::AngleAxis<Scalar>(Theta(2), Plucker::Vector3<Scalar>::UnitZ());
        dU = Eigen::AngleAxis<Scalar>(delta(0), Plucker::Vector3<Scalar>::UnitX()) *
             Eigen::AngleAxis<Scalar>(delta(1), Plucker::Vector3<Scalar>::UnitY()) *
             Eigen::AngleAxis<Scalar>(delta(2), Plucker::Vector3<Scalar>::UnitZ());
        U_plus_dU = U * dU;

        Scalar theta = Theta(3) + delta(3),
               theta_x = std::atan2(-U_plus_dU(1, 2), U_plus_dU(2, 2)),
               theta_y = std::atan2(U_plus_dU(0, 2), U_plus_dU(2, 2) / cos(theta_x)),
               theta_z = std::atan2(-U_plus_dU(0, 1), U_plus_dU(0, 0));
        Line3<Scalar> L_plus_delta;
        L_plus_delta.FromOrthonormal(Eigen::Vector4d(theta_x, theta_y, theta_z, theta));
        return L_plus_delta;
    }
};

template <class Derived>
std::ostream& operator<<(std::ostream& s, const Line3Base<Derived>& line) {
    s << "l(" << line.l()(0) << "," << line.l()(1) << "," << line.l()(2) <<
       ") m(" << line.m()(0) << "," << line.m()(1) << "," << line.m()(2) << ").";
    return s;
}

template <class Derived>
Line3<typename Line3Base<Derived>::Scalar> operator*(const Sophus::SE3<typename Line3Base<Derived>::Scalar>& T21, const Line3Base<Derived>& L1) {
    using Scalar = typename Line3Base<Derived>::Scalar;
    Plucker::Vector3<Scalar> l2, m2;
    m2 = T21.so3() * L1.m() + Sophus::SO3<Scalar>::hat(T21.translation()) * (T21.so3() * L1.l());
    l2 = T21.so3() * L1.l();
    Line3<Scalar> L2(l2, m2, PLUCKER_L_M);
    return L2;
}

template <class Scalar>
class Line3 : public Line3Base<Line3<Scalar>> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line3() : l_(Scalar(1), Scalar(0), Scalar(0)), m_(Scalar(0), Scalar(0), Scalar(0)) {}
    Line3(const Line3& line)
        : l_(line.l()), m_(line.m())
    {}

    template<class OtherDerived>
    Line3(const Line3Base<OtherDerived>& line)
        : l_(line.l()), m_(line.m())
    {}

    Line3(const Plucker::Vector3<Scalar>& a, const Plucker::Vector3<Scalar>& b, LineInitMethod method) {
        Plucker::Vector3<Scalar> l, m;
        if(method == PLUCKER_L_M) {
            l = a;
            m = b;
        }
        else if (method == POINT_DIR) {
            l = b;
            m = a.cross(b);
        }
        else if(method == TWO_POINT) {
            l = b - a;
            m = a.cross(b);
        }
        else
            throw std::runtime_error("Vector3d constructor only support PLUCKER_L_M, POINT_DIR and TWO_POINT");

        Scalar l_norm = l.norm();
        if(l_norm <= std::numeric_limits<Scalar>::min())
            throw std::runtime_error("l close to zero vector!!!");
        l_ = l / l_norm;
        m_ = m / l_norm;
    }

    Line3(const Plucker::Vector4<Scalar>& a, const Plucker::Vector4<Scalar>& b, LineInitMethod method) {
        Plucker::Vector3<Scalar> l, m;
        if(method == TWO_POINT) {
            l = a(3) * b.template head<3>() - b(3) * a.template head<3>();
            m = a.template head<3>().cross(b.template head<3>());
        }
        else if (method == TWO_PLANE) {
            l = a.template head<3>().cross(b.template head<3>());
            m = a(3) * b.template head<3>() - b(3) * a.template head<3>();
        }
        else
            throw std::runtime_error("Vector4d constructor only support TWO_POINT and TWO_PLANE");

        Scalar l_norm = l.norm();
        if(l_norm <= std::numeric_limits<Scalar>::min())
            throw std::runtime_error("l close to zero vector!!!");
        l_ = l / l_norm;
        m_ = m / l_norm;
    }

    Plucker::Vector3<Scalar>& l() {
        return l_;
    }

    const Plucker::Vector3<Scalar>& l() const {
        return l_;
    }

    Plucker::Vector3<Scalar>& m() {
        return m_;
    }

    const Plucker::Vector3<Scalar>& m() const {
        return m_;
    }

    Scalar* data() {
        return l_.data();
    }

    const Scalar* data() const {
        return l_.data();
    }

protected:
    Plucker::Vector3<Scalar> l_, m_;
};

}

namespace Eigen {

template <class Scalar>
class Map<Plucker::Line3<Scalar>> : public Plucker::Line3Base<Map<Plucker::Line3<Scalar>>>
{
public:
    using Base = Plucker::Line3Base<Map<Plucker::Line3<Scalar>>>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Map(Scalar* data_raw)
        : l_(data_raw), m_(data_raw + 3) {}

    // LCOV_EXCL_START
    EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
    // LCOV_EXCL_STOP

    Map<Plucker::Vector3<Scalar>>& l() {
        return l_;
    }

    const Map<Plucker::Vector3<Scalar>>& l() const {
        return l_;
    }

    Map<Plucker::Vector3<Scalar>>& m() {
        return m_;
    }

    const Map<Plucker::Vector3<Scalar>>& m() const {
        return m_;
    }

protected:
    Map<Plucker::Vector3<Scalar>> l_, m_;
};

template <class Scalar>
class Map<const Plucker::Line3<Scalar>> : public Plucker::Line3Base<Map<const Plucker::Line3<Scalar>>>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Map(const Scalar* data_raw)
        : l_(data_raw), m_(data_raw + 3) {}

    const Map<const Plucker::Vector3<Scalar>>& l() const {
        return l_;
    }

    const Map<const Plucker::Vector3<Scalar>>& m() const {
        return m_;
    }

protected:
    const Map<const Plucker::Vector3<Scalar>> l_, m_;
};

}

namespace Plucker {
// # Corollary 2 #
// Two lines L1 and L2 are co-planar if and only if the reciprocal product of their
// Plucker coordinates is zero.
// (1-8)
template <class Derived, class OtherDerived>
typename Line3Base<Derived>::Scalar ReciprocalProduct(const Line3Base<Derived>& L1, const Line3Base<OtherDerived>& L2) {
    // (l1, m1) * (l2, m2) = l1.dot(m2) + l2.dot(m1)
    return L1.l().dot(L2.m()) + L2.l().dot(L1.m());
}
//// # Theorem 1 #
//// (1-10)
template <class Derived, class OtherDerived>
typename Line3Base<Derived>::Scalar Distance(const Line3Base<Derived>& L1, const Line3Base<OtherDerived>& L2) {
    using Scalar = typename Line3Base<Derived>::Scalar;
    using DirectionType = typename Line3Base<Derived>::DirectionType;
    using NormalType = typename Line3Base<Derived>::NormalType;

    const DirectionType &l1 = L1.l(), &l2 = L2.l();
    const NormalType &m1 = L1.m(), &m2 = L2.m();
    Vector3<Scalar> l1xl2 = l1.cross(l2);
    Scalar norm_l1xl2 =l1xl2.norm();
    Scalar d = 0;
    if(norm_l1xl2 <= std::numeric_limits<Scalar>::min()) {
        Scalar s = l1.dot(l2);
        d = l1.cross(m1 - m2 / s).norm();
    }
    else {
        d = std::abs(ReciprocalProduct(L1, L2)) / norm_l1xl2;
    }
    return d;
}
//// # Theorem 4 #
//// (1-17), (1-18)
template <class Derived1, class Derived2, class Derived3>
bool CommonPerpendicular(const Line3Base<Derived1>& L1, const Line3Base<Derived2>& L2,
                         Line3Base<Derived3>& Lcp, LinesStatus* status = nullptr)
{
    using Scalar = typename Line3Base<Derived1>::Scalar;
    using DirectionType = typename Line3Base<Derived1>::DirectionType;
    using NormalType = typename Line3Base<Derived1>::NormalType;

    const DirectionType &l1 = L1.l(), &l2 = L2.l();
    const NormalType &m1 = L1.m(), &m2 = L2.m();
    Scalar L1_star_L2 = ReciprocalProduct(L1, L2);
    Vector3<Scalar> l1xl2 = l1.cross(l2);
    Scalar norm_l1xl2 = l1xl2.norm();
    if(std::abs(L1_star_L2) <= std::numeric_limits<Scalar>::min()) {
        if(norm_l1xl2 <= std::numeric_limits<Scalar>::min()) {
            if(status)
                *status = PARALLEL_LINES;
            return false;
        }
        else {
            if(status)
                *status = INTERSECT_LINES;
            Vector3<Scalar> l_cp, m_cp;
            l_cp = l1xl2;
            m_cp = m1.cross(l2) - m2.cross(l1);
            Lcp.SetPlucker(l_cp, m_cp);
            return true;
        }
    }
    else {
        if(status)
            *status = SKEW_LINES;
        Vector3<Scalar> l_cp, m_cp;
        l_cp = l1xl2;
        m_cp = m1.cross(l2) - m2.cross(l1) + ((L1_star_L2 * l1.dot(l2))/(norm_l1xl2 * norm_l1xl2)) * l1xl2;
        Lcp.SetPlucker(l_cp, m_cp);
        return true;
    }
}

template <class Derived, class OtherDerived>
bool Feet(const Line3Base<Derived>& L1, const Line3Base<OtherDerived>& L2,
          Vector3<typename Line3Base<Derived>::Scalar>& p1_star,
          Vector3<typename Line3Base<Derived>::Scalar>& p2_star, LinesStatus* status = nullptr)
{
    using Scalar = typename Line3Base<Derived>::Scalar;
    using DirectionType = typename Line3Base<Derived>::DirectionType;
    using NormalType = typename Line3Base<Derived>::NormalType;

    const DirectionType &l1 = L1.l(), &l2 = L2.l();
    const NormalType&m1 = L1.m(), &m2 = L2.m();
    Scalar L1_star_L2 = ReciprocalProduct(L1, L2);
    Vector3<Scalar> l1xl2 = l1.cross(l2);
    Scalar norm_l1xl2 = l1xl2.norm();

    if(std::abs(L1_star_L2) <= std::numeric_limits<Scalar>::min()) {
        if(norm_l1xl2 <= std::numeric_limits<Scalar>::min()) {
            if(status)
                *status = PARALLEL_LINES;
            return false;
        }
        else {
            if(status)
                *status = INTERSECT_LINES;
            Eigen::Matrix<Scalar, 3, 3> I3 = Eigen::Matrix<Scalar, 3, 3>::Identity();
            p1_star = ((m1.dot(l2)*I3 + l1*m2.transpose() - l2*m1.transpose()) * l1xl2)/(norm_l1xl2 * norm_l1xl2);
            p2_star = p1_star;
            return true;
        }
    }
    else {
        if(status)
            *status = SKEW_LINES;
        Scalar norm_l1xl2_2 = norm_l1xl2 * norm_l1xl2;
        p1_star = (-m1.cross(l2.cross(l1xl2)) + (m2.dot(l1xl2))*l1) / norm_l1xl2_2;
        p2_star = ( m2.cross(l1.cross(l1xl2)) - (m1.dot(l1xl2))*l2) / norm_l1xl2_2;
        return true;
    }
}
}
