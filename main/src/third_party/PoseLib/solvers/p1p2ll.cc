// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "p1p2ll.h"

#include "PoseLib/misc/univariate.h"
#include "p3p_common.h"

namespace poselib {
namespace {
// Intersect two homogeneous conics using a degenerate member of their pencil.
// Only a cubic and two quadratics are solved; no quartic root isolation is needed.
int intersect_conics(Eigen::Matrix3d c, Eigen::Matrix3d d, Eigen::Vector3d roots[4]) {
    c /= c.norm();
    d /= d.norm();
    double a = d.determinant();
    double b = (adjugate(d).cwiseProduct(c)).sum();
    double e = c.determinant();
    double f = (adjugate(c).cwiseProduct(d)).sum();
    if (std::abs(a) < std::abs(e)) {
        std::swap(c, d);
        std::swap(a, e);
        std::swap(b, f);
    }
    double mu[3];
    int n;
    if (a == 0.0) {
        double rr[2];
        n = univariate::solve_quadratic_real(b, f, e, rr);
        for (int i = 0; i < n; ++i)
            mu[i] = rr[i];
    } else {
        n = univariate::solve_cubic_real(b / a, f / a, e / a, mu);
    }
    for (int i = 0; i < n; ++i) {
        Eigen::Matrix3d B = c + mu[i] * d;
        const auto lines = compute_pq(B, ConicFactorStrategy::MAX_ABS);
        if (!lines[0].allFinite() || !lines[1].allFinite() || lines[0].squaredNorm() == 0.0 ||
            lines[1].squaredNorm() == 0.0)
            continue;
        int count = 0;
        for (const Eigen::Vector3d &line : lines) {
            const Eigen::Matrix<double, 3, 2> basis = perpendicular(line);
            const Eigen::Matrix2d q = basis.transpose() * c * basis;
            Eigen::Vector2d uv[2];
            const int m = univariate::solve_quadratic_real(q(0, 0), 2.0 * q(0, 1), q(1, 1), uv);
            for (int j = 0; j < m; ++j) {
                Eigen::Vector3d x = basis * uv[j];
                // Refine on the two original conics in a bounded affine chart.
                Eigen::Index fixed;
                x.cwiseAbs().maxCoeff(&fixed);
                const int p = (fixed + 1) % 3, q = (fixed + 2) % 3;
                for (int iter = 0; iter < 3; ++iter) {
                    const Eigen::Vector3d cx = c * x, dx = d * x;
                    const double r = x.dot(cx), t = x.dot(dx);
                    const double det = 2.0 * (cx(p) * dx(q) - cx(q) * dx(p));
                    if (det == 0.0)
                        break;
                    x(p) -= (r * dx(q) - t * cx(q)) / det;
                    x(q) -= (t * cx(p) - r * dx(p)) / det;
                }
                roots[count++] = x.normalized();
            }
        }
        return count;
    }
    return 0;
}

} // namespace

int p1p2ll(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
           const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output) {
    output->clear();
    output->reserve(4);
    const Eigen::Vector3d l1_normalized = l[0].normalized(), l2_normalized = l[1].normalized();
    const Eigen::Vector3d cross_n = l1_normalized.cross(l2_normalized);
    const double cross_n2 = cross_n.squaredNorm();
    // Coincident interpretation planes do not give an isolated minimal pose.
    // Allow for roundoff (including fused multiply-add) in the cross product.
    if (!(cross_n2 > 1e-30))
        return 0;
    // Choose the first line to give the better translation pivot.
    const int i = std::abs(l1_normalized.dot(xp[0])) >= std::abs(l2_normalized.dot(xp[0])) ? 0 : 1;
    const int j = 1 - i;
    Eigen::Matrix3d Rc;
    Rc.col(1) = i == 0 ? l1_normalized : l2_normalized;
    Rc.col(2) = (i == 0 ? 1.0 : -1.0) * cross_n / std::sqrt(cross_n2);
    Rc.col(0) = Rc.col(1).cross(Rc.col(2));
    // xp_rot and l2_rot are the observations in the Rc camera frame.
    const Eigen::Vector3d xp_rot = Rc.transpose() * xp[0];
    const Eigen::Vector3d l2_rot = Rc.transpose() * l[j];
    // Center at Xp[0], as in E3Q3; indices 1 and 2 follow the selected line order.
    const Eigen::Vector3d dX1 = X[i] - Xp[0], dX2 = X[j] - Xp[0];
    const Eigen::Vector3d V2 = V[j].normalized();
    const double k = -l2_rot(1) / l2_rot(0);
    const double h = xp_rot(0) / xp_rot(1) - k;
    if (xp_rot(1) == 0.0 || l2_rot(0) == 0.0 || V[i].squaredNorm() == 0.0)
        return 0;

    // Express the linear row constraints with orthonormal bases instead of division by
    // individual world coordinates: r2 = S*[u,v], r1 = C*[u,v] + w*g.
    const Eigen::Matrix<double, 3, 2> S = perpendicular(V[i]);
    const Eigen::Vector3d cross = V2.cross(dX2);
    const double cross2 = cross.squaredNorm();
    if (cross2 == 0.0)
        return 0;
    const Eigen::Vector3d g = cross / std::sqrt(cross2);
    const Eigen::Vector3d dual1 = dX2.cross(cross) / cross2;
    const Eigen::Vector3d dual2 = cross.cross(V2) / cross2;
    Eigen::Matrix<double, 3, 2> C;
    for (int col = 0; col < 2; ++col)
        C.col(col) = k * V2.dot(S.col(col)) * dual1 + (k * dX2.dot(S.col(col)) + h * dX1.dot(S.col(col))) * dual2;

    // Remove the common scale: |r1|^2=|r2|^2
    // and r1^T*r2=0. Their conic pencil provides the resolvent cubic directly.
    Eigen::Matrix3d c = Eigen::Matrix3d::Zero(), d = Eigen::Matrix3d::Zero();
    c.topLeftCorner<2, 2>() = C.transpose() * C - Eigen::Matrix2d::Identity();
    c(2, 2) = 1.0;
    const Eigen::Matrix2d CS = C.transpose() * S;
    d.topLeftCorner<2, 2>() = 0.5 * (CS + CS.transpose());
    d.block<2, 1>(0, 2) = 0.5 * S.transpose() * g;
    d.block<1, 2>(2, 0) = d.block<2, 1>(0, 2).transpose();
    Eigen::Vector3d roots[4];
    const int count = intersect_conics(c, d, roots);
    for (int root = 0; root < count; ++root) {
        const Eigen::Vector3d uvw = roots[root] / roots[root].head<2>().norm();
        const Eigen::Vector3d r2 = S * uvw.head<2>();
        const double lambda = -r2.dot(dX1) / xp_rot(1);
        if (!std::isfinite(lambda) || lambda == 0.0)
            continue;
        Eigen::Vector3d r1 = C * uvw.head<2>() + uvw(2) * g;
        r1 = (r1 - r1.dot(r2) * r2).normalized();
        const Eigen::Vector3d r3 = r1.cross(r2);
        const Eigen::Matrix3d A = Rc.col(0) * r1.transpose() + Rc.col(1) * r2.transpose();
        const Eigen::Matrix3d B = Rc.col(2) * r3.transpose();
        const Eigen::Matrix3d R = (lambda > 0.0 ? 1.0 : -1.0) * A + B;
        output->emplace_back(R, std::abs(lambda) * xp[0] - R * Xp[0]);
    }
    return output->size();
}

} // namespace poselib
