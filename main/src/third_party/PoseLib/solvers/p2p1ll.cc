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

#include "p2p1ll.h"

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/univariate.h"

namespace poselib {

int p2p1ll(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
           const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output) {
    output->clear();
    output->reserve(2);
    const Eigen::Vector3d n = l[0].normalized();
    // Same centered differences as the E3Q3 solver, before its lxp1 scaling.
    const Eigen::Vector3d dX21 = Xp[1] - Xp[0];
    const double dX21_norm = dX21.norm();
    const Eigen::Vector3d a = dX21 / dX21_norm;
    const Eigen::Vector3d dX01 = X[0] - Xp[0];
    // Incidence coefficients use the normalized line normal n.
    const double lxp1 = n.dot(xp[0]), lxp2 = n.dot(xp[1]);
    if (dX21_norm == 0.0 || V[0].squaredNorm() == 0.0)
        return 0;

    // In the paper's special frame, r is the first column and s the second
    // row of R. Work in the original frames: r = R*a and s = R^T*n.
    // Write s in V's perpendicular plane, and r in the span of the two rays.
    // Incidence gives two linear constraints on the two depths and s.
    const Eigen::Matrix<double, 3, 2> basis = perpendicular(V[0]);
    Eigen::Matrix<double, 2, 4> A;
    A << -lxp1, lxp2, -a.dot(basis.col(0)), -a.dot(basis.col(1)), lxp1, 0.0, dX01.dot(basis.col(0)) / dX21_norm,
        dX01.dot(basis.col(1)) / dX21_norm;
    // Pivot on the largest minor, so points on the image line do not force
    // division by zero. The remaining two variables parameterize the kernel.
    int p = 0, q = 1;
    double det = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            const double minor = A(0, i) * A(1, j) - A(0, j) * A(1, i);
            if (std::abs(minor) > std::abs(det)) {
                det = minor;
                p = i;
                q = j;
            }
        }
    }
    if (det == 0.0)
        return 0;
    Eigen::Matrix<double, 4, 2> N = Eigen::Matrix<double, 4, 2>::Zero();
    int col = 0;
    for (int k = 0; k < 4; ++k) {
        if (k == p || k == q)
            continue;
        N(k, col) = 1.0;
        N(p, col) = (A(0, q) * A(1, k) - A(1, q) * A(0, k)) / det;
        N(q, col) = (A(1, p) * A(0, k) - A(0, p) * A(1, k)) / det;
        ++col;
    }
    const Eigen::Matrix<double, 3, 2> C = xp[1] * N.row(1) - xp[0] * N.row(0);
    const Eigen::Matrix<double, 3, 2> S = basis * N.bottomRows<2>();
    // Equal unit norms leave a homogeneous quadratic.
    const Eigen::Matrix2d Q = C.transpose() * C - S.transpose() * S;
    Eigen::Vector2d roots[2];
    const int count = univariate::solve_quadratic_real(Q(0, 0), 2.0 * Q(0, 1), Q(1, 1), roots);
    for (int i = 0; i < count; ++i) {
        const double depth1 = N.row(0).dot(roots[i]);
        const double depth2 = N.row(1).dot(roots[i]);
        // Both depths share a positive scale and change sign together.
        if (!std::isfinite(depth1) || !std::isfinite(depth2) || depth1 == 0.0 || depth2 == 0.0 ||
            (depth1 > 0.0) != (depth2 > 0.0))
            continue;
        const double scale = 1.0 / (S * roots[i]).norm();
        const double lambda = dX21_norm * scale * depth1;
        if (!std::isfinite(lambda) || lambda == 0.0)
            continue;
        const Eigen::Vector3d s = scale * S * roots[i];
        const Eigen::Vector3d r = (C * roots[i]).normalized();
        const Eigen::Vector3d w = s.cross(a);
        const double w2 = w.squaredNorm();
        if (w2 == 0.0)
            continue;
        const Eigen::Vector3d z = n.cross(r);
        const Eigen::Matrix3d F = r * a.transpose();
        const Eigen::Matrix3d G = (n - r.dot(n) * r) * (s - a.dot(s) * a).transpose() / w2;
        const Eigen::Matrix3d H = z * w.transpose() / w2;
        const Eigen::Matrix3d R = (lambda > 0.0 ? 1.0 : -1.0) * (F + G) + H;
        output->emplace_back(R, std::abs(lambda) * xp[0] - R * Xp[0]);
    }
    return output->size();
}

} // namespace poselib
