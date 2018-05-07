//
// Created by martin on 21.02.18.
//

#ifndef THRAX_MATHUTIL_H
#define THRAX_MATHUTIL_H

#include <Eigen/Dense>

using namespace Eigen;

class MathUtil {
public:
    static double logsumexp(VectorXd x) {
        double max = x.maxCoeff();
        double sum = 0;
        for (int i = 0; i < x.size(); ++i) {
            sum += exp(x[i] - max);
        }
        return std::log(sum) + max;
    }

    static double logSoftmax(double& positiveScore, VectorXd& negativeScores) {
        return positiveScore - MathUtil::logsumexp(negativeScores);
    }

    static double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    static double dot(VectorXd& a, VectorXd& b, VectorXd& c) {
        return (a.array()*b.array()*c.array()).sum();
    }

    static double softplus(double x) {
        return std::log(1 + exp(x));
    }
};


#endif //THRAX_MATHUTIL_H
