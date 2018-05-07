//
// Created by martin on 09.01.18.
//

#ifndef THRAX_RANDOMUTIL_H
#define THRAX_RANDOMUTIL_H

#include <random>

namespace RandomUtil {
    std::random_device rd;
    std::mt19937 gen(rd());

    /**
     * Randomly samples from a real uniform distribution; low value inclusive, high value exclusive
     * @param low
     * @param high
     * @return random real number
     */
    inline double uniformReal(const double low = -0.1, const double high = 0.1) {
        std::uniform_real_distribution<double> distribution(low, high);
        return distribution(gen);
    }

    inline int uniformInt(const int low, const int high) {
        std::uniform_int_distribution<> distribution(low, high - 1);
        return distribution(gen);
    }

    inline double normalReal(const double mean, const double var) {
        std::normal_distribution<double> distribution(mean, var);
        return distribution(gen);
    }
}

#endif //THRAX_RANDOMUTIL_H
