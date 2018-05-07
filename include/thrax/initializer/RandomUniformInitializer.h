//
// Created by martin on 07.02.18.
//

#ifndef THRAX_RANDOMUNIFORMINITIALIZER_H
#define THRAX_RANDOMUNIFORMINITIALIZER_H

#include <thrax/initializer/Initializer.h>

/**
 * RandomUniformInitializer initializes the given parameters by drawing from a uniform distribution in intervall [low, high).
 */
class RandomUniformInitializer: public Initializer {
public:
    RandomUniformInitializer(double low, double high) : low(low), high(high) {}

    /**
     * Sets all values of the parameter set to <value>
     * @param parameterSet
     */
    virtual void initialize(EmbeddingParameterSet &parameterSet, const std::string name) override {
        int m = parameterSet.getNumberOfEmbeddings();
        int k = parameterSet.getEmbeddingDimension();
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; ++j) {
                parameterSet(i, j) = RandomUtil::uniformReal(low, high);
            }
        }
    }

private:
    double low;
    double high;
};


#endif //THRAX_RANDOMUNIFORMINITIALIZER_H
