//
// Created by martin on 07.02.18.
//

#ifndef THRAX_RANDOMNORMALINITIALIZER_H
#define THRAX_RANDOMNORMALINITIALIZER_H

#include <thrax/initializer/Initializer.h>

/**
 * RandomNormalInitializer initializes the given parameters by drawing from a normal distribution with mean and var.
 */
class RandomNormalInitializer: public Initializer {
public:
    RandomNormalInitializer(double mean, double var) : mean(mean), var(var) {}

    /**
     * Sets all values of the parameter set to <value>
     * @param parameterSet
     */
    virtual void initialize(EmbeddingParameterSet &parameterSet, const std::string name) override {
        int m = parameterSet.getNumberOfEmbeddings();
        int k = parameterSet.getEmbeddingDimension();
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; ++j) {
                parameterSet(i, j) = RandomUtil::normalReal(mean, var);
            }
        }
    }

private:
    double mean;
    double var;
};


#endif //THRAX_RANDOMNORMALINITIALIZER_H
