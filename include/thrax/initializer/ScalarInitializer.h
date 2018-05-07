//
// Created by martin on 07.02.18.
//

#ifndef THRAX_SCALARINITIALIZER_H
#define THRAX_SCALARINITIALIZER_H

#include <thrax/initializer/Initializer.h>

/**
 * ScalarInitializer initializes the given parameters to a fixed double value.
 */
class ScalarInitializer: public Initializer {
public:
    ScalarInitializer(double value) : value(value) {}

    /**
     * Sets all values of the parameter set to <value>
     * @param parameterSet
     */
    virtual void initialize(EmbeddingParameterSet &parameterSet, const std::string name) override {
        int m = parameterSet.getNumberOfEmbeddings();
        int k = parameterSet.getEmbeddingDimension();
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; ++j) {
                parameterSet(i, j) = value;
            }
        }
    }

private:
    double value;
};


#endif //THRAX_SCALARINITIALIZER_H
