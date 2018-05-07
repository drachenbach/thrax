//
// Created by martin on 07.02.18.
//

#ifndef THRAX_XAVIERINITIALIZER_H
#define THRAX_XAVIERINITIALIZER_H

#include <thrax/initializer/Initializer.h>

class XavierInitializer: public Initializer {

    virtual void initialize(EmbeddingParameterSet &parameterSet, const std::string name) override {
        int m = parameterSet.getNumberOfEmbeddings();
        int k = parameterSet.getEmbeddingDimension();
        double high = 6 / sqrt(k);
        double low = -high;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; ++j) {
                parameterSet(i, j) = RandomUtil::uniformReal(low, high);
            }
        }
    }
};


#endif //THRAX_XAVIERINITIALIZER_H
