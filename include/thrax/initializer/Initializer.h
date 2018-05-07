//
// Created by martin on 09.01.18.
//

#ifndef THRAX_INITIALIZER_H
#define THRAX_INITIALIZER_H


#include <thrax/util/RandomUtil.h>
#include <thrax/model/EmbeddingParameterSet.h>

class EmbeddingParameterSet;

/**
 * Superclass for all Initializers that are used to initialize parameters.
 */
class Initializer {
public:
    virtual void initialize(EmbeddingParameterSet &parameterSet, const std::string name) {};
};


#endif //THRAX_INITIALIZER_H
