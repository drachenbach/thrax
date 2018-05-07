//
// Created by martin on 09.03.18.
//

#ifndef THRAX_LOSSFUNCTIONFACTORY_H
#define THRAX_LOSSFUNCTIONFACTORY_H

#include <boost/property_tree/ptree.hpp>

#include "LossFunction.h"
#include "SoftmaxLossFunction.h"
#include "LogisticLossFunction.h"
#include "PairwiseLossFunction.h"

namespace pt = boost::property_tree;

class LossFunctionFactory {
public:
    static LossFunction* buildLossFunction(AbstractModel* model, pt::ptree& config) {
        LossFunction* lossFunction;
        std::string lossFunctionType = config.get<std::string>("optimizer.type", "pair");
        if (lossFunctionType == "pair") {
            lossFunction = new PairwiseLossFunction(model, config);
        } else if (lossFunctionType == "softmax") {
            lossFunction = new SoftmaxLossFunction(model, config);
        } else if (lossFunctionType == "logistic") {
            lossFunction = new LogisticLossFunction(model, config);
        } else {
            BOOST_LOG_TRIVIAL(error) << "Loss function " << lossFunctionType << " not implemented";
        }
        BOOST_LOG_TRIVIAL(info) << "Using loss function " << lossFunctionType;
        return lossFunction;
    }
};


#endif //THRAX_LOSSFUNCTIONFACTORY_H
