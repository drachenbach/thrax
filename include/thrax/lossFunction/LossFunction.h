//
// Created by martin on 09.03.18.
//

#ifndef THRAX_LOSSFUNCTION_H
#define THRAX_LOSSFUNCTION_H

#include <boost/property_tree/ptree.hpp>

#include <thrax/model/BaseModel.h>

namespace pt = boost::property_tree;

class LossFunction {
protected:
    AbstractModel* model;
    pt::ptree& config;
    double loss;
    std::string type;
    double numberOfTriplesInBatch; /** counts the number of gradient computations per epoch **/
    double numberOfGradientComputations; /** number of gradient computations per epoch, used for loss logging **/
    int numberOfNegatives;
    int batchSize;

    void init() {
        type = config.get<std::string>("optimizer.type");
        batchSize = config.get<int>("optimizer.batchSize");
        numberOfNegatives = config.get<int>("optimizer.sampling.numberOfNegatives");
        reset();
    }

public:
    LossFunction(AbstractModel *model, pt::ptree& config) : model(model), config(config) {
        init();
    }

    virtual void gradient(std::vector<Triple*> positives, std::vector<std::vector<Triple> >& negatives) = 0;

    void reset() {
        loss = 0.0;
        numberOfGradientComputations = 0.0;
    }

    std::string getType() {
        return type;
    }

    double getLoss() {
        return loss;
    }

    virtual void printLoss() = 0;

};

#endif //THRAX_LOSSFUNCTION_H
