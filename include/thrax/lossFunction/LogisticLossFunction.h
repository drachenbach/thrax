//
// Created by martin on 09.03.18.
//

#ifndef THRAX_LOGISTICLOSSFUNCTION_H
#define THRAX_LOGISTICLOSSFUNCTION_H

#include <boost/property_tree/ptree.hpp>

#include "LossFunction.h"

namespace pt = boost::property_tree;

class LogisticLossFunction: public LossFunction {
private:
    void logisticGradient(Triple &positive, std::vector<Triple> &negatives, double &loss) {
        double positiveScore = model->score(positive);
        loss += MathUtil::softplus(-positiveScore); // TODO add regularization term
        double scale = - MathUtil::sigmoid(-positiveScore);
        model->gradient(positive, scale/numberOfTriplesInBatch);

        // negative triples
        double negativeScore;
        for (int i = 0; i < negatives.size(); ++i) {
            // calculate score
            negativeScore = model->score(negatives[i]);
            loss += MathUtil::softplus(negativeScore); // TODO add regularization term
            scale = MathUtil::sigmoid(negativeScore);
            model->gradient(negatives[i], scale/numberOfTriplesInBatch);
        }
    }

    void init() {
        numberOfTriplesInBatch = batchSize * (numberOfNegatives + 1);
    }

public:
    LogisticLossFunction(AbstractModel *model, pt::ptree& config) : LossFunction(model, config) {
        init();
    }

    virtual void gradient(std::vector<Triple*> positives, std::vector<std::vector<Triple> >& negatives) {
        for (int i = 0; i < positives.size(); ++i) {
            logisticGradient(*positives[i], negatives[i], loss);
        }
        model->l2(1.0/numberOfTriplesInBatch);
        numberOfGradientComputations += numberOfTriplesInBatch;
    }

    virtual void printLoss() {
        BOOST_LOG_TRIVIAL(info) << "Loss: " << loss / (double)numberOfGradientComputations;
    }
};


#endif //THRAX_LOGISTICLOSSFUNCTION_H
