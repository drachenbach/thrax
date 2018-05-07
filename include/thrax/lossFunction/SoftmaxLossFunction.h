//
// Created by martin on 09.03.18.
//

#ifndef THRAX_SOFTMAXLOSSFUNCTION_H
#define THRAX_SOFTMAXLOSSFUNCTION_H

#include <boost/property_tree/ptree.hpp>

#include <thrax/util/MathUtil.h>
#include "LossFunction.h"

namespace pt = boost::property_tree;

class SoftmaxLossFunction: public LossFunction {
private:
    VectorXd negativeScores;

    double softmaxLoss(Triple &positive, std::vector<Triple> &negatives) {
        double positiveScore = model->score(positive);
        for (int i = 0; i < negatives.size(); ++i) {
            negativeScores[i] = model->score(negatives[i]);
        }
        return -MathUtil::logSoftmax(positiveScore, negativeScores);
    }

    void softmaxGradient(Triple& positive, std::vector<Triple>& negatives, double& loss) {
        loss += softmaxLoss(positive, negatives);
        // calculate model scores
        for (int i = 0; i < negatives.size(); ++i) {
            negativeScores[i] = model->score(negatives[i]);
        }
        // gradient of positive triple
        model->gradient(positive, -1/numberOfTriplesInBatch);

        // gradient of negative triples
        double negativeLoss;
        for (int i = 0; i < negatives.size(); ++i) {
            negativeLoss = exp(MathUtil::logSoftmax(negativeScores[i], negativeScores));
            model->gradient(negatives[i], negativeLoss/numberOfTriplesInBatch);
        }
    }

    void init() {
        numberOfTriplesInBatch = batchSize;
        negativeScores.resize(numberOfNegatives + 1);
    }

public:
    SoftmaxLossFunction(AbstractModel *model, pt::ptree& config) : LossFunction(model, config) {
        init();
    }

    virtual void gradient(std::vector<Triple*> positives, std::vector<std::vector<Triple> >& negatives) {
        for (int i = 0; i < positives.size(); ++i) {
            softmaxGradient(*positives[i], negatives[i], loss);
        }
        model->l2(1.0/numberOfTriplesInBatch);
        numberOfGradientComputations += numberOfTriplesInBatch;
    }

    virtual void printLoss() {
        BOOST_LOG_TRIVIAL(info) << "Loss: " << loss / (double)numberOfGradientComputations;
    }
};


#endif //THRAX_SOFTMAXLOSSFUNCTION_H
