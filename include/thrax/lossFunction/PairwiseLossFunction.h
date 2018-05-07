//
// Created by martin on 09.03.18.
//

#ifndef THRAX_PAIRWISELOSSFUNCTION_H
#define THRAX_PAIRWISELOSSFUNCTION_H

#include <boost/property_tree/ptree.hpp>

#include "LossFunction.h"

namespace pt = boost::property_tree;

class PairwiseLossFunction: public LossFunction {
private:
    double margin;

    double pairwiseLoss(Triple& positive, Triple& negative) {
        return std::max(model->score(negative) - model->score(positive) + margin, 0.0);
    }

    void pairwiseGradient(Triple& positive, Triple& negative, double& loss) {
        // only calculate gradient for non-negative losses
        if (pairwiseLoss(positive, negative) > 0) {
            loss += 1.0;
            model->gradient(positive, -1.0/numberOfTriplesInBatch);
            model->gradient(negative, 1.0/numberOfTriplesInBatch);
        }
    }

    void init() {
        margin = config.get<double>("model.hyperParameters.margin");
        numberOfTriplesInBatch = batchSize * numberOfNegatives;
    }

public:
    PairwiseLossFunction(AbstractModel *model, pt::ptree& config) : LossFunction(model, config) {
        init();
    }

    virtual void gradient(std::vector<Triple*> positives, std::vector<std::vector<Triple> >& negatives) {
        for (int i = 0; i < positives.size(); ++i) {
            for (int j = 0; j < negatives[i].size(); ++j) {
                pairwiseGradient(*positives[i], negatives[i][j], loss);
            }
        }
        model->l2(1.0/numberOfTriplesInBatch);
        numberOfGradientComputations += numberOfTriplesInBatch;
    }

    virtual void printLoss() {
        BOOST_LOG_TRIVIAL(info) << "Violations: " << loss << " (" << 100.0 * loss / numberOfGradientComputations << "%)";
    }
};


#endif //THRAX_PAIRWISELOSSFUNCTION_H
