//
// Created by martin on 02.03.18.
//

#ifndef THRAX_ADADELTAPARAMETERUPDATER_H
#define THRAX_ADADELTAPARAMETERUPDATER_H

#include <boost/property_tree/ptree.hpp>
#include <Eigen/Dense>

#include <thrax/struct/Gradient.h>
#include <unordered_map>

#include <thrax/initializer/ScalarInitializer.h>
#include "ParameterUpdater.h"

namespace pt = boost::property_tree;
using namespace Eigen;

class AdaDeltaParameterUpdater: public ParameterUpdater {
public:
    AdaDeltaParameterUpdater(pt::ptree& hyperParameters, const ParameterMap& parameters): ParameterUpdater(hyperParameters, parameters) {
        delta = hyperParameters.get<double>("delta", 1e-7);
        rho = hyperParameters.get<double>("rho");
        ScalarInitializer initializer(0.0);
        int m;
        int k;
        // initialize gradient accumulation data structures
        for (auto const& parameter: parameters) {
            m = parameter.second.getNumberOfEmbeddings();
            k = parameter.second.getEmbeddingDimension();
            accumulatedGradients.insert({parameter.first, EmbeddingParameterSet(k, m)});
            initializer.initialize(accumulatedGradients.at(parameter.first), parameter.first);
            accumulatedParameterUpdates.insert({parameter.first, EmbeddingParameterSet(k, m)});
            initializer.initialize(accumulatedParameterUpdates.at(parameter.first), parameter.first);
        }
    }

    virtual void calculateUpdate(GradientMap& gradients) {
        for (auto& gradient: gradients) {
            for (auto& ptr: gradient.second.getIdToCol()) {
                accumulatedGradients.at(gradient.first).col(ptr.first) = rho * accumulatedGradients.at(gradient.first).col(ptr.first) + (1 - rho) * gradient.second.get(ptr.first).array().square().matrix();
                VectorXd d = (gradient.second.get(ptr.first).array() * (accumulatedParameterUpdates.at(gradient.first).col(ptr.first).array() + delta).sqrt() / (accumulatedGradients.at(gradient.first).col(ptr.first).array() + delta).sqrt()).matrix();
                accumulatedParameterUpdates.at(gradient.first).col(ptr.first) = rho * accumulatedParameterUpdates.at(gradient.first).col(ptr.first) + (1 - rho) * d.array().square().matrix();
                gradient.second.setGradient(ptr.first, d);
            }
        }
    }

private:
    double rho; /** decay constant **/
    double delta; /** minimum value not zero, used to avoid zero division **/
    ParameterMap accumulatedGradients;
    ParameterMap accumulatedParameterUpdates;
};


#endif //THRAX_ADADELTAPARAMETERUPDATER_H
