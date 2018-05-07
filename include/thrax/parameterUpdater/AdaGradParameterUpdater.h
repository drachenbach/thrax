//
// Created by martin on 13.02.18.
//

#ifndef THRAX_ADAGRADPARAMETERUPDATE_H
#define THRAX_ADAGRADPARAMETERUPDATE_H

#include <string>
#include <boost/property_tree/ptree.hpp>
#include <Eigen/Dense>

#include <thrax/struct/Gradient.h>
#include <unordered_map>

#include <thrax/initializer/ScalarInitializer.h>
#include "ParameterUpdater.h"

namespace pt = boost::property_tree;
using namespace Eigen;

class AdaGradParameterUpdater: public ParameterUpdater {
public:
    AdaGradParameterUpdater(pt::ptree& hyperParameters, const ParameterMap& parameters): ParameterUpdater(hyperParameters, parameters) {
        delta = hyperParameters.get<double>("delta", 1e-7);
        alpha = hyperParameters.get<double>("alpha");
        ScalarInitializer initializer(0.0);
        int m;
        int k;
        // initialize gradient accumulation data structures
        for (auto const& parameter: parameters) {
            m = parameter.second.getNumberOfEmbeddings();
            k = parameter.second.getEmbeddingDimension();
            accumulatedGradients.insert({parameter.first, EmbeddingParameterSet(k, m)});
            initializer.initialize(accumulatedGradients.at(parameter.first), parameter.first);
        }
    }

    virtual void calculateUpdate(GradientMap& gradients) {
        for (auto& gradient: gradients) {
            for (auto& ptr: gradient.second.getIdToCol()) {
                accumulatedGradients.at(gradient.first).col(ptr.first) += gradient.second.get(ptr.first).array().square().matrix();
                gradient.second.setGradient(ptr.first, (alpha * gradient.second.get(ptr.first).array() / (delta + accumulatedGradients.at(gradient.first).col(ptr.first).array().sqrt())).matrix());
            }
        }
    }

private:
    double alpha; /** step size **/
    double delta; /** minimum value not zero, used to avoid zero division **/
    ParameterMap accumulatedGradients;
};


#endif //THRAX_ADAGRADPARAMETERUPDATE_H
