//1e-7
// Created by martin on 12.02.18.
//

#ifndef THRAX_SGDPARAMETERUPDATER_H
#define THRAX_SGDPARAMETERUPDATER_H

#include <string>
#include <boost/property_tree/ptree.hpp>
#include <thrax/struct/Gradient.h>
#include <unordered_map>
#include <thrax/util/Typedefs.h>

#include "ParameterUpdater.h"

namespace pt = boost::property_tree;

class SGDParameterUpdater: public ParameterUpdater {
private:
    double alpha;
public:
    SGDParameterUpdater(pt::ptree& hyperParameters, const ParameterMap& parameters): ParameterUpdater(hyperParameters, parameters) {
        alpha = hyperParameters.get<double>("alpha");
    }

    virtual void calculateUpdate(GradientMap& gradients) {
        for (auto& gradient: gradients) {
            gradient.second.setGradients(alpha * gradient.second.getGradients());
        }
    }
};


#endif //THRAX_SGDPARAMETERUPDATER_H
