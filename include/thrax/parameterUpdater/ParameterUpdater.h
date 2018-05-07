//
// Created by martin on 10.01.18.
//

#ifndef THRAX_PARAMETERUPDATER_H
#define THRAX_PARAMETERUPDATER_H

#include <string>
#include <thrax/struct/Gradient.h>
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <thrax/util/Typedefs.h>

namespace pt = boost::property_tree;

class ParameterUpdater {
public:
    ParameterUpdater(pt::ptree& hyperParameters, const ParameterMap& parameters): hyperParameters(hyperParameters) {
    }

    virtual void calculateUpdate(GradientMap& gradients) { }

protected:
    pt::ptree hyperParameters;
};


#endif //THRAX_PARAMETERUPDATER_H
