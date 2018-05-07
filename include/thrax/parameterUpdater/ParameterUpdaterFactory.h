//
// Created by martin on 02.03.18.
//

#ifndef THRAX_PARAMETERUPDATERFACTORY_H
#define THRAX_PARAMETERUPDATERFACTORY_H

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include <thrax/struct/Data.h>

#include "ParameterUpdater.h"
#include "SGDParameterUpdater.h"
#include "AdaDeltaParameterUpdater.h"
#include "AdaGradParameterUpdater.h"
#include "RMSPropParameterUpdater.h"

namespace pt = boost::property_tree;

class ParameterUpdaterFactory {
public:
    static ParameterUpdater* buildParameterUpdater(ParameterMap& parameterMap, pt::ptree& hyperParameters) {
        std::string updateStrategy = hyperParameters.get<std::string>("type", "sgd");
        boost::algorithm::to_lower(updateStrategy);
        ParameterUpdater* updater;
        if (updateStrategy == "sgd") {
            updater = new SGDParameterUpdater(hyperParameters, parameterMap);
        } else if (updateStrategy == "adagrad") {
            updater = new AdaGradParameterUpdater(hyperParameters, parameterMap);
        } else if (updateStrategy == "adadelta") {
            updater = new AdaDeltaParameterUpdater(hyperParameters, parameterMap);
        } else if (updateStrategy == "rmsprop") {
            updater = new RMSPropParameterUpdater(hyperParameters, parameterMap);
        } else {
            BOOST_LOG_TRIVIAL(error) << "Could not find update strategy " << updateStrategy;
        }
        BOOST_LOG_TRIVIAL(info) << "Using parameter updater " << updateStrategy;
        return updater;
    }
};


#endif //THRAX_PARAMETERUPDATERFACTORY_H
