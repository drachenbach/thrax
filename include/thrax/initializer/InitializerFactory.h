//
// Created by martin on 20.03.18.
//

#ifndef THRAX_INITIALIZERFACTORY_H
#define THRAX_INITIALIZERFACTORY_H

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <thrax/struct/Data.h>
#include "XavierInitializer.h"
#include "PreTrainedInitializer.h"
#include "RandomNormalInitializer.h"
#include "RandomUniformInitializer.h"

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

class InitializerFactory {
public:
    static Initializer* buildInitializer(pt::ptree& config) {
        std::string initializerType = config.get<std::string>("initializer.type");
        boost::algorithm::to_lower(initializerType);
        Initializer* initializer;
        if (initializerType == "xavier") {
            initializer = new XavierInitializer;
        } else if (initializerType == "pretrained") {
            fs::path modelPath(config.get<std::string>("initializer.location"));
            modelPath /= "parameters";
            initializer = new PreTrainedInitializer(modelPath.string());
        } else if (initializerType == "normal") {
            double mean = config.get<double>("initializer.mean", 0.0);
            double var = config.get<double>("initializer.var", 1.0);
            initializer = new RandomNormalInitializer(mean, var);
        } else if (initializerType == "uniform") {
            double low = config.get<double>("initializer.low", -0.1);
            double high = config.get<double>("initializer.high", 0.1);
            initializer = new RandomUniformInitializer(low, high);
        } else {
            BOOST_LOG_TRIVIAL(error) << "Initializer " << initializerType << " not implemented";
        }
        BOOST_LOG_TRIVIAL(info) << "Building initializer " << initializerType;
        return initializer;
    }
};


#endif //THRAX_INITIALIZERFACTORY_H
