//
// Created by martin on 02.03.18.
//

#ifndef THRAX_SAMPLERFACTORY_H
#define THRAX_SAMPLERFACTORY_H

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include "Sampler.h"
#include "LCWASampler.h"
#include "CorruptionSampler.h"

namespace pt = boost::property_tree;

class SamplerFactory {
public:
    static Sampler* buildSampler(Data* data, pt::ptree& hyperParameters) {
        std::string samplingStrategy = hyperParameters.get<std::string>("type", "lcwa");
        boost::algorithm::to_lower(samplingStrategy);
        Sampler* sampler;
        if (samplingStrategy == "lcwa") {
            sampler = new LCWASampler(data, hyperParameters);
        } else if (samplingStrategy == "corruption") {
            sampler = new CorruptionSampler(data, hyperParameters);
        } else {
            BOOST_LOG_TRIVIAL(error) << "Could not find sampling strategy " << samplingStrategy;
        }
        BOOST_LOG_TRIVIAL(info) << "Using sampling strategy " << samplingStrategy;
        return sampler;
    }
};


#endif //THRAX_SAMPLERFACTORY_H
