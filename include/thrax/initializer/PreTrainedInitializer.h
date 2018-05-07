//
// Created by martin on 20.03.18.
//

#ifndef THRAX_PRETRAINEDINITIALIZER_H
#define THRAX_PRETRAINEDINITIALIZER_H

#include <boost/filesystem.hpp>

#include <thrax/util/FileUtil.h>
#include "Initializer.h"

namespace fs = boost::filesystem;

class PreTrainedInitializer: public Initializer {
public:
    PreTrainedInitializer(const std::string& baseLocation) : baseLocation(baseLocation), Initializer() {}

    virtual void initialize(EmbeddingParameterSet& parameterSet, const std::string name) override {
        if (fs::exists(baseLocation)) {
            fs::path dir(baseLocation);
            // load parameters
            fs::path parameterPath = dir / name;
            FileUtil::loadMatrix(parameterPath.string(), parameterSet);
        } else {
            BOOST_LOG_TRIVIAL(error) << "Could not find location " << baseLocation << " to load embeddings";
        }
    };

private:
    std::string baseLocation;
};


#endif //THRAX_PRETRAINEDINITIALIZER_H
