//
// Created by martin on 20.02.18.
//

#ifndef THRAX_MODELFACTORY_H
#define THRAX_MODELFACTORY_H

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include <thrax/struct/Data.h>
#include "BaseModel.h"
#include "TransE.h"
#include "DISTMULT.h"
#include "RESCAL.h"
#include "ComplEx.h"

namespace pt = boost::property_tree;

class ModelFactory {
public:
    static BaseModel* buildModel(Data* data, pt::ptree& config) {
        std::string modelType = config.get<std::string>("type");
        boost::algorithm::to_lower(modelType);
        BaseModel* model;
        if (modelType == "transe") {
            model = new TransE(data, config);
        } else if (modelType == "distmult") {
            model = new DISTMULT(data, config);
        } else if (modelType == "rescal") {
            model = new RESCAL(data, config);
        } else if (modelType == "complex") {
            model = new ComplEx(data, config);
        } else {
            BOOST_LOG_TRIVIAL(error) << "Model " << modelType << " not implemented";
        }
        model->initParameterUpdater();
        model->initDumpLocation();
        BOOST_LOG_TRIVIAL(info) << "Building model " << modelType;
        return model;
    }
};


#endif //THRAX_MODELFACTORY_H
