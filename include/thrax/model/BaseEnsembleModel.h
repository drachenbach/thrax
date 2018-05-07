//
// Created by root on 23.03.18.
//

#ifndef THRAX_BASEENSEMBLEMODEL_H
#define THRAX_BASEENSEMBLEMODEL_H

#include <boost/property_tree/ptree.hpp>

#include <thrax/util/Typedefs.h>
#include "AbstractModel.h"
#include "BaseModel.h"
#include "ModelFactory.h"

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;
using namespace Eigen;

/**
 * This is the superclass for all ensemble embedding models. It maintains a list of individual models, as well
 * as parameters for the ensemble itself (e.g. weights).
 */
class BaseEnsembleModel: public AbstractModel {
public:
    /** ##### CONSTRUCTORS ##### **/

    BaseEnsembleModel(): AbstractModel() {

    }

    BaseEnsembleModel(Data* data, pt::ptree& config): AbstractModel(data, config) {
        init();
    }

    /** ##### GRADIENTS ##### **/

    /**
     * Calls the reset gradient method as well as the update method of each model
     */
    virtual void resetGradients() override {
        AbstractModel::resetGradients();
        if (trainModels) {
            for (int i = 0; i < m; ++i) {
                models[i]->resetGradients();
            }
        }
    }

    /** ##### MODEL UPDATES ##### **/

    /**
     * Calls the ensemble update method as well as the update method of each model
     */
    virtual void update() override {
        AbstractModel::update();
        if (trainModels) {
            for (int i = 0; i < m; ++i) {
                models[i]->update();
            }
        }
    }

    /** ##### REGULARIZATION ##### **/

    /**
     * Apply L2 regularization to embeddings.
     * @param scale is multiplied with lambda_* hyper parameters, e.g. when dividing by batch size
     */
    virtual void l2(double scale) override {
        if (trainModels) {
            for (int i = 0; i < m; ++i) {
                models[i]->l2(scale);
            }
        }
    }

    /** ##### HOOKS ##### **/

    /**
     * Calls the post epoch hooks of each model
     */
    virtual void postEpoch() override {
        if (trainModels) {
            for (int i = 0; i < m; ++i) {
                models[i]->postEpoch();
            }
        }
    }

    /**
     * Calls the post batch hooks of each model
     */
    virtual void postBatch() override {
        if (trainModels) {
            for (int i = 0; i < m; ++i) {
                models[i]->postBatch();
            }
        }
    }

    /** ##### SERIALIZATION ##### **/

    virtual void dump(std::string location) override {
        // dump ensemble hyper parameters
        AbstractModel::dump(location);
        // dump all models
        fs::path dir(location);
        dir /= "models";
        fs::create_directories(dir);
        for (int i = 0; i < m; ++i) {
            fs::path modelPath = dir / models[i]->dumpName();
            models[i]->dump(modelPath.string());
        }
    }

    virtual std::string dumpName() override {
        return "ensemble";
    }

protected:
    std::vector<BaseModel*> models;
    int m; /** number of models **/
    bool trainModels; /** whether to train individual models or keep them fixed **/

private:
    void init() {
        pt::ptree modelConfig;
        for (auto& ptr: config.get_child("models")) {
            // add the update config to each model
            modelConfig = ptr.second;
            modelConfig.put_child("update", config.get_child("update"));
            BaseModel* model = ModelFactory::buildModel(data, modelConfig);
            models.push_back(model);
        }
        m = models.size();
        trainModels = config.get<bool>("trainModels", true);
    }
};


#endif //THRAX_BASEENSEMBLEMODEL_H
