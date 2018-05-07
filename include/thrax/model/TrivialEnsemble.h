//
// Created by martin on 22.03.18.
//

#ifndef THRAX_TRIVIALENSEMBLEL_H
#define THRAX_TRIVIALENSEMBLEL_H

#include <boost/property_tree/ptree.hpp>

#include <ostream>

#include <thrax/model/BaseModel.h>
#include <thrax/model/EmbeddingParameterSet.h>
#include <thrax/struct/Triple.h>
#include <thrax/struct/Data.h>
#include <thrax/util/Typedefs.h>
#include <thrax/util/MathUtil.h>

#include <Eigen/Dense>
#include "ModelFactory.h"
#include "BaseEnsembleModel.h"

namespace pt = boost::property_tree;
using namespace Eigen;

class TrivialEnsemble: public BaseEnsembleModel {
private:
    EmbeddingParameterSet* weights;
    Gradient* dWeights;
    VectorXd tmp;
    bool weightsByRelation; /** by relation (true) or global (false) **/
    bool normalizeWeights;
    bool fixWeights;
    int k; /** number of weight vectors, #relation or 1 (global) **/

    void initHyperParameters() {
        weightsByRelation = config.get<bool>("hyperParameters.weightsByRelation");
        normalizeWeights = config.get<bool>("hyperParameters.normalizeWeights", false);
        fixWeights = config.get<bool>("hyperParameters.fixWeights", false);
        tmp.resize(m);
        k = 1;
        if (weightsByRelation) {
            k = data->getNumberOfRelations();
        }
    }

    void initParameters() {
        weights = &(getParameter("weights"));
        dWeights = &(getGradient("weights"));
        // uniform weight initialization
        for (int i = 0; i < k; ++i) {
            weights->col(i) = VectorXd::Constant(m, 1.0/m);
        }
    }
public:
    /** ##### CONSTRUCTORS ##### **/

    TrivialEnsemble(Data* data, pt::ptree& config): BaseEnsembleModel(data, config) {
        initHyperParameters();
        addParameter("weights", m, k);
        initParameters();
    }

    /** ##### SCORING ##### **/

    /**
     * Trivial scoring function.
     * With global weights: \sum_{m \in models} weight_m * score_m(triple)
     * With relation weights: \sum_{m \in models} weight_m^{triple.relation} * score_m(triple)
     * @param triple
     * @return
     */
    virtual double score(Triple &triple) override {
        double score = 0.0;
        int j = 0;
        if (weightsByRelation) {
            j = triple.relation;
        }
        for (int i = 0; i < m; ++i) {
            score += (*weights)(i, j) * models[i]->score(triple);
        }
        return score;
    }

    /** ##### GRADIENTS ##### **/

    virtual void gradient(Triple &triple, double scale) override {
        int j = 0;
        if (weightsByRelation) {
            j = triple.relation;
        }
        for (int i = 0; i < m; ++i) {
            tmp[i] = models[i]->score(triple);
            if (trainModels) {
                models[i]->gradient(triple, (*weights)(i, j) * scale);
            }
        }
        if (!fixWeights) {
            dWeights->add(j, tmp*scale);
        }
    }

    /** ##### SERIALIZATION ##### **/

    virtual std::string dumpName() override {
        std::stringstream ss;
        ss << "ensemble";
        if (weightsByRelation) {
            ss << "-relation";
        } else {
            ss << "-global";
        }
        if (trainModels) {
            ss << "-finetune";
        }
        for (int i = 0; i < m; ++i) {
            ss << "-" << models[i]->dumpName();
        }
        return ss.str();
    }

    virtual void postBatch() override {
        BaseEnsembleModel::postBatch();
        if (normalizeWeights) {
            dWeights->normalize();
        }
    }
};


#endif //THRAX_TRIVIALENSEMBLEL_H
