//
// Created by martin on 09.01.18.
//

#ifndef THRAX_TRANSE_H
#define THRAX_TRANSE_H

#include <boost/property_tree/ptree.hpp>

#include <ostream>

#include <thrax/model/BaseModel.h>
#include <thrax/model/EmbeddingParameterSet.h>
#include <thrax/struct/Triple.h>
#include <thrax/struct/Data.h>
#include <thrax/util/Typedefs.h>
#include <thrax/util/MathUtil.h>

#include <Eigen/Dense>

namespace pt = boost::property_tree;
using namespace Eigen;

/**
 * Model class of TransE model
 */
class TransE: public BaseModel {
private:
    int k; /** dimension of embeddings **/
    bool useL1; /** hyper parameter whether to use L1 as dissimilarity measure, if false L2 is used */
    EmbeddingParameterSet* E;
    EmbeddingParameterSet* R;
    Gradient* dE;
    Gradient* dR;
    VectorXd tmp;
    VectorXd subjectEmbedding;
    VectorXd relationEmbedding;
    VectorXd objectEmbedding;

    void initHyperParameters() {
        k = config.get<int>("hyperParameters.k");
        useL1 = config.get<bool>("hyperParameters.useL1");
    }

    void initParameters() {
        E = &(getParameter("E"));
        R = &(getParameter("R"));
        dE = &(getGradient("E"));
        dR = &(getGradient("R"));
        tmp.resize(k);
        subjectEmbedding.resize(k);
        relationEmbedding.resize(k);
        objectEmbedding.resize(k);
    }

public:
    TransE(Data* data, pt::ptree& config): BaseModel(data, config) {
        initHyperParameters();
        addParameter("E", k, numberOfEntities, ENTITY);
        addParameter("R", k, numberOfRelations, RELATION);
        initParameters();
    }

    /**
     * Calculate TransE score of a given triple
     * @param triple
     * @return TransE score of a given triple
     */
    virtual double score(Triple &triple) override {
        tmp = E->col(triple.subject) + R->col(triple.relation) - E->col(triple.object);
        if (useL1) {
            return -tmp.lpNorm<1>();
        } else {
            return -tmp.squaredNorm();
        }
    }

    virtual void gradient(Triple& triple, double scale) override {
        tmp = E->col(triple.subject);
        tmp += R->col(triple.relation);
        tmp -= E->col(triple.object);
        tmp *= 2;
        if (useL1) {
            for (int i = 0; i < k; ++i) {
                if (tmp[i] > 0) {
                    tmp[i] = 1;
                } else {
                    tmp[i] = -1;
                }
            }
        }
        tmp *= -1;
        // store gradients
        dE->add(triple.subject, scale*tmp);
        dR->add(triple.relation, scale*tmp);
        dE->add(triple.object, -scale*tmp);
    }

    virtual std::string dumpName() override {
        std::stringstream ss;
        ss << "transe-" << k;
        if (useL1) {
            ss << "-l1";
        } else {
            ss << "-l2";
        }
        return ss.str();
    }
};


#endif //THRAX_TRANSE_H
