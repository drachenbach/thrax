//
// Created by martin on 14.02.18.
//

#ifndef THRAX_DISTMULT_H
#define THRAX_DISTMULT_H

#include <boost/property_tree/ptree.hpp>
#include <Eigen/Dense>

#include <thrax/model/BaseModel.h>
#include <thrax/model/EmbeddingParameterSet.h>
#include <thrax/struct/Triple.h>
#include <thrax/struct/Data.h>
#include <thrax/util/Typedefs.h>
#include <thrax/util/MathUtil.h>

namespace pt = boost::property_tree;
using namespace Eigen;

class DISTMULT: public BaseModel {
private:
    int k;
    EmbeddingParameterSet* E;
    EmbeddingParameterSet* R;
    Gradient* dE;
    Gradient* dR;
    VectorXd subjectEmbedding;
    VectorXd relationEmbedding;
    VectorXd objectEmbedding;

    void initHyperParameters() {
        k = config.get<int>("hyperParameters.k");
    }

    void initParameters() {
        E = &(getParameter("E"));
        R = &(getParameter("R"));
        dE = &(getGradient("E"));
        dR = &(getGradient("R"));
        subjectEmbedding.resize(k);
        relationEmbedding.resize(k);
        objectEmbedding.resize(k);
    }

public:
    DISTMULT(Data* data, pt::ptree& config): BaseModel(data, config) {
        initHyperParameters();
        addParameter("E", k, numberOfEntities, ENTITY);
        addParameter("R", k, numberOfRelations, RELATION);
        initParameters();
    }

    virtual double score(Triple &triple) override {
        return (E->col(triple.subject).array() * R->col(triple.relation).array() * E->col(triple.object).array()).sum();
    }

    virtual void gradient(Triple& triple, double scale) override {
        subjectEmbedding = R->col(triple.relation).array() * E->col(triple.object).array();
        relationEmbedding = E->col(triple.subject).array() * E->col(triple.object).array();
        objectEmbedding = R->col(triple.relation).array() * E->col(triple.subject).array();

        // store gradients
        dE->add(triple.subject, subjectEmbedding*scale);
        dR->add(triple.relation, relationEmbedding*scale);
        dE->add(triple.object, objectEmbedding*scale);
    }

    virtual std::string dumpName() override {
        std::stringstream ss;
        ss << "distmult-" << k;
        return ss.str();
    }
};


#endif //THRAX_DISTMULT_H
