//
// Created by martin on 26.02.18.
//

#ifndef THRAX_RESCAL_H
#define THRAX_RESCAL_H

#include <boost/property_tree/ptree.hpp>
#include <Eigen/Dense>

#include <thrax/struct/Data.h>
#include <boost/property_tree/ptree.hpp>
#include "BaseModel.h"

namespace pt = boost::property_tree;
using namespace Eigen;

class RESCAL: public BaseModel {
private:
    int k;
    EmbeddingParameterSet* E;
    EmbeddingParameterSet* R;
    Gradient* dE;
    Gradient* dR;
    MatrixXd tmpR;

    void initHyperParameters() {
        k = config.get<int>("hyperParameters.k");
    }

    void initParameters() {
        E = &(getParameter("E"));
        R = &(getParameter("R"));
        dE = &(getGradient("E"));
        dR = &(getGradient("R"));
        tmpR.resize(k, k);
    }

public:
    RESCAL(Data* data, pt::ptree& config) : BaseModel(data, config) {
        initHyperParameters();
        addParameter("E", k, data->getNumberOfEntities(), ENTITY);
        addParameter("R", k*k, data->getNumberOfRelations(), RELATION);
        initParameters();
    }

    virtual double score(Triple &triple) override {
        double score = E->col(triple.subject).transpose() * Map<MatrixXd>(R->col(triple.relation).data(), k, k) * E->col(triple.object);
        return score;
    }

    virtual void scoreSubjectRelation(int subjectId, int relationId, VectorXd &scores) override {
        scores = E->col(subjectId).transpose() * Map<MatrixXd>(R->col(relationId).data(), k, k) * (*E);
    }

    virtual void scoreRelationObject(int relationId, int objectId, VectorXd &scores) override {
        scores = E->transpose() * Map<MatrixXd>(R->col(relationId).data(), k, k) * E->col(objectId);
    }

    virtual void gradient(Triple& triple, double scale) override {
        // triple subject: -R_r*E_o
        dE->add(triple.subject, scale*(Map<MatrixXd>(R->col(triple.relation).data(), k, k) * E->col(triple.object)));

        // triple object: -E_s^T*R_r
        dE->add(triple.object, scale*(E->col(triple.subject).transpose() * Map<MatrixXd>(R->col(triple.relation).data(), k, k)));

        // triple relation: -E_s*E_o^T
        tmpR = scale*(E->col(triple.subject) * E->col(triple.object).transpose());
        dR->add(triple.relation, Map<VectorXd>(tmpR.data(), k*k));
    }

    virtual std::string dumpName() override {
        std::stringstream ss;
        ss << "rescal-" << k;
        return ss.str();
    }
};


#endif //THRAX_RESCAL_H
