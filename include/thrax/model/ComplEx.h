//
// Created by martin on 06.03.18.
//

#ifndef THRAX_COMPLEX_H
#define THRAX_COMPLEX_H


#include <thrax/struct/Data.h>
#include <thrax/util/MathUtil.h>
#include "BaseModel.h"

class ComplEx: public BaseModel {
private:
    int k; /** dimension of embeddings **/

    EmbeddingParameterSet* Er;
    EmbeddingParameterSet* Ei;
    EmbeddingParameterSet* Rr;
    EmbeddingParameterSet* Ri;
    Gradient* dEr;
    Gradient* dEi;
    Gradient* dRr;
    Gradient* dRi;
    VectorXd r_r;
    VectorXd r_i;
    VectorXd s_r;
    VectorXd s_i;
    VectorXd o_r;
    VectorXd o_i;

    void initHyperParameters() {
        k = config.get<int>("hyperParameters.k");
    }

    void initParameters() {
        Er = &(getParameter("Er"));
        Rr = &(getParameter("Rr"));
        Ei = &(getParameter("Ei"));
        Ri = &(getParameter("Ri"));
        dEr = &(getGradient("Er"));
        dRr = &(getGradient("Rr"));
        dEi = &(getGradient("Ei"));
        dRi = &(getGradient("Ri"));
        r_r.resize(k);
        r_i.resize(k);
        s_r.resize(k);
        s_i.resize(k);
        o_r.resize(k);
        o_i.resize(k);
    }

public:
    ComplEx(Data* data, pt::ptree& config): BaseModel(data, config) {
        initHyperParameters();
        addParameter("Er", k, numberOfEntities, ENTITY);
        addParameter("Rr", k, numberOfRelations, RELATION);
        addParameter("Ei", k, numberOfEntities, ENTITY);
        addParameter("Ri", k, numberOfRelations, RELATION);
        initParameters();
    }

    virtual double score(Triple &triple) override {
        s_r = Er->col(triple.subject);
        s_i = Ei->col(triple.subject);
        r_r = Rr->col(triple.relation);
        r_i = Ri->col(triple.relation);
        o_r = Er->col(triple.object);
        o_i = Ei->col(triple.object);
        return MathUtil::dot(r_r, s_r, o_r) + MathUtil::dot(r_r, s_i, o_i) + MathUtil::dot(r_i, s_r, o_i) - MathUtil::dot(r_i, s_i, o_r);
    }

    virtual void gradient(Triple& triple, double scale) override {
        s_r = Er->col(triple.subject);
        s_i = Ei->col(triple.subject);
        r_r = Rr->col(triple.relation);
        r_i = Ri->col(triple.relation);
        o_r = Er->col(triple.object);
        o_i = Ei->col(triple.object);

        dEr->add(triple.subject,  (r_r.array()*o_r.array() + r_i.array()*o_i.array())*scale);
        dEi->add(triple.subject,  (r_r.array()*o_i.array() - r_i.array()*o_r.array())*scale);
        dRr->add(triple.relation, (s_r.array()*o_r.array() + s_i.array()*o_i.array())*scale);
        dRi->add(triple.relation, (s_r.array()*o_i.array() - s_i.array()*o_r.array())*scale);
        dEr->add(triple.object,   (r_r.array()*s_r.array() - r_i.array()*s_i.array())*scale);
        dEi->add(triple.object,   (r_r.array()*s_i.array() + r_i.array()*s_r.array())*scale);
    }

    virtual std::string dumpName() override {
        std::stringstream ss;
        ss << "complex-" << k;
        return ss.str();
    }
};


#endif //THRAX_COMPLEX_H
