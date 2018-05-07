//
// Created by martin on 09.01.18.
//

#ifndef THRAX_BASEMODEL_H
#define THRAX_BASEMODEL_H

#include <boost/property_tree/ptree.hpp>

#include <thrax/util/Typedefs.h>
#include "AbstractModel.h"

namespace pt = boost::property_tree;

/**
 * This is the superclass for all embedding models which implements all common functionality
 */
class BaseModel: public AbstractModel {
public:
    /** ##### CONSTRUCTORS ##### **/

    BaseModel(): AbstractModel() {}

    BaseModel(Data* data, pt::ptree& config): AbstractModel(data, config) {
        init();
    }

    /** ##### REGULARIZATION ##### **/

    /**
     * Apply L2 regularization to embeddings.
     * @param scale is multiplied with lambda_* hyper parameters, e.g. when dividing by batch size
     */
    virtual void l2(double scale) override {
        if (scale * lambda_e > 0.0) {
            for (auto name: entityEmbeddings) {
                getGradient(name).l2(lambda_e*scale);
            }
        }
        if (scale * lambda_r > 0.0) {
            for (auto name: relationEmbeddings) {
                getGradient(name).l2(lambda_r*scale);
            }
        }
    }

    /** ##### HOOKS ##### **/
    virtual void postBatch() override {
        // normalize embeddings
        if (normalizeEntities) {
            for (auto name: entityEmbeddings) {
                getGradient(name).normalize();
            }
        }
        if (normalizeRelations) {
            for (auto name: relationEmbeddings) {
                getGradient(name).normalize();
            }
        }
    }

    /** ##### GRADIENT CHECKING ##### **/

    enum ParameterType {
        ENTITY, /** parameter for both subjects and objects **/
        SUBJECT, /** parameter for subjects only **/
        RELATION, /** parameter for relations **/
        OBJECT, /** parameter for objects only **/
        META /** meta parameter e.g. ensemble model weights **/
    };

    const std::set<std::string>& getSubjectEmbeddings() const {
        return subjectEmbeddings;
    }

    const std::set<std::string>& getRelationEmbeddings() const {
        return relationEmbeddings;
    }

    const std::set<std::string>& getObjectEmbeddings() const {
        return objectEmbeddings;
    }

protected:
    std::set<std::string> subjectEmbeddings; /** list of parameter set names used for subject embeddings (used by gradient checking) **/
    std::set<std::string> relationEmbeddings; /** list of parameter set names used for relation embeddings (used by gradient checking and L2 regularization) **/
    std::set<std::string> objectEmbeddings; /** list of parameter set names used for object embeddings (used by gradient checking) **/
    std::set<std::string> entityEmbeddings; /** list of parameter set names used for entity embeddings (used by L2 regularization) **/
    double lambda_e; /** L2 regularization hyper parameter for entity embeddings **/
    double lambda_r; /** L2 regularization hyper parameter for relation embeddings **/
    bool normalizeEntities; /** normalize entity embeddings to unit norm **/
    bool normalizeRelations; /** normalize relation embeddings to unit norm **/

    /**
     * Adds a embedding parameter set to the model
     * @param name name of the parameter
     * @param k embedding dimension
     * @param m number of rows (entities/relations)
     */
    void addParameter(std::string name, int k, int m, ParameterType parameterType) {
        // call super-class to register parameter
        AbstractModel::addParameter(name, k, m);

        // add parameter names to respective list (used by gradient checking)
        if (parameterType == ENTITY || parameterType == SUBJECT) {
            subjectEmbeddings.insert(name);
            entityEmbeddings.insert(name);
        }
        if (parameterType == RELATION) {
            relationEmbeddings.insert(name);
        }
        if (parameterType == ENTITY || parameterType == OBJECT) {
            objectEmbeddings.insert(name);
            entityEmbeddings.insert(name);
        }
    }

private:
    void init() {
        lambda_e = config.get<double>("hyperParameters.lambda_e", 0.0);
        lambda_r = config.get<double>("hyperParameters.lambda_r", 0.0);
        normalizeEntities = config.get<bool>("hyperParameters.normalizeEntities", false);
        normalizeRelations = config.get<bool>("hyperParameters.normalizeRelations", false);
    }
};


#endif //THRAX_BASEMODEL_H
