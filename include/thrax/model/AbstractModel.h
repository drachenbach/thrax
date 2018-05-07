//
// Created by root on 23.03.18.
//

#ifndef THRAX_ABSTRACTMODEL_H
#define THRAX_ABSTRACTMODEL_H

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <sys/time.h>
#include <time.h>

#include <thrax/model/EmbeddingParameterSet.h>
#include <thrax/initializer/Initializer.h>
#include <thrax/initializer/InitializerFactory.h>
#include <thrax/struct/Gradient.h>
#include <thrax/struct/Triple.h>
#include <thrax/util/Typedefs.h>
#include <thrax/util/FileUtil.h>
#include <thrax/parameterUpdater/ParameterUpdater.h>

class ParameterUpdaterFactory;
#include <thrax/parameterUpdater/ParameterUpdaterFactory.h>

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;
using namespace Eigen;

class AbstractModel {
public:
    /** ##### CONSTRUCTORS ##### **/

    AbstractModel() {

    }

    /**
     * Construct a model using statistics of data (number of entities and relations) and config
     * @param data
     * @param config property tree of config
     */
    AbstractModel(Data* data, pt::ptree& config): config(config), data(data), numberOfEntities(data->getNumberOfEntities()), numberOfRelations(data->getNumberOfRelations()) {
        init();
    }

    void initParameterUpdater() {
        // initialize updater
        updater = ParameterUpdaterFactory::buildParameterUpdater(parameters, config.get_child("update"));
    }

    void initDumpLocation() {
        // initialize dumpDir
        std::string dumpDir = config.get<std::string>("serialization.dumpDirectory", "auto");
        if (dumpDir == "auto") {
            dumpDir = "./models";
        }
        fs::path dir(dumpDir);

        // initialize dumpLocation
        dumpLocation = config.get<std::string>("serialization.dumpLocation", "auto");
        if (dumpLocation == "auto") {
            dumpLocation = dumpName();
            // append current time to dump folder
            char buffer[80];
            int milli;
            struct tm* tm_info;
            struct timeval tv;

            gettimeofday(&tv, NULL);

            milli = lrint(tv.tv_usec / 1000.0); // Round to nearest millisec
            if (milli >= 1000) { // Allow for rounding up to nearest second
                milli -= 1000;
                tv.tv_sec++;
            }

            tm_info = localtime(&tv.tv_sec);

            strftime(buffer, 80, "-%Y-%m-%d-%H-%M-%S", tm_info);

            dumpLocation.append(buffer);
            dumpLocation.append("-");
            dumpLocation.append(std::to_string(milli));
        }
        fs::path modelDir(dumpLocation);

        // merge paths
        dir /= modelDir;
        dumpLocation = dir.string();
    }

    /** ##### DESTRUCTOR ##### **/
    ~AbstractModel() {
        delete initializer;
        delete updater;
    }

    /** ##### SCORING ##### **/

    /**
     * Calculate model score of a given triple
     * @param triple
     * @return model score of a given triple
     */
    virtual double score(Triple& triple) = 0;

    /**
     * Calculate scores of <subject, relation, ?> triples with multiple objects.
     * Naively calls score computation for each triple separately, can be overwritten to be more efficient
     * @param subjectId
     * @param relationId
     * @param scores
     */
    virtual void scoreSubjectRelation(int subjectId, int relationId, VectorXd& scores) {
        Triple triple(subjectId, relationId, 0);
        for (int i = 0; i < numberOfEntities; ++i) {
            triple.object = i;
            scores[i] = score(triple);
        }
    }

    /**
     * Calculate scores of <?, relation, object> triples with multiple objects.
     * Naively calls score computation for each triple separately, can be overwritten to be more efficient
     * @param subjectId
     * @param relationId
     * @param scores
     */
    virtual void scoreRelationObject(int relationId, int objectId, VectorXd& scores) {
        Triple triple(0, relationId, objectId);
        for (int i = 0; i < numberOfEntities; ++i) {
            triple.subject = i;
            scores[i] = score(triple);
        }
    }

    /** ##### GRADIENTS ##### **/

    virtual void gradient(Triple& triple, double scale) = 0;

    virtual void resetGradients() {
        for (auto& gradient: gradients) {
            gradient.second.reset();
        }
    };

    /** ##### MODEL UPDATES ##### **/

    virtual void update() {
        updater->calculateUpdate(gradients);
        for (auto& gradient: gradients) {
            for (auto& ptr: gradient.second.getIdToCol()) {
                parameters.at(gradient.first).col(ptr.first) -= gradient.second.get(ptr.first);
            }
        }
    };

    /** ##### REGULARIZATION ##### **/

    /**
     * Apply L2 regularization to embeddings.
     * @param scale is multiplied with lambda_* hyper parameters, e.g. when dividing by batch size
     */
    virtual void l2(double scale) = 0;

    /** ##### HOOKS ##### **/

    virtual void postEpoch() {
        // do nothing by default
    }

    virtual void postBatch() {
        // do nothing by default
    }

    /** ##### SERIALIZATION ##### **/

    /**
     * Dumps model to disk into the specified location
     * @param subPath path starting from model.dumpLocation
     */
    virtual void dump(std::string location) {
        fs::path dir(location);
        fs::create_directories(dir);
        BOOST_LOG_TRIVIAL(info) << "Dumping model to " << dir.string();
        // dump parameters into separate files
        for(auto& parameter: parameters) {
            fs::path file(parameter.first);
            fs::path parameterPath = dir / "parameters";
            fs::create_directories(parameterPath);
            fs::path fullPath = parameterPath / file;
            FileUtil::dumpMatrix(fullPath.string(), parameter.second);
        }
    }

    virtual std::string dumpName() {
        return "model";
    }

    /** ##### GETTER AND SETTER ##### **/

    /**
     * Get parameter set by name
     * @param name
     * @return parameter set
     */
    EmbeddingParameterSet& getParameter(const std::string name) {
        return parameters.at(name);
    }

    const ParameterMap& getParameters() const {
        return parameters;
    }

    /**
     * Get gradient by name
     * @param name
     * @return gradient
     */
    Gradient& getGradient(const std::string name) {
        return gradients.at(name);
    }

    GradientMap& getGradients() {
        return gradients;
    }

    const std::string &getDumpLocation() const {
        return dumpLocation;
    }

protected:
    pt::ptree config; /** boost property tree of config **/
    int numberOfEntities; /** number of entities **/
    int numberOfRelations; /** number of relations **/
    Initializer* initializer; /** the initializer to initialize parameters **/
    ParameterMap parameters; /** hash map: parameter set name -> parameter set **/
    GradientMap gradients; /** hash map: parameter set name -> gradient **/
    ParameterUpdater* updater; /** pointer to a parameter updates **/
    Data* data;
    std::string dumpLocation;

    /**
     * Adds a embedding parameter set to the model
     * @param name name of the parameter
     * @param k embedding dimension
     * @param m number of rows (entities/relations)
     */
    void addParameter(std::string name, int k, int m) {
        // add parameter to map
        parameters.insert({name, EmbeddingParameterSet(k, m)});
        // initialize parameter
        initializer->initialize(parameters.at(name), name);
        // add parameter to gradient map
        gradients.insert({name, Gradient(&parameters.at(name))});
    }

private:
    void init() {
        // initialize initializer
        initializer = InitializerFactory::buildInitializer(config);
    }
};


#endif //THRAX_ABSTRACTMODEL_H
