#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include "boost/variant.hpp"

#include <thrax/struct/Data.h>
#include <thrax/model/BaseModel.h>
#include <thrax/model/TrivialEnsemble.h>
#include <thrax/model/ModelFactory.h>
#include <thrax/optimizer/Optimizer.h>
#include <thrax/evaluation/Evaluation.h>
#include <thrax/util/GradientChecker.h>

namespace pt = boost::property_tree;
namespace po = boost::program_options;
namespace logging = boost::log;

int main(int argc, char** argv) {
    std::string configFilePath;
    po::options_description description("Allowed options");
    description.add_options()
            ("help,h", "produce help message")
            ("configFile,c", po::value<std::string>(&configFilePath)->required()->default_value("./config/config.json"), "path to config file");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, description), vm);
    po::notify(vm);

    // set up logging
    logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::info);

    // show help and quit
    if (vm.count("help") > 0) {
        BOOST_LOG_TRIVIAL(info) << description;
        return 0;
    }

    // load config
    BOOST_LOG_TRIVIAL(info) << "Using config file " << configFilePath;
    pt::ptree config;
    pt::read_json(configFilePath, config);

    // example access properties
    std::string dataDirectory = config.get<std::string>("data.dir");
    std::string trainPath = dataDirectory + config.get<std::string>("data.trainFile");
    std::string testPath = dataDirectory + config.get<std::string>("data.testFile");
    std::string validPath = dataDirectory + config.get<std::string>("data.validFile");
    int limit = config.get<int>("data.limit", -1);
    BOOST_LOG_TRIVIAL(info) << "Using data files in directory " << dataDirectory;

    // load train data
    Data trainData;
    if (config.get<bool>("data.loadMappings", false)) {
        std::string dumpLocation = config.get<std::string>("data.location");
        trainData.loadMappings(dumpLocation);
    }
    trainData.load(trainPath, false, limit);
    if (config.get<bool>("data.dumpMappings", true)) {
        std::string dumpLocation = config.get<std::string>("data.dumpLocation", "auto");
        trainData.dumpMappings(dumpLocation);
    }

    bool ignoreNewConstituents = config.get<bool>("data.ignoreNewConstituents", true);

    // load validation data
    Data validData;
    validData.setMaps(trainData.getEntityMap(), trainData.getRelationMap());
    validData.load(validPath, ignoreNewConstituents, limit);

    // load test data
    Data testData;
    testData.setMaps(trainData.getEntityMap(), trainData.getRelationMap());
    testData.load(testPath, ignoreNewConstituents, limit);

    // optionally append validation data to train data
    bool trainOnValidation = config.get<bool>("optimizer.trainOnValidation", false);
    if (trainOnValidation) {
        trainData.addTriples(validData);
    }

    // set up model
    AbstractModel* model;
    std::string modelType = config.get<std::string>("model.type");
    boost::algorithm::to_lower(modelType);
    if (modelType == "ensemble") {
        BOOST_LOG_TRIVIAL(info) << "Building model " << modelType;
        model = new TrivialEnsemble(&trainData, config.get_child("model"));
        model->initParameterUpdater();
        model->initDumpLocation();
    } else {
        model = ModelFactory::buildModel(&trainData, config.get_child("model"));
    }

    FileUtil::dumpConfig(model->getDumpLocation(), config);

    // set up optimizer
    Optimizer* optimizer = new Optimizer(&trainData, &validData, &testData, model, config);
    BOOST_LOG_TRIVIAL(info) << "##### START OF TRAINING #####";
    optimizer->fit();

    // optionally check gradients
    if (config.get<bool>("checkGradients", false)) {
        Triple& triple = trainData.getTriple(0);
        GradientChecker::checkGradients(dynamic_cast<BaseModel*>(model), triple);
        return 0;
    }

    BOOST_LOG_TRIVIAL(info) << "##### START OF EVALUATION #####";

    // set up evaluation
    std::vector<Data*> lookupDataSets {&trainData, &validData, &testData};
    Evaluation evaluation(lookupDataSets);
    if (config.get<bool>("optimizer.testOnValidation", false)) {
        evaluation.evaluate(model, &validData, model->getDumpLocation());
    } else {
        evaluation.evaluate(model, &testData, model->getDumpLocation());
    }
    return 0;
}