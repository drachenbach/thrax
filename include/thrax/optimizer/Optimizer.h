//
// Created by martin on 20.02.18.
//

#ifndef THRAX_OPTIMIZER_H
#define THRAX_OPTIMIZER_H

#include <numeric>
#include <random>
#include <algorithm>
#include <vector>
#include <boost/timer/timer.hpp>
#include <boost/property_tree/ptree.hpp>

#include <thrax/sampler/Sampler.h>
#include <thrax/sampler/SamplerFactory.h>
#include <thrax/struct/Data.h>
#include <thrax/util/Typedefs.h>
#include <thrax/initializer/Initializer.h>
#include <thrax/initializer/ScalarInitializer.h>
#include <thrax/struct/Gradient.h>
#include <thrax/evaluation/Evaluation.h>
#include <thrax/lossFunction/LossFunction.h>
#include <thrax/lossFunction/LossFunctionFactory.h>
#include <thrax/model/AbstractModel.h>

namespace pt = boost::property_tree;

class Optimizer {
public:
    Optimizer(Data* trainData, Data* validData, Data* testData, AbstractModel* model, pt::ptree config):
    trainData(trainData),
    validData(validData),
    testData(testData),
    model(model),
    config(config) {
        init();
    }

    void fit() {
        // create vector of indices
        std::vector<int> indices(trainData->getNumberOfTriples());
        std::iota(indices.begin(), indices.end(), 0);
        int numberOfBatches = (trainData->getNumberOfTriples() + batchSize - 1) / batchSize;
        boost::timer::cpu_timer totalTimer;
        boost::timer::cpu_timer epochTimer;
        std::vector<Data*> lookupDataSets {trainData, validData, testData};
        Evaluation evaluation(lookupDataSets);
        bool isModelDumped = false;
        int epochsTrained = 0;

        // start training loop
        for (int epoch = 0; epoch < maxEpochs; ++epoch) {
            BOOST_LOG_TRIVIAL(info) << "Start epoch " << epoch + 1 << "/" << maxEpochs;
            preEpoch();
            epochTimer.start();
            // randomly shuffle examples
            std::random_shuffle(indices.begin(), indices.end());
            // loop over batches
            for (int batch = 0; batch < numberOfBatches; ++batch) {
                processBatch(indices, batch * batchSize, std::min(batch * batchSize + batchSize, trainData->getNumberOfTriples()));
            }
            model->postEpoch();
            epochTimer.stop();
            BOOST_LOG_TRIVIAL(info) << "Finished epoch in " << epochTimer.format(3, "%w sec");
            double time = epochTimer.elapsed().wall / 1000000000.0;
            times[epoch] = time;
            losses[epoch] = lossFunction->getLoss();
            epochsTrained++;
            postEpoch();
            // early stopping
            if (useEarlyStopping && (epoch + 1) % earlyStoppingEveryNEpochs == 0) {
                BOOST_LOG_TRIVIAL(info) << "Early stopping: start evaluation on validation data.";
                double rawMRR;
                double filteredMRR;
                evaluation.MRR(model, validData, earlyStoppingN, rawMRR, filteredMRR);
                mrrs[epoch] = filteredMRR;
                // check if score on validation data improves, if not break the loop
                if (filteredMRR > earlyStoppingBestValue) {
                    BOOST_LOG_TRIVIAL(info) << "Early stopping: performance on validation data did improve. Filtered MRR: " << earlyStoppingBestValue << " -> " << filteredMRR << ", raw MRR: " << rawMRR;
                    earlyStoppingBestValue = filteredMRR;
                    // dump so far best performing model
                    model->dump(model->getDumpLocation());
                    isModelDumped = true;
                } else {
                    BOOST_LOG_TRIVIAL(info) << "Early stopping: performance on validation data did not improve. Filtered MRR: " << earlyStoppingBestValue << " -> " << filteredMRR << ", raw MRR: " << rawMRR << ". Stop training.";
                    break;
                }
            }
        }
        // dump model if not already dumped
        if (!isModelDumped) {
            model->dump(model->getDumpLocation());
        }
        totalTimer.stop();
        BOOST_LOG_TRIVIAL(info) << "Total time: " << totalTimer.format(3, "%w sec");
        // log training statistics
        FileUtil::dumpTrainingStatistics(model->getDumpLocation(), losses, times, mrrs, epochsTrained);
    }

protected:
    Data* trainData; /** reference to the knowledge base **/
    Data* validData; /** reference to the knowledge base **/
    Data* testData; /** reference to the knowledge base **/
    AbstractModel* model; /** pointer to the model **/
    Sampler* sampler; /** pointer to a sampler **/
    LossFunction* lossFunction;
    pt::ptree config; /** property tree of hyper parameters **/

    int batchSize; /** size of a single batch in mini-batch gradient descent **/
    int maxEpochs; /** maximum number of epochs for gradient descent **/
    int numberOfNegatives; /** number of negative triples that should be sampled for each positive triple **/
    bool includePositiveInNegatives;
    bool useEarlyStopping;
    int earlyStoppingEveryNEpochs; /** evaluate model on validation set every n epochs **/
    double earlyStoppingBestValue; /** historically best value of validation metric **/
    int earlyStoppingN;

    std::vector<std::vector<Triple> > negatives; /** re-usable data structure for sampled negative triples of a batch **/
    std::vector<Triple*> positives; /** re-usable data structure for positive triples of a batch **/

    // logging
    std::vector<double> losses;
    std::vector<double> times;
    std::vector<double> mrrs;

    void init() {
        // initialize hyper parameters
        batchSize = config.get<int>("optimizer.batchSize");
        maxEpochs = config.get<int>("optimizer.maxEpochs");
        numberOfNegatives = config.get<int>("optimizer.sampling.numberOfNegatives");
        useEarlyStopping = config.get<bool>("optimizer.earlyStopping.useEarlyStopping", true);
        earlyStoppingEveryNEpochs = config.get<int>("optimizer.earlyStopping.everyNEpochs", 50);
        earlyStoppingN = config.get<int>("optimizer.earlyStopping.n", 1000);
        earlyStoppingBestValue = 0.0;

        //  initialize loss function
        lossFunction = LossFunctionFactory::buildLossFunction(model, config);
        // only include positive triple in list of negative if using a softmax loss function
        if (lossFunction->getType() == "softmax") {
            includePositiveInNegatives = true;
        } else {
            includePositiveInNegatives = false;
        }

        // initialize sampler
        sampler = SamplerFactory::buildSampler(trainData, config.get_child("optimizer.sampling"));

        // initialize cache variables
        positives.resize(batchSize);
        negatives.resize(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            if (!includePositiveInNegatives) {
                negatives[i].resize(numberOfNegatives);
            } else {
                // one additional entry for positive triple
                negatives[i].resize(numberOfNegatives + 1);
            }
        }
        // initialize logging data structures
        losses.resize(maxEpochs);
        times.resize(maxEpochs);
        mrrs.resize(maxEpochs);
    }

    virtual void processBatch(std::vector<int> &indices, int start, int end) {
        // reset gradients
        model->resetGradients();
        // build batch
        for (int pi = 0; pi < end-start; ++pi) {
            // select positive triple
            Triple& positive = trainData->getTriple(indices[start+pi]);

            positives[pi] = &positive;

            // sample negative triples
            sampler->sample(positive, negatives[pi].begin());
            if (includePositiveInNegatives) {
                // add the positive one at the last index
                negatives[pi][numberOfNegatives] = positive;
            }
        }
        // calculate gradients
        lossFunction->gradient(positives, negatives);
        // update parameters
        model->update();
        // call post batch hook
        model->postBatch();
    }

    virtual void preEpoch() {
        lossFunction->reset();
    }

    virtual void postEpoch() {
        lossFunction->printLoss();
    }
};


#endif //THRAX_OPTIMIZER_H
