//
// Created by martin on 09.02.18.
//

#ifndef THRAX_EVALUATION_H
#define THRAX_EVALUATION_H

#include <Eigen/Dense>
#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

using namespace Eigen;
namespace fs = boost::filesystem;

class Evaluation {
public:
    Evaluation(std::vector<Data*> lookupDataSets = std::vector<Data*>()) : lookupDataSets(lookupDataSets) {}
    /**
     * Performs evaluation for the model (that was trained on trainData) on the data.
     * Dumps results to file found in path.
     * @param model
     * @param data
     */
    void evaluate(AbstractModel* model, Data* data, std::string path) {
        int m = data->getNumberOfTriples();
        int N = data->getNumberOfEntities();
        int numberOfRelations = data->getNumberOfRelations();

        // temporary variables
        double positiveScore;
        VectorXd scores(N);
        int rawRank;
        int filteredRank;

        // rank data structures
        VectorXi allRanksRaw = VectorXi::Zero(2 * m); /** raw ranks by replacing subject and object **/
        VectorXi allRanksFiltered = VectorXi::Zero(2 * m); /** filtered ranks by replacing subject and object **/
        VectorXi subjectRanksRaw = VectorXi::Zero(m); /** raw ranks by replacing subject **/
        VectorXi subjectRanksFiltered = VectorXi::Zero(m); /** filtered ranks by replacing subject **/
        VectorXi objectRanksRaw = VectorXi::Zero(m); /** raw ranks by replacing object **/
        VectorXi objectRanksFiltered = VectorXi::Zero(m); /** filtered ranks by replacing object **/

        std::vector<VectorXi> allRanksRawByRelation(numberOfRelations); /** raw ranks by replacing subject and object by relation**/
        std::vector<VectorXi> allRanksFilteredByRelation(numberOfRelations); /** filtered ranks by replacing subject and object by relation**/
        std::vector<VectorXi> subjectRanksRawByRelation(numberOfRelations); /** raw ranks by replacing subject by relation**/
        std::vector<VectorXi> subjectRanksFilteredByRelation(numberOfRelations); /** filtered ranks by replacing subject by relation**/
        std::vector<VectorXi> objectRanksRawByRelation(numberOfRelations); /** raw ranks by replacing object by relation**/
        std::vector<VectorXi> objectRanksFilteredByRelation(numberOfRelations); /** filtered ranks by replacing object by relation**/
        for (int i = 0; i < numberOfRelations; ++i) {
            allRanksRawByRelation[i] = VectorXi::Zero(2 * m);
            allRanksFilteredByRelation[i] =  VectorXi::Zero(2 * m);
            subjectRanksRawByRelation[i] =  VectorXi::Zero(m);
            subjectRanksFilteredByRelation[i] =  VectorXi::Zero(m);
            objectRanksRawByRelation [i] =  VectorXi::Zero(m);
            objectRanksFilteredByRelation[i] =  VectorXi::Zero(m);
        }

        // timer
        boost::timer::cpu_timer timer;
        // loop over test triples
        for (int i = 0; i < m; ++i) {
            // get test triple
            Triple &triple = data->getTriple(i);
            // calculate positive score
            positiveScore = model->score(triple);

            // calculate scores by replacing object with all entities
            calculateScores(model, triple, scores, false);
            // calculate rank
            calculateRank(triple, positiveScore, scores, rawRank, filteredRank, false);
            allRanksRaw[i] = rawRank;
            allRanksFiltered[i] = filteredRank;
            objectRanksRaw[i] = rawRank;
            objectRanksFiltered[i] = filteredRank;
            allRanksRawByRelation[triple.relation][i] = rawRank;
            allRanksFilteredByRelation[triple.relation][i] = filteredRank;
            objectRanksRawByRelation[triple.relation][i] = rawRank;
            objectRanksFilteredByRelation[triple.relation][i] = filteredRank;

            // calculate scores by replacing object with all entities
            calculateScores(model, triple, scores, true);
            // calculate rank
            calculateRank(triple, positiveScore, scores, rawRank, filteredRank, true);
            allRanksRaw[m+i] = rawRank;
            allRanksFiltered[m+i] = filteredRank;
            subjectRanksRaw[i] = rawRank;
            subjectRanksFiltered[i] = filteredRank;
            allRanksRawByRelation[triple.relation][m+i] = rawRank;
            allRanksFilteredByRelation[triple.relation][m+i] = filteredRank;
            subjectRanksRawByRelation[triple.relation][i] = rawRank;
            subjectRanksFilteredByRelation[triple.relation][i] = filteredRank;
        }
        // calculate metrics
        double meanRank;
        double hitsAtTen;
        double hitsAtOne;
        double meanReciprocalRank;
        int n;

        // dump metrics to file
        fs::path dir(path);
        fs::path metricsPath = dir / "metrics.csv";
        std::ofstream outf(metricsPath.string());
        outf << "MR,hits@10,hits@1,MRR,filtered,target" << std::endl;

        BOOST_LOG_TRIVIAL(info) << "##### RAW SETTING #####";
        calculateMetrics(allRanksRaw, meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
        BOOST_LOG_TRIVIAL(info) << "Combined: MR: " << meanRank << ", MRR: " << meanReciprocalRank << ", hits@10: " << hitsAtTen << ", hits@1: " << hitsAtOne;
        outf << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "false" << "," << "combined" << std::endl;
        calculateMetrics(objectRanksRaw, meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
        BOOST_LOG_TRIVIAL(info) << "Predicting objects: MR: " << meanRank << ", MRR: " << meanReciprocalRank << ", hits@10: " << hitsAtTen << ", hits@1: " << hitsAtOne;
        outf << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "false" << "," << "object" << std::endl;
        calculateMetrics(subjectRanksRaw, meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
        BOOST_LOG_TRIVIAL(info) << "Predicting subjects: MR: " << meanRank << ", MRR: " << meanReciprocalRank << ", hits@10: " << hitsAtTen << ", hits@1: " << hitsAtOne;
        outf << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "false" << "," << "subject" << std::endl;

        BOOST_LOG_TRIVIAL(info) << "##### FILTERED SETTING #####";
        calculateMetrics(allRanksFiltered, meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
        BOOST_LOG_TRIVIAL(info) << "Combined: MR: " << meanRank << ", MRR: " << meanReciprocalRank << ", hits@10: " << hitsAtTen << ", hits@1: " << hitsAtOne;
        outf << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "true" << "," << "combined" << std::endl;
        calculateMetrics(objectRanksFiltered, meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
        BOOST_LOG_TRIVIAL(info) << "Predicting objects: MR: " << meanRank << ", MRR: " << meanReciprocalRank << ", hits@10: " << hitsAtTen << ", hits@1: " << hitsAtOne;
        outf << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "true" << "," << "object" << std::endl;
        calculateMetrics(subjectRanksFiltered, meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
        BOOST_LOG_TRIVIAL(info) << "Predicting subjects: MR: " << meanRank << ", MRR: " << meanReciprocalRank << ", hits@10: " << hitsAtTen << ", hits@1: " << hitsAtOne;
        outf << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "true" << "," << "subject" << std::endl;

        outf.close();

        // dump metrics by relation to file
        fs::path metricsByRelationPath = dir / "metrics-by-relation.csv";
        outf.open(metricsByRelationPath.string());
        outf << "relation,MR,hits@10,hits@1,MRR,filtered,target,n" << std::endl;
        for (int i = 0; i < numberOfRelations; ++i) {
            calculateMetrics(allRanksRawByRelation[i], meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
            outf << i << "," << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "false" << "," << "combined" << "," << n << std::endl;
            calculateMetrics(allRanksFilteredByRelation[i], meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
            outf << i << "," << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "true" << "," << "combined" << "," << n << std::endl;
            calculateMetrics(subjectRanksRawByRelation[i], meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
            outf << i << "," << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "false" << "," << "subject" << "," << n << std::endl;
            calculateMetrics(subjectRanksFilteredByRelation[i], meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
            outf << i << "," << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "true" << "," << "subject" << "," << n << std::endl;
            calculateMetrics(objectRanksRawByRelation[i], meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
            outf << i << "," << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "false" << "," << "object" << "," << n << std::endl;
            calculateMetrics(objectRanksFilteredByRelation[i], meanRank, hitsAtTen, hitsAtOne, meanReciprocalRank, n);
            outf << i << "," << meanRank << "," << hitsAtTen << "," << hitsAtOne << "," << meanReciprocalRank << "," << "true" << "," << "object" << "," << n << std::endl;
        }
        outf.close();

        timer.stop();
        BOOST_LOG_TRIVIAL(info) << "Finished evaluation in " << timer.format(3, "%w sec");
    }

    /**
     * Calculates the mean reciprocal rank that model achieves on data
     * Used for early stopping
     * @param data
     * @param model
     * @param limit number of triples to estimate the performance, -1 means use all
     * @return
     */
    void MRR(AbstractModel* model, Data* data, int limit, double& rawMRR, double& filteredMRR) {
        int m = data->getNumberOfTriples();
        // create vector of indices
        std::vector<int> indices(m);
        std::iota(indices.begin(), indices.end(), 0);

        if (limit > -1) {
            // use only a subset of triples, shuffle indices // TODO can be done more efficiently
            m = limit;
            std::random_shuffle(indices.begin(), indices.end());
        }
        int N = data->getNumberOfEntities();
        VectorXi rawRanks(2 * m);
        VectorXi filteredRanks(2 * m);
        double positiveScore;
        VectorXd scores(N);

        int rawRank;
        int filteredRank;

        // loop over test triples
        for (int i = 0; i < m; ++i) {
            // get test triple
            Triple& triple = data->getTriple(indices[i]);
            // calculate positive score
            positiveScore = model->score(triple);

            // calculate scores by replacing object with all entities
            calculateScores(model, triple, scores, false);
            // calculate rank
            calculateRank(triple, positiveScore, scores, rawRank, filteredRank, false);
            rawRanks[i] = rawRank;
            filteredRanks[i] = filteredRank;

            // calculate scores by replacing object with all entities
            calculateScores(model, triple, scores, true);
            // calculate rank
            calculateRank(triple, positiveScore, scores, rawRank, filteredRank, true);
            rawRanks[m+i] = rawRank;
            filteredRanks[m+i] = filteredRank;
        }
        // calculate metrics
        double meanRank;
        double hitsAtTen;
        double hitsAtOne;
        double meanReciprocalRank;
        int n;

        calculateMetrics(rawRanks, meanRank, hitsAtTen, hitsAtOne, rawMRR, n);
        calculateMetrics(filteredRanks, meanRank, hitsAtTen, hitsAtOne, filteredMRR, n);
    }

    void calculateScores(AbstractModel* model, Triple& triple, VectorXd& scores, bool alterSubject) {
        if (alterSubject) {
            model->scoreRelationObject(triple.relation, triple.object, scores);
        } else {
            model->scoreSubjectRelation(triple.subject, triple.relation, scores);
        }
    }

private:
    std::vector<Data*> lookupDataSets; /** used to check if the existence of triples in the filtered setting **/

    static int rawRank(Triple triple, double& positiveScore, const VectorXd& scores) {
        int rawRank = 1;
        for (int i = 0; i < scores.size(); ++i) {
            // if score is better than the score of the positive triple, increase the raw rank
            if (scores[i] > positiveScore) {
                rawRank++;
            }
        }
        return rawRank;
    }

    /**
     * Calculate the rank
     * @param triple original triple
     * @param positiveScore score of original triple
     * @param scores scores of all triples obtained by either replacing subject or object
     * @param rawRank used to store raw rank
     * @param filteredRank used to store filtered rank
     * @param alterSubject whether subject (true) or object (false) is altered
     */
    void calculateRank(Triple triple, double& positiveScore, const VectorXd& scores,
                       int &rawRank, int &filteredRank, bool alterSubject) {
        rawRank = 1;
        filteredRank = 1;
        for (int i = 0; i < scores.size(); ++i) {
            // if score is better than the score of the positive triple, increase the raw rank
            if (scores[i] > positiveScore) {
                rawRank++;
                // if additionally the triple does not already exist, also increase the filtered rank
                if (alterSubject) {
                    triple.subject = i;
                } else {
                    triple.object = i;
                }
                if (!tripleExistsGlobal(triple)) {
                    filteredRank++;
                }
            }
        }
    }

    /**
     * Calculates various metrics on the given ranks
     * @param ranks triple ranks
     * @param meanRank
     * @param hitsAtTen
     * @param hitsAtOne
     * @param meanReciprocalRank
     */
    void calculateMetrics(const VectorXi& ranks, double& meanRank, double& hitsAtTen, double& hitsAtOne, double& meanReciprocalRank, int& n) {
        n = 0;
        meanRank = 0.0;
        hitsAtTen = 0.0;
        hitsAtOne = 0.0;
        meanReciprocalRank = 0.0;
        for (int i = 0; i < ranks.size(); ++i) {
            // skip 0 ranks
            if (ranks[i] == 0) {
                continue;
            }
            ++n;
            meanRank += ranks[i];
            if (ranks[i] <= 10) {
                hitsAtTen++;
            }
            if (ranks[i] == 1) {
                hitsAtOne++;
            }
            meanReciprocalRank += (1.0 / ranks[i]);
        }
        if (n == 0) {
            // set everything to -1 if no valid rank was found
            meanRank = -1;
            hitsAtTen = -1;
            hitsAtOne = -1;
            meanReciprocalRank = -1;
        } else {
            meanRank /= n;
            hitsAtTen /= n;
            hitsAtOne /= n;
            meanReciprocalRank /= n;
        }
    }

    /**
     * Checks whether a triple exists globally in at least one of the lookupDataSets
     * @param triple
     * @return
     */
    bool tripleExistsGlobal(Triple& triple) {
        for (int i = 0; i < lookupDataSets.size(); ++i) {
            if (lookupDataSets[i]->hasTriple(triple)) {
                return true;
            }
        }
        return false;
    }
};


#endif //THRAX_EVALUATION_H
