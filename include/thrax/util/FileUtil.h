//
// Created by martin on 08.01.18.
//

#ifndef THRAX_FILEUTIL_H
#define THRAX_FILEUTIL_H

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <vector>

#include <thrax/struct/Triple.h>
#include <thrax/struct/DataContainer.h>
#include <thrax/util/Typedefs.h>

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;
using namespace Eigen;

/**
 * This namespace is a collection of helpful utilities mainly to handle file I/O
 */
namespace FileUtil {
    /**
     * Loads triple data from a file. Each line is one triple of format <subject><sep><relation><sep><object>
     * @param path path to file
     * @param entityMap map from entity name -> entity id
     * @param relationMap map from relation name -> relation id
     * @param triples vector of triples
     * @param sep separtor, default: tab
     */
    void loadData(const std::string path,
                  std::unordered_map<std::string, int> &entityMap,
                  std::unordered_map<std::string, int> &relationMap,
                  std::vector<Triple> &triples,
                  DataContainer &multiIndex,
                  Pair2ListMap &subjectRelation2Object,
                  Pair2ListMap &relationObject2Subject,
                  std::unordered_map<int, std::vector<int> > &relation2Subject,
                  std::unordered_map<int, std::vector<int> > &relation2Object,
                  int limit,
                  bool ignoreNewConstituents,
                  const std::string sep="\t") {
        std::ifstream file(path);
        if (!file.good()) {
            BOOST_LOG_TRIVIAL(error) << "Cannot find file " << path << std::endl;
            exit(1);
        }

        int numTriples = 0;
        std::string line;
        std::vector<std::string> splits;
        int subjectId, relationId, objectId;
        std::unordered_map<std::string, int>::iterator it;

        while (std::getline(file, line)) {
            boost::trim(line);
            if (line.length() < 1) continue;

            boost::split(splits, line, boost::is_any_of(sep)); // split into subject, relation, object

            for (std::vector<int>::size_type i = 0; i != splits.size(); i++) {
                boost::trim(splits[i]);
            }

            // map subject
            it = entityMap.find(splits[0]);
            if (it == entityMap.end()) {
                if (ignoreNewConstituents) {
                    // ignore triple with non-existent constituent
                    continue;
                }
                // entity not yet mapped, add to map
                subjectId = entityMap.size();
                entityMap[splits[0]] = subjectId;
            } else {
                subjectId = it->second;
            }

            // map relation
            it = relationMap.find(splits[1]);
            if (it == relationMap.end()) {
                if (ignoreNewConstituents) {
                    // ignore triple with non-existent constituent
                    continue;
                }
                // relation not yet mapped, add to map
                relationId = relationMap.size();
                relationMap[splits[1]] = relationId;
            } else {
                relationId = it->second;
            }

            // map object
            it = entityMap.find(splits[2]);
            if (it == entityMap.end()) {
                if (ignoreNewConstituents) {
                    // ignore triple with non-existent constituent
                    continue;
                }
                // entity not yet mapped, add to map
                objectId = entityMap.size();
                entityMap[splits[2]] = objectId;
            } else {
                objectId = it->second;
            }

            // count triples
            ++numTriples;

            // finally add mapped triple
            multiIndex.insert(Triple(subjectId, relationId, objectId));
            triples.push_back(Triple(subjectId, relationId, objectId));
            subjectRelation2Object[std::make_pair(subjectId, relationId)].push_back(objectId);
            relationObject2Subject[std::make_pair(relationId, objectId)].push_back(subjectId);
            relation2Subject[relationId].push_back(subjectId);
            relation2Object[relationId].push_back(objectId);

            BOOST_LOG_TRIVIAL(trace) << "<" << subjectId << "><" << relationId << "><" << objectId << ">, <" << splits[0] << "><" << splits[1] << "><" << splits[2] << ">";

            if (limit > -1 && limit == numTriples) break;
        }

        file.close();
    };

    /**
     * Dumps the transpose of the given matrix M to file (easier to load in column major format)
     * @param path
     * @param M
     * @param sep
     * @param rowSep
     */
    void dumpMatrix(std::string path, MatrixXd& M, std::string sep=",") {
        std::ofstream outf(path);
        if (outf.is_open()) {
            outf << M.transpose().format(IOFormat(FullPrecision, DontAlignCols, sep, "\n", "", "", "", ""));
        } else {
            BOOST_LOG_TRIVIAL(error) << "Could not open file " << path << " to dump matrix";
        }
        outf.close();
    }

    /**
     * Load matrix M from path
     * @param path
     * @param M
     * @param sep
     * @param rowSep
     */
    void loadMatrix(std::string path, MatrixXd& M, std::string sep=",") {
        std::ifstream inf(path);
        std::string line;
        std::vector<double> values;
        std::vector<std::string> rowValues;
        long cols = 0;
        while(std::getline(inf, line)) {
            boost::split(rowValues, line, boost::is_any_of(sep));
            for (int i = 0; i < rowValues.size(); ++i) {
                values.push_back(std::stod(rowValues[i]));
            }
            ++cols;
        }
        M = Map<MatrixXd>(values.data(), values.size()/cols, cols);
    }

    /**
     * Dump a map to given path
     * @param path
     * @param map
     * @param sep
     */
    void dumpMap(std::string path, std::unordered_map<std::string, int>& map, std::string sep=",") {
        std::ofstream outf(path);
        if (outf.is_open()) {
            for (auto& kv: map) {
                outf << kv.first << sep << kv.second << std::endl;
            }
        } else {
            BOOST_LOG_TRIVIAL(error) << "Could not open file " << path << " to dump map";
        }
        outf.close();
    }

    void loadMap(std::string path, std::unordered_map<std::string, int>& map, std::string sep=",") {
        std::ifstream inf(path);
        std::string line;
        std::vector<std::string> splits;
        while (std::getline(inf, line)) {
            boost::trim(line);
            boost::split(splits, line, boost::is_any_of(sep)); // split into key, value
            map[splits[0]] = std::stoi(splits[1]);
        }
    }

    void dumpConfig(std::string path, pt::ptree config) {
        // dump config
        fs::path dir(path);
        fs::create_directories(dir);
        fs::path configPath = dir / "config.json";
        pt::write_json(configPath.string(), config);
    }

    void dumpTrainingStatistics(std::string path, std::vector<double>& losses, std::vector<double>& times, std::vector<double>& mrrs, int& epochsTrained) {
        fs::path dir(path);
        fs::create_directories(dir);
        fs::path statsPath = dir / "stats.csv";
        std::ofstream outf(statsPath.string());
        if (outf.is_open()) {
            outf << "epoch,loss,time,mrr" << std::endl;
            for (int i = 0; i < epochsTrained; ++i) {
                outf << i << "," << losses[i] << "," << times[i] << "," << mrrs[i] << std::endl;
            }
        } else {
            BOOST_LOG_TRIVIAL(error) << "Could not open file " << path << " to dump training statistics";
        }
        outf.close();
    }
};


#endif //THRAX_FILEUTIL_H
