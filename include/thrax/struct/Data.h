//
// Created by martin on 08.01.18.
//

#ifndef THRAX_DATA_H
#define THRAX_DATA_H

#include <iostream>
#include <unordered_map>
#include <tuple>
#include <string>
#include <vector>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/functional/hash.hpp>
#include <boost/filesystem.hpp>

#include <thrax/util/FileUtil.h>
#include <thrax/struct/DataContainer.h>
#include <thrax/struct/Triple.h>
#include <thrax/util/Typedefs.h>

using namespace boost;
using namespace boost::multi_index;
namespace fs = boost::filesystem;

/**
 * Class to access knowledge base triples.
 */
class Data {
public:
    /**
     * Default constructor
     */
    Data() {}

    /**
     * Sets the internal maps to map from string representations to unique IDs.
     * Used to use the same maps in train, validation and test data.
     * @param entityMap
     * @param relationMap
     */
    void setMaps(std::unordered_map<std::string, int> entityMap, std::unordered_map<std::string, int> relationMap) {
        entity2id = entityMap;
        relation2id = relationMap;
    }

    /**
     * Loads the first limit lines of the dataset from a file at path
     * @param path
     * @param limit
     */
    void load(const std::string &path, bool ignoreNewConstituents, int limit=-1) {
        BOOST_LOG_TRIVIAL(info) << "Loading triples from " << path;
        // load data from file
        loadData(path, ignoreNewConstituents, limit);
        init();
        BOOST_LOG_TRIVIAL(info) << "Number of triples: " << numberOfTriples;
        BOOST_LOG_TRIVIAL(info) << "Number of entities: " << N;
        BOOST_LOG_TRIVIAL(info) << "Number of relations: " << K;
    }

    int getNumberOfTriples() const {
        return numberOfTriples;
    }

    int getNumberOfEntities() const {
        return N;
    }

    int getNumberOfRelations() const {
        return K;
    }

    Triple& getTriple(int i) {
        return triples[i];
    }

    std::vector<Triple>& getTriples() {
        return triples;
    }

    /**
     * Append the triples of data to the internal data structures.
     * Assumes that both triple lists are mapped identically.
     * @param newTriples
     */
    void addTriples(Data& data) {
        BOOST_LOG_TRIVIAL(info) << "Adding " << data.getNumberOfTriples() << " triples";
        triples.insert(triples.end(), data.getTriples().begin(), data.getTriples().end());
        multiIndex.insert(data.getTriples().begin(), data.getTriples().end());
        for (auto& kv: data.getSubjectsForRelation()) {
            relation2Subject[kv.first].insert(relation2Subject[kv.first].begin(), kv.second.begin(), kv.second.end());
        }
        for (auto& kv: data.getObjectsForRelation()) {
            relation2Object[kv.first].insert(relation2Object[kv.first].begin(), kv.second.begin(), kv.second.end());
        }
        for (auto& kv: data.getSubjectsForRelationObject()) {
            subjectRelation2Object[kv.first].insert(subjectRelation2Object[kv.first].begin(), kv.second.begin(), kv.second.end());
        }
        for (auto& kv: data.getObjectsForSubjectRelation()) {
            relationObject2Subject[kv.first].insert(relationObject2Subject[kv.first].begin(), kv.second.begin(), kv.second.end());
        }
        init();
        BOOST_LOG_TRIVIAL(info) << "Number of triples: " << numberOfTriples;
        BOOST_LOG_TRIVIAL(info) << "Number of entities: " << N;
        BOOST_LOG_TRIVIAL(info) << "Number of relations: " << K;
    }

    std::unordered_map<std::string, int> &getEntityMap() {
        return entity2id;
    }

    std::unordered_map<std::string, int> &getRelationMap() {
        return relation2id;
    }

    const std::vector<int>& getEntities() const {
        return entities;
    }

    const std::vector<int>& getRelations() const {
        return relations;
    }

    std::vector<int>& getObjectsForSubjectRelation(int subjectId, int relationId) {
        return subjectRelation2Object[std::make_pair(subjectId, relationId)];
    }

    Pair2ListMap& getObjectsForSubjectRelation() {
        return subjectRelation2Object;
    }

    std::vector<int>& getSubjectsForRelationObject(int relationId, int objectId) {
        return relationObject2Subject[std::make_pair(relationId, objectId)];
    }

    Pair2ListMap& getSubjectsForRelationObject() {
        return relationObject2Subject;
    }

    std::vector<int>& getSubjectsForRelation(int relationId) {
        return relation2Subject[relationId];
    }

    std::unordered_map<int, std::vector<int> > getSubjectsForRelation() {
        return relation2Subject;
    };

    std::vector<int>& getObjectsForRelation(int relationId) {
        return relation2Object[relationId];
    }

    std::unordered_map<int, std::vector<int> >& getObjectsForRelation() {
        return relation2Object;
    }

    /**
     * Checks if given triple already exists in the knowledge base
     * @param triple
     * @return
     */
    bool hasTriple(Triple &triple) {
        return multiIndex.get<byAll>().find(std::make_tuple(triple.subject, triple.relation, triple.object)) != multiIndex.get<byAll>().end();
    }

    /**
     * Dumps the mappings to files
     * @param path
     */
    void dumpMappings(std::string path) {
        fs::path dir(path);
        fs::create_directories(dir);
        BOOST_LOG_TRIVIAL(info) << "Dumping mappings to " << dir.string();
        // entities
        fs::path entityPath = dir / "entityMappings.csv";
        FileUtil::dumpMap(entityPath.string(), entity2id);
        // relations
        fs::path relationPath = dir / "relationMappings.csv";
        FileUtil::dumpMap(relationPath.string(), relation2id);
    }

    void loadMappings(std::string path) {
        fs::path dir(path);
        BOOST_LOG_TRIVIAL(info) << "Loading mappings from " << dir.string();
        // entities
        fs::path entityPath = dir / "entityMappings.csv";
        FileUtil::loadMap(entityPath.string(), entity2id);
        // relations
        fs::path relationPath = dir / "relationMappings.csv";
        FileUtil::loadMap(relationPath.string(), relation2id);
    }

private:
    int numberOfTriples; /** number of triples in data */
    int N; /** number of entities */
    int K; /** number of relations */

    std::unordered_map<std::string, int> entity2id; /** map of entity name -> entity id */
    std::unordered_map<std::string, int> relation2id; /** map of relation name -> relation id */

    // TODO how to handle these indices for validation/test data
    Pair2ListMap subjectRelation2Object; /** map of (subject, relation) -> list of objects **/
    Pair2ListMap relationObject2Subject; /** map of (relation, object) -> list of subjects **/

    std::unordered_map<int, std::vector<int> > relation2Subject; /** map from relation -> list of subjects **/
    std::unordered_map<int, std::vector<int> > relation2Object; /** map from relation -> list of objects **/

    std::vector<int> entities; /** list of unique entity ids **/
    std::vector<int> relations; /** list of unique relation ids **/

    std::vector<Triple> triples; /** vector of triples */

    DataContainer multiIndex; /** multi index structure for queries */

    void init() {
        // gather statistics
        numberOfTriples = triples.size();
        N = entity2id.size();
        K = relation2id.size();

        // build entity and relation lists
        entities.clear();
        entities.reserve(N);
        relations.clear();
        relations.reserve(K);
        for (auto kv: entity2id) {
            entities.push_back(kv.second);
        }
        for (auto kv: relation2id) {
            relations.push_back(kv.second);
        }
    }

    /**
     * Load data from files.
     * @param path path to file
     */
    void loadData(const std::string &path, bool ignoreNewConstituents, int limit){
        FileUtil::loadData(path, entity2id, relation2id, triples, multiIndex, subjectRelation2Object, relationObject2Subject, relation2Subject, relation2Object, limit, ignoreNewConstituents);
    };
};

#endif //THRAX_DATA_H
