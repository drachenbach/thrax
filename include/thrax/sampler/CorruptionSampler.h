//
// Created by martin on 14.02.18.
//

#ifndef THRAX_CORRUPTIONSAMPLER_H
#define THRAX_CORRUPTIONSAMPLER_H


#include <boost/property_tree/ptree.hpp>

#include <thrax/struct/Data.h>
#include "Sampler.h"

namespace pt = boost::property_tree;

/**
 * Samples negative triples by corrupting either subject or object of a positive one. If object is to be corrupted
 * it will randomly choose an entity that acts as an object for a given relation. Subject corruption analogously.
 * If none is found, fall back to LCWA sampling method.
 */
class CorruptionSampler: public Sampler {
public:
    CorruptionSampler(Data* data, pt::ptree &hyperParameters): Sampler(data, hyperParameters) { }

    virtual void sampleSingle(Triple &positive, Triple &negative, std::string mode) {
        int k;
        bool foundNegative;
        for (int j = 0; j < numberOfRetries; ++j) {
            negative = Triple(positive.subject, positive.relation, positive.object);
            // flip coin to decide if subject or object is corrupted
            if (mode == "subject" || mode == "both") {
                // corrupt subject
                std::vector<int> &list = data->getSubjectsForRelation(positive.relation);
                negative.subject = list[RandomUtil::uniformInt(0, list.size())];
            } if (mode == "object" || mode == "both") {
                // corrupt object
                std::vector<int> &list = data->getObjectsForRelation(positive.relation);
                negative.object = list[RandomUtil::uniformInt(0, list.size())];
            }
            if (!data->hasTriple(negative)) {
                return;
            }
        }
        // corruption didn't work, fall back to LCWASampler
        for (int j = 0; j < numberOfRetries; ++j) {
            negative = Triple(positive.subject, positive.relation, positive.object);
            // flip coin to decide if subject or object is corrupted
            if (mode == "subject" || mode == "both") {
                // corrupt subject
                negative.subject = RandomUtil::uniformInt(0, data->getNumberOfEntities());
            } if (mode == "object" || mode == "both") {
                // corrupt object
                negative.object = RandomUtil::uniformInt(0, data->getNumberOfEntities());
            }
            if (!data->hasTriple(negative)) {
                return;
            }
        }
        // corruption didn't work
        BOOST_LOG_TRIVIAL(info) << "Could not sample negative triple for " << positive << " in " << numberOfRetries << " tries.";
    }
};


#endif //THRAX_CORRUPTIONSAMPLER_H
