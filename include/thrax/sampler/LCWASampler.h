//
// Created by martin on 12.02.18.
//

#ifndef THRAX_RANDOMSAMPLER_H
#define THRAX_RANDOMSAMPLER_H

#include <boost/property_tree/ptree.hpp>

#include "Sampler.h"

namespace pt = boost::property_tree;

/**
 * Samples negatives triples according to the closed world assumption i.e. we assume that if we observe a triple for a
 * subject/object-relation pair, we're locally complete and all other triples are incorrect.
 */
class LCWASampler: public Sampler {
public:
    LCWASampler(Data* data, pt::ptree &hyperParameters): Sampler(data, hyperParameters) { }

    /**
     * Randomly chooses either subject or object to corrupt
     * @param positive positive input triple
     * @param negative negative candidate output triple
     */
    virtual void sampleSingle(Triple &positive, Triple &negative, std::string mode) {
        int k;
        int k2;
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


#endif //THRAX_RANDOMSAMPLER_H
