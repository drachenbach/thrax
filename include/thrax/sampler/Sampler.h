//
// Created by martin on 09.01.18.
//

#ifndef THRAX_SAMPLER_H
#define THRAX_SAMPLER_H


#include <vector>

#include <boost/property_tree/ptree.hpp>

#include <thrax/struct/Triple.h>
#include <thrax/util/RandomUtil.h>

namespace pt = boost::property_tree;

class Sampler {
public:
    Sampler(Data* data, pt::ptree &hyperParameters): data(data), hyperParameters(hyperParameters) {
        numberOfNegatives = hyperParameters.get<int>("numberOfNegatives", 1);
        numberOfRetries = hyperParameters.get<int>("numberOfRetries", 10);
        mode = hyperParameters.get<std::string>("mode", "random");
    }

    void sample(Triple &positive, std::vector<Triple>::iterator negativeIterator) {
        sample(positive, negativeIterator, mode);
    }
    
    /**
     * For a given positive triple, sample numberOfNegatives negative triples according to the given strategy
     * @param positive positive input triple
     * @param negatives numberOfNegatives negative output triples
     */
    void sample(Triple &positive, std::vector<Triple>::iterator negativeIterator, std::string mode) {
        std::string tmp = mode;
        // sample numberOfNegatives negative triples
        for (int i = 0; i < numberOfNegatives; ++i, ++negativeIterator) {
            // determine mode
            if (tmp == "random") {
                // randomly choose subject or object
                if (RandomUtil::uniformInt(0, 2) > 0) {
                    mode = "subject";
                } else {
                    mode = "object";
                }
            }
            sampleSingle(positive, *negativeIterator, mode);
        }
    }

protected:
    pt::ptree hyperParameters;
    int numberOfNegatives; /** number of negative triples that should be sampled for each positive triple */
    int numberOfRetries; /** number of retries for perturbation */
    std::string mode; /** mode to perturb: random (randomly choose between corrupting subject or object), subject (corrupt only subjects), object (corrupt only objects) */ // TODO  add both (corrupt once subject, once object)
    Data* data;
    Triple negative;

    /**
     * To be implemented by subclasses
     * @param positive
     * @param negative
     */
    virtual void sampleSingle(Triple &positive, Triple &negative, std::string mode) { }
};


#endif //THRAX_SAMPLER_H
