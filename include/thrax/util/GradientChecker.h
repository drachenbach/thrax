//
// Created by martin on 20.02.18.
//

#ifndef THRAX_GRADIENTCHECKER_H
#define THRAX_GRADIENTCHECKER_H


#include <thrax/model/BaseModel.h>
#include <thrax/model/EmbeddingParameterSet.h>
#include <thrax/sampler/Sampler.h>
#include "thrax/struct/Data.h"
#include "Typedefs.h"

/**
 * Checks the implementation of a model's scoring and gradient computation function
 */
class GradientChecker {
public:
    /**
     * Checks if the model computes the correct gradient for its scoring function.
     * Gradient is approximated numerically by changing each embedding by epsilon.
     * Gradient checking passes if error is smaller than threshold delta.
     * @param model
     * @param triple
     * @param epsilon
     * @param delta error threshold
     */
    static void checkGradients(BaseModel* model, Triple& triple, double epsilon=1e-7, double delta=1e-6) {
        // initialize variables
        EmbeddingParameterSet* tmpParameter;
        Gradient* tmpGradient;
        double tmp;
        double scorePlus;
        double scoreMinus;
        double approximationValue;
        double gradientValue;
        double diff;

        double sumDiff = 0;
        double sumGradients = 0;
        double sumApproximations = 0;

        // calculate gradients
        model->gradient(triple, 1.0);
        // subject
        for (std::string parameterName: model->getSubjectEmbeddings()) {
            tmpParameter = &(model->getParameter(parameterName));
            tmpGradient = &(model->getGradient(parameterName));
            for (int i = 0; i < tmpParameter->getEmbeddingDimension(); ++i) {
                tmp = (*tmpParameter)(i, triple.subject);
                // add epsilon
                (*tmpParameter)(i, triple.subject) += epsilon;
                scorePlus = model->score(triple);
                // subtract epsilon
                (*tmpParameter)(i, triple.subject) = tmp - epsilon;
                scoreMinus = model->score(triple);
                // calculate approximation
                approximationValue = (scorePlus - scoreMinus) / (2 * epsilon);
                gradientValue = tmpGradient->get(triple.subject)(i);
                diff = approximationValue - gradientValue;
                // store values
                sumApproximations += approximationValue * approximationValue;
                sumGradients += gradientValue * gradientValue;
                sumDiff += diff * diff;
                // reset to original value
                (*tmpParameter)(i, triple.subject) = tmp;
            }
        }
        // relation
        for (std::string parameterName: model->getRelationEmbeddings()) {
            tmpParameter = &(model->getParameter(parameterName));
            tmpGradient = &(model->getGradient(parameterName));
            for (int i = 0; i < tmpParameter->getEmbeddingDimension(); ++i) {
                tmp = (*tmpParameter)(i, triple.relation);
                // add epsilon
                (*tmpParameter)(i, triple.relation) += epsilon;
                scorePlus = model->score(triple);
                // subtract epsilon
                (*tmpParameter)(i, triple.relation) = tmp - epsilon;
                scoreMinus = model->score(triple);
                // calculate approximation
                approximationValue = (scorePlus - scoreMinus) / (2 * epsilon);
                gradientValue = tmpGradient->get(triple.relation)(i);
                diff = approximationValue - gradientValue;
                // store values
                sumApproximations += approximationValue * approximationValue;
                sumGradients += gradientValue * gradientValue;
                sumDiff += diff * diff;
                // reset to original value
                (*tmpParameter)(i, triple.relation) = tmp;
            }
        }
        // object
        for (std::string parameterName: model->getObjectEmbeddings()) {
            tmpParameter = &(model->getParameter(parameterName));
            tmpGradient = &(model->getGradient(parameterName));
            for (int i = 0; i < tmpParameter->getEmbeddingDimension(); ++i) {
                tmp = (*tmpParameter)(i, triple.object);
                // add epsilon
                (*tmpParameter)(i, triple.object) += epsilon;
                scorePlus = model->score(triple);
                // subtract epsilon
                (*tmpParameter)(i, triple.object) = tmp - epsilon;
                scoreMinus = model->score(triple);
                // calculate approximation
                approximationValue = (scorePlus - scoreMinus) / (2 * epsilon);
                gradientValue = tmpGradient->get(triple.object)(i);
                diff = approximationValue - gradientValue;
                // store values
                sumApproximations += approximationValue * approximationValue;
                sumGradients += gradientValue * gradientValue;
                sumDiff += diff * diff;
                // reset to original value
                (*tmpParameter)(i, triple.object) = tmp;
            }
        }
        double error = sqrt(sumDiff) / (sqrt(sumApproximations) + sqrt(sumGradients));
        if (error < delta) {
            BOOST_LOG_TRIVIAL(info) << "Passed gradient checking, error: " << error;
        } else {
            BOOST_LOG_TRIVIAL(info) << "Failed gradient checking, error: " << error;
        }
    }
};


#endif //THRAX_GRADIENTCHECKER_H
