//
// Created by martin on 07.02.18.
//

#ifndef THRAX_GRADIENT_H
#define THRAX_GRADIENT_H

#include <unordered_map>

#include <thrax/model/EmbeddingParameterSet.h>
#include <thrax/util/Typedefs.h>

#include <Eigen/Dense>

using namespace Eigen;

/**
 * This class holds the gradients calculated for one batch.
 * It pre-allocates a matrix that stores the gradient values and reuses the memory efficiently.
 */
class Gradient {
public:
    Gradient(EmbeddingParameterSet* parameterSet):
            parameterSet(parameterSet),
            max(parameterSet->getNumberOfEmbeddings()),
            size(0),
            d(parameterSet->getEmbeddingDimension(), max) // TODO this can be smaller than max
    {}

    /**
     * Sets the gradient of embedding i to value v
     * @param i id of embedding
     * @param v value of embedding
     */
    void add(int i, VectorXd v) {
        if (idToCol.find(i) != idToCol.end()) {
            // id already present, add to col
            d.col(idToCol[i]) += v;
            idToCount[i]++;
        } else {
            // id not present, add new col
            d.col(size) = v;
            idToCol[i] = size;
            idToCount[i] = 1;
            size++;
        }
    }

    /**
     * Update the gradient of embedding i with value v
     * @param i embedding id
     * @param v embeddings values
     */
    void setGradient(int i, VectorXd v) {
        d.col(idToCol[i]) = v;
    }

    /**
     * Get the gradient of embedding i
     * @param i embedding id
     * @return
     */
    VectorXd get(int i) {
        return d.col(idToCol[i]);
    }

    /**
     * Resets the gradient matrix by setting the size to 0
     */
    void reset() {
        idToCol.clear();
        idToCount.clear();
        size = 0;
    }

    int getSize() const {
        return size;
    }

    /**
     * Get the first <size> rows of the gradient matrix
     * @return
     */
    MatrixXd getGradients() {
        return d.leftCols(size);
    }

    /**
     * Set the values of the first <size> rows of the gradient matrix
     * @param gradients
     */
    void setGradients(MatrixXd gradients) {
        d.leftCols(size) = gradients;
    }

    std::unordered_map<int, int>& getIdToCol() {
        return idToCol;
    };

    long getEmbeddingDimension() {
        return d.rows();
    }

    /**
     * Applies L2 regularization with hyper parameter lambda to every gradient
     * @param lambda regularization hyper parameter
     */
    void l2(double lambda) {
        for (auto& ptr: idToCol) {
            d.col(ptr.second) +=  2 * lambda * parameterSet->col(ptr.first) * idToCount[ptr.first];
        }
    }

    /**
     * Normalizes the embeddings of the batch
     */
    void normalize() {
        for (auto& ptr: idToCol) {
            parameterSet->col(ptr.first).normalize();
        }
    }

private:
    int size; /** current number of gradients **/
    int max; /** maximum number of embeddings **/
    MatrixXd d; /** gradient values **/
    std::unordered_map<int, int> idToCol; /** hash map that maps embedding ids to rows of the gradient matrix **/
    std::unordered_map<int, int> idToCount; /** hash map that maps embedding ids to counts how often an ID was used in a batch **/
    EmbeddingParameterSet* parameterSet;
};


#endif //THRAX_GRADIENT_H
