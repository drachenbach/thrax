//
// Created by martin on 09.01.18.
//

#ifndef THRAX_EMBEDDINGPARAMETERSET_H
#define THRAX_EMBEDDINGPARAMETERSET_H

#include <vector>

#include <thrax/initializer/Initializer.h>
#include <Eigen/Dense>

using namespace Eigen;

/**
 * Class to hold a parameter set of an embedding model e.g. the relation embeddings.
 */
class EmbeddingParameterSet: public MatrixXd {
public:
    /**
     * Copy constructor
     * @param M
     */
    EmbeddingParameterSet(MatrixXd M): MatrixXd(M) { }

    /**
     * Constructor of EmbeddingParameterSet
     * @param k dimension of one embedding
     * @param m number of embeddings
     */
    EmbeddingParameterSet(int k, int m) : Matrix(k, m) {
    }

    /**
     * Get the total number of parameters
     * @return total number of parameters in this parameter set
     */
    long totalNumberOfParameters() {
        return this->size();
    }

    long getNumberOfEmbeddings() const {
        return this->cols();
    }

    long getEmbeddingDimension() const {
        return this->rows();
    }

//    friend std::ostream& operator<<(std::ostream &stream, const EmbeddingParameterSet &parameterSet) {
//        return stream << "Embedding parameter set \"" // << parameterSet.getName()
//                      << "\" of size (" << parameterSet.getNumberOfEmbeddings() << "," << parameterSet.getEmbeddingDimension() << ")";
//    }

//    /**
//     * Normalizes each embedding to have norm=1
//     */
//    void normalize() {
//        for (int i = 0; i < m; ++i) {
//            row(*this, i) /= norm_2(row(*this, i));
//        }
//    }

//    /**
//     * Dump embeddings matrix to specified file location
//     * @param location
//     */
//    void dump(std::string location) {
//        std::ofstream outf(location);
//        if (outf.is_open()) {
//            outf << this->format(IOFormat(FullPrecision));
//        } else {
//            BOOST_LOG_TRIVIAL(error) << "Could not open file " << location << " to dump data";
//        }
//        outf.close();
//    }

//private:
//    int m; /** number of embeddings */
//    int k; /** dimensionality of a single embedding */
};


#endif //THRAX_EMBEDDINGPARAMETERSET_H
