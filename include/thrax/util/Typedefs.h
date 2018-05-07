//
// Created by martin on 14.02.18.
//

#ifndef THRAX_TYPEDEFS_H
#define THRAX_TYPEDEFS_H

#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <thrax/model/EmbeddingParameterSet.h>
#include <thrax/struct/Gradient.h>
#include <Eigen/Dense>

typedef std::unordered_map<std::pair<int, int>, std::vector<int>, boost::hash<std::pair<int, int> > > Pair2ListMap;
typedef std::unordered_map<std::string, EmbeddingParameterSet> ParameterMap;
typedef std::unordered_map<std::string, Gradient> GradientMap;
typedef std::unordered_map<std::string, VectorXd> VectorMap;


#endif //THRAX_TYPEDEFS_H
