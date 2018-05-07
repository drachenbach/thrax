//
// Created by martin on 15.01.18.
//

#ifndef THRAX_DATACONTAINER_H
#define THRAX_DATACONTAINER_H

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>

#include <thrax/struct/Triple.h>

using namespace boost;
using namespace boost::multi_index;

struct byAll {};

typedef multi_index_container< // multi index
            Triple, // on triples
            indexed_by<
                hashed_unique< // using hashed index
                    tag<byAll>,
                    composite_key< // on a composite key of subject, relation, and object
                        Triple,
                        member<Triple,int,&Triple::subject>,
                        member<Triple,int,&Triple::relation>,
                        member<Triple,int,&Triple::object>
                    >
                >
            >
        > DataContainer;

#endif //THRAX_DATACONTAINER_H
