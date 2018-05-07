//
// Created by martin on 08.01.18.
//

#ifndef THRAX_TRIPLE_H
#define THRAX_TRIPLE_H

struct Triple{
public:
    int subject;
    int relation;
    int object;

    Triple(){};
    Triple(int subject, int relation, int object):subject(subject), relation(relation), object(object) {}

    bool operator==(const Triple &b) const {
        return (this->subject == b.subject) && (this->relation == b.relation) && (this->object == b.object);
    }

    friend std::ostream& operator<<(std::ostream &stream, const Triple &triple) {
        return stream << "<" << triple.subject << "," <<  triple.relation << "," << triple.object << ">";
    }
};

#endif //THRAX_TRIPLE_H
