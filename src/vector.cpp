#include "vector.hpp"

DNN::Vector::Vector(int nbRow, float expr) : Matrix(nbRow, 1, expr) {}

DNN::Vector::Vector(const cl::vector<float> &initialiser) : Matrix(initialiser.size(), 1, new cl::vector<float>(initialiser)) {}

DNN::Vector::Vector(cl::vector<float> &&initialiser) : Matrix(initialiser.size(), 1, new cl::vector<float>((cl::vector<float> &&) initialiser)) {}

DNN::Vector::Vector(Vector &toCopy) : Matrix(toCopy) {}

DNN::Vector::Vector(Vector &&toMove) noexcept : Matrix((Matrix &&) toMove) {}

DNN::Vector::Vector(Matrix &toCopy) : Matrix(toCopy) {
    assert(columns == 1);
}

DNN::Vector::Vector(Matrix &&toMove) noexcept : Matrix((Matrix &&) toMove) {
    assert(columns == 1);
}

DNN::Vector DNN::Vector::operator+(Vector &operand) {
    Vector vectorResult(rows,
        new cl::Buffer (CLSetup->context, CL_MEM_READ_WRITE, sizeof(float)*rows)
    );
    opAdd(*this, operand, vectorResult);

    return vectorResult;
}

DNN::Vector DNN::Vector::operator-(Vector &operand) {
    Vector vectorResult(rows,
        new cl::Buffer (CLSetup->context, CL_MEM_READ_WRITE, sizeof(float)*rows)
    );
    opSub(*this, operand, vectorResult);

    return vectorResult;
}

DNN::Vector DNN::Vector::operator-() {
    Vector vectorResult(rows,
        new cl::Buffer (CLSetup->context, CL_MEM_READ_WRITE, sizeof(float)*rows)
    );
    opOpp(*this, vectorResult);

    return vectorResult;
}

DNN::Vector::Vector(int nbRow, cl::Buffer *existingBuffer) : Matrix(nbRow, 1, existingBuffer) {}

DNN::Vector::Vector(int nbRow, cl::vector<float> *existingVector) : Matrix(nbRow, 1, existingVector) {}

DNN::Vector DNN::operator*(Matrix &AL, Vector &X) {
    Vector vectorResult(AL.getRowCount(),
        new cl::Buffer (X.CLSetup->context, CL_MEM_READ_WRITE, sizeof(float)*AL.getRowCount()) //ENH : Setup is not usual !!!
    );
    Matrix::opMul(AL, X, vectorResult);

    return vectorResult;
}
