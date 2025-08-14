#include "vector.hpp"

void DNN::Vector::buildCLSetup() {
    getCLSetup()->addKernelsFromSource(libFile,
        {"vector_additionOM", "vector_transAdditionOM", "vector_subtractionOM", "vector_transSubtractionOM"},
        libCode
    );
}

void DNN::Vector::_addOver(const Vector &A, const Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount());
    BaseMatrix::basicBinaryOp(getBase(A), getBase(B), getBase(R), B.getRowCount(), B.getColumnCount(),"vector_additionOM");
}

void DNN::Vector::_subOver(const Vector &A, const Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount());
    BaseMatrix::basicBinaryOp(getBase(A), getBase(B), getBase(R), B.getRowCount(), B.getColumnCount(),"vector_subtractionOM");
}
