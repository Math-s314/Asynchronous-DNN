#include "scalar.hpp"

void DNN::Scalar::buildCLSetup() {
    getCLSetup()->addKernelsFromSource(libFile,
        {
            "scalar_additionOM", "scalar_subtractionOM", "scalar_multiplicationOM", "scalar_divisionOM",
            "scalar_division", "scalar_inversion"
        },
        libCode
    );
}

void DNN::Scalar::_addOver(const Scalar &A, const Matrix &B, Matrix &R) {
    BaseMatrix::basicBinaryOp(getBase(A), getBase(B), getBase(R), B.getRowCount(), B.getColumnCount(),"scalar_additionOM");
}

void DNN::Scalar::_subOver(const Scalar &A, const Matrix &B, Matrix &R) {
    BaseMatrix::basicBinaryOp(getBase(A), getBase(B), getBase(R), B.getRowCount(), B.getColumnCount(),"scalar_subtractionOM");
}

void DNN::Scalar::_div(const Scalar &A, const Scalar &B, DNN::Scalar &R) {
    BaseMatrix::basicBinaryOp(getBase(A), getBase(B), getBase(R), 1, 1,"scalar_division");
}

void DNN::Scalar::_inv(const DNN::Scalar &A, DNN::Scalar &R) {
    BaseMatrix::basicUnaryOp(getBase(A), getBase(R), 1, 1,"scalar_inversion");
}

void DNN::Scalar::_mulOver(const DNN::Scalar &A, const DNN::Matrix &B, DNN::Matrix &R) {
    BaseMatrix::basicBinaryOp(getBase(A), getBase(B), getBase(R), B.getRowCount(), B.getColumnCount(),"scalar_multiplicationOM");
}

void DNN::Scalar::_divOver(const DNN::Scalar &A, const DNN::Matrix &B, DNN::Matrix &R) {
    BaseMatrix::basicBinaryOp(getBase(A), getBase(B), getBase(R), B.getRowCount(), B.getColumnCount(),"scalar_divisionOM");
}

DNN::Scalar DNN::inv(const Scalar &A) {
    Scalar scalarResult(A.getCLSetup());
    Scalar::_inv(A, scalarResult);
    return scalarResult;
}

DNN::Matrix DNN::operator+(const Matrix &A, const Scalar &B) {
    Matrix matrixResult(A.getCLSetup());
    Scalar::_addOver(B, A, matrixResult);
    return matrixResult;
}

DNN::Matrix DNN::operator-(const Matrix &A, const Scalar &B) {
    Matrix matrixResult(A.getCLSetup());
    Scalar::_subOver(B, A, matrixResult);
    return matrixResult;
}

DNN::Matrix DNN::operator*(const Matrix &A, const Scalar &X) {
    Matrix matrixResult(A.getCLSetup());
    Scalar::_mulOver(X, A, matrixResult);
    return matrixResult;
}

DNN::Scalar DNN::operator/(const Scalar &A, const Scalar &B) {
    Scalar scalarResult(A.getCLSetup());
    Scalar::_div(A, B, scalarResult);
    return scalarResult;
}

DNN::Matrix DNN::operator/(const Matrix &A, const Scalar &X) {
    Matrix matrixResult(A.getCLSetup());
    Scalar::_divOver(X, A, matrixResult);
    return matrixResult;
}
