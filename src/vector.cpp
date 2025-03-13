#include "vector.hpp"

DNN::Vector::Vector(int nbRow, float expr, std::shared_ptr<CLMatrixSetup> setup) : Matrix(nbRow, 1, expr, setup) {
    Vector::setCLSetup(CLSetup);
}

DNN::Vector::Vector(const cl::vector<float> &initializer, std::shared_ptr<CLMatrixSetup> setup) : Matrix(initializer.size(), 1, new cl::vector<float>(initializer), setup) {
    Vector::setCLSetup(CLSetup);
}

DNN::Vector::Vector(cl::vector<float> &&initializer, std::shared_ptr<CLMatrixSetup> setup) : Matrix(initializer.size(), 1, new cl::vector<float>((cl::vector<float> &&) initializer), setup) {
    Vector::setCLSetup(CLSetup);
}

DNN::Vector::Vector(Vector &toCopy) : Matrix(toCopy) {}

DNN::Vector::Vector(Vector &&toMove) noexcept : Matrix((Matrix &&) toMove) {}

DNN::Vector::Vector(Matrix &toCopy) : Matrix(toCopy) {
    assert(columns == 1);
    Vector::setCLSetup(CLSetup);
}

DNN::Vector::Vector(Matrix &&toMove) noexcept : Matrix((Matrix &&) toMove) {
    assert(columns == 1);
    Vector::setCLSetup(CLSetup);
}

DNN::Vector DNN::Vector::operator+(Vector &operand) {
    Vector vectorResult(getRowCount(),
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*getRowCount()),
        CLSetup
    );
    opAdd(*this, operand, vectorResult);

    return vectorResult;
}

DNN::Vector DNN::Vector::operator-(Vector &operand) {
    Vector vectorResult(getRowCount(),
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*getRowCount()),
        CLSetup
    );
    opSub(*this, operand, vectorResult);

    return vectorResult;
}

DNN::Vector DNN::Vector::operator-() {
    Vector vectorResult(getRowCount(),
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*getRowCount()),
        CLSetup
    );
    opOpp(*this, vectorResult);

    return vectorResult;
}

DNN::Vector DNN::Vector::hadamardProduct(Vector &operand) {
    Vector vectorResult(getRowCount(),
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*getRowCount()),
        CLSetup
    );
    opHad(*this, operand, vectorResult);

    return vectorResult;
}

DNN::Vector DNN::Vector::executeKernel(cl::KernelFunctor<cl::Buffer &, cl::Buffer &> kernel) {
    assert(isValid());

    //Prepare result (no need for TS behavior, see constructors)
    Vector vectorResult(getRowCount(),
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*getRowCount()),
        CLSetup
    );
    basicUnaryOp(*this, vectorResult, transpose, rows, columns, kernel);

    return vectorResult;
}

DNN::Matrix DNN::Vector::addOverMatrix(Matrix &operand) {
    Matrix matrixResult(operand.getRowCount(), operand.getColumnCount(), //As R.transpose = A.transpose
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*operand.rows*operand.columns), 
        CLSetup
    );
    opAOM(*this, operand, matrixResult);

    return matrixResult;
}

DNN::Matrix DNN::Vector::subOverMatrix(Matrix &operand) {
    Matrix matrixResult(operand.getRowCount(), operand.getColumnCount(), //As R.transpose = A.transpose
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*operand.rows*operand.columns),
        CLSetup
    );
    opSOM(*this, operand, matrixResult);

    return matrixResult;
}

void DNN::Vector::setCLSetup(std::shared_ptr<CLMatrixSetup> newSetup) {
    newSetup->addKernelsFromSource(libFile, 
        {"vector_additionOM", "vector_transAdditionOM", "vector_subtractionOM", "vector_transSubtractionOM"},
        libCode
    );
    Matrix::setCLSetup(newSetup);
}

DNN::Vector::Vector(int nbRow, cl::Buffer *existingBuffer, std::shared_ptr<CLMatrixSetup> setup) : Matrix(nbRow, 1, existingBuffer, setup) {
    Vector::setCLSetup(CLSetup);
}

DNN::Vector::Vector(int nbRow, cl::vector<float> *existingVector, std::shared_ptr<CLMatrixSetup> setup) : Matrix(nbRow, 1, existingVector, setup) {
    Vector::setCLSetup(CLSetup);
}

void DNN::Vector::opAOM(Vector &A, Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount());

    //Prepare result and transposition (no need for TS behavior, see constructors)
    CLMatrixSetup::AddKerType kernel(B.getTranspose() ? A.CLSetup->getKernel("vector_additionOM") : A.CLSetup->getKernel("vector_transAdditionOM"));
    basicBinaryOp(A, B, R, false, B.getRowCount(), B.getColumnCount(), kernel);
}

void DNN::Vector::opSOM(Vector &A, Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount());

    //Prepare result and transposition (no need for TS behavior, see constructors)
    CLMatrixSetup::AddKerType kernel(B.getTranspose() ? A.CLSetup->getKernel("vector_subtractionOM") : A.CLSetup->getKernel("vector_transSubtractionOM"));
    basicBinaryOp(A, B, R, false, B.getRowCount(), B.getColumnCount(), kernel);
}

DNN::Vector DNN::operator*(Matrix &AL, Vector &X) {
    Vector vectorResult(AL.getRowCount(),
        new cl::Buffer (AL.getCLSetup()->getContext(), CL_MEM_READ_WRITE, sizeof(float)*AL.getRowCount()),
        AL.getCLSetup()
    );
    Matrix::opMul(AL, X, vectorResult);

    return vectorResult;
}
