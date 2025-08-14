#include "matrix.hpp"

const cl::string DNN::VectorisedFunction::preKernelStr = 
    "kernel void main(global float *A, global float *R) {"
            "const int N = get_global_size(1);"
            "const int i = get_global_id(0);"
            "const int j = get_global_id(1);"
            "R[i*N + j] = ";
const cl::string DNN::VectorisedFunction::postKernelStr = ";}";
const cl::string DNN::VectorisedFunction::indicator = "$x";

float &DNN::RowAccesser::operator[](cl::size_type col) {
    return linkedMatrix.getLValueElement(row, col);
}

DNN::VectorisedFunction::VectorisedFunction(const cl::string &operation, std::weak_ptr<CLMatrixSetup> _setup) : setup(_setup), 
    kernel(cl::Program(setup.lock()->getContext(), cl::util::read_text_file("matrix.ocl") , true ), "main") { }

DNN::Matrix DNN::VectorisedFunction::operator()(const Matrix &arg) const {
    return arg.executeKernel(kernel);
}

cl::string DNN::VectorisedFunction::prepareString(const cl::string &operation) {
    size_t pos = 0;
    cl::string parsed = operation;

    while ((pos = parsed.find("$x")) != std::string::npos) {
        parsed.replace(pos, indicator.length(), "A[i*N+j]");
        pos += 8;
    }
    
    return preKernelStr + parsed + postKernelStr;
}


/// BaseMatrix class definitions

void DNN::Matrix::buildCLSetup() {
    getCLSetup()->addKernelsFromSource(libFile,
        {
            "matrix_addition", "matrix_transAddition",
            "matrix_subtraction", "matrix_transSubtraction",
            "matrix_product", "matrix_transLProduct", "matrix_transRProduct",
            "matrix_hadamard", "matrix_transHadamard",
            "matrix_opposite"
        },
        libCode
    );
}

// Operations' library

void DNN::Matrix::_add(const Matrix &A, const Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount() && A.getColumnCount() == B.getColumnCount());
    BaseMatrix::basicBinaryOp(A.internalMatrix, B.internalMatrix, R.internalMatrix, A.getRowCount(), A.getColumnCount(), "matrix_addition");
}

void DNN::Matrix::_sub(const Matrix &A, const Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount() && A.getColumnCount() == B.getColumnCount());
    BaseMatrix::basicBinaryOp(A.internalMatrix, B.internalMatrix, R.internalMatrix, A.getRowCount(), A.getColumnCount(), "matrix_subtraction");
}

void DNN::Matrix::_mul(const Matrix &A, const Matrix &B, Matrix &R) {
    assert(A.getColumnCount() == B.getRowCount());
    BaseMatrix::basicBinaryOp(A.internalMatrix, B.internalMatrix, R.internalMatrix, A.getRowCount(), B.getColumnCount(), "matrix_product", B.getRowCount());
}

void DNN::Matrix::_hadamard(const Matrix &A, const Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount() && A.getColumnCount() == B.getColumnCount());
    BaseMatrix::basicBinaryOp(A.internalMatrix, B.internalMatrix, R.internalMatrix, A.getRowCount(), A.getColumnCount(), "matrix_hadamard");
}

void DNN::Matrix::_neg(const Matrix &A, Matrix &R) {
    BaseMatrix::basicUnaryOp(A.internalMatrix, R.internalMatrix, A.getRowCount(), A.getColumnCount(), "matrix_opposite");
}

// Operators

DNN::Matrix DNN::operator+(const Matrix &A, const Matrix &B) {
    Matrix matrixResult(A.getCLSetup());
    Matrix::_add(A, B, matrixResult);
    return matrixResult;
}

DNN::Matrix DNN::operator-(const Matrix &A, const Matrix &B) {
    Matrix matrixResult(A.getCLSetup());
    Matrix::_sub(A, B, matrixResult);
    return matrixResult;
}

DNN::Matrix DNN::operator*(const Matrix &A, const Matrix &B) {
    Matrix matrixResult(A.getCLSetup());
    Matrix::_mul(A, B, matrixResult);
    return matrixResult;
}

DNN::Matrix DNN::operator-(const Matrix &A) {
    Matrix matrixResult(A.getCLSetup());
    Matrix::_neg(A, matrixResult);
    return matrixResult;
}

DNN::Matrix DNN::hadamardProduct(const Matrix &A, const Matrix &B) {
    Matrix matrixResult(A.getCLSetup());
    Matrix::_hadamard(A, B, matrixResult);
    return matrixResult;
}

std::ostream &DNN::operator<<(std::ostream &output, DNN::Matrix &matrix) {
    const char fill = output.fill();
    const std::streamsize precision = output.precision();
    const std::streamsize width = output.width() > precision ? output.width() : precision + 5;

    matrix.waitForConstResults();

    for(int i = 0; i < matrix.getRowCount(); ++i) {
    for(int j = 0; j < matrix.getColumnCount(); ++j) {
        if(j != 0) output << ' ';
        output << std::setw(width) << std::setfill(fill)
            << std::fixed << std::showpoint << std::setprecision(precision)
            << matrix.getRValueElement(i, j);
        }
        output << '\n';
    }
    output << std::endl;
    output.width(0);

    return output;
}
