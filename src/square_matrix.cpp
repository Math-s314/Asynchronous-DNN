#include "square_matrix.hpp"

DNN::SquareMatrix::SquareMatrix(int N, float expr, std::shared_ptr<CLMatrixSetup> setup) : Matrix(N, N, expr, setup) {
    SquareMatrix::setCLSetup(CLSetup);
}

DNN::SquareMatrix::SquareMatrix(const cl::vector<cl::vector<float>> &initializer, bool transposed, std::shared_ptr<CLMatrixSetup> setup) : Matrix(initializer, transposed, setup){
    SquareMatrix::setCLSetup(CLSetup);
}

DNN::SquareMatrix::SquareMatrix(SquareMatrix &toCopy) : Matrix(toCopy) {}

DNN::SquareMatrix::SquareMatrix(SquareMatrix &&toMove) noexcept : Matrix((Matrix &&) toMove) {}

DNN::SquareMatrix::SquareMatrix(Matrix &toCopy) : Matrix(toCopy) {
    assert(columns == rows);
    SquareMatrix::setCLSetup(CLSetup);
}

DNN::SquareMatrix::SquareMatrix(Matrix &&toMove) noexcept : Matrix(toMove) {
    assert(columns == rows);
    SquareMatrix::setCLSetup(CLSetup);
}

DNN::SquareMatrix DNN::SquareMatrix::operator+(const SquareMatrix &operand) const {
    SquareMatrix matrixResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opAdd(*this, operand, matrixResult);

    return matrixResult;
}

DNN::SquareMatrix DNN::SquareMatrix::operator-(const SquareMatrix &operand) const {
    SquareMatrix matrixResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opSub(*this, operand, matrixResult);

    return matrixResult;
}

DNN::SquareMatrix DNN::SquareMatrix::operator*(const SquareMatrix &operand) const {
    SquareMatrix matrixResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opMul(*this, operand, matrixResult);

    return matrixResult;
}

DNN::SquareMatrix DNN::SquareMatrix::operator^(int exp) const {
    SquareMatrix matrixResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );

    if(exp >= 0) {
        opPow(*this, (unsigned int) exp, matrixResult);
    }
    else {

    }
    
    return matrixResult;
    
}

DNN::SquareMatrix DNN::SquareMatrix::operator-() const {
    SquareMatrix matrixResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opOpp(*this, matrixResult);

    return matrixResult;
}

DNN::SquareMatrix DNN::SquareMatrix::hadamardProduct(const SquareMatrix &operand) const {
    SquareMatrix matrixResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opHad(*this, operand, matrixResult);

    return matrixResult;
}

DNN::SquareMatrix DNN::SquareMatrix::executeKernel(cl::KernelFunctor<cl::Buffer &, cl::Buffer &> kernel) const {
    assert(isValid());

    //Prepare result (no need for TS behavior, see constructors)
    SquareMatrix matrixResul(rows, 
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    basicUnaryOp(*this, matrixResul, transpose, rows, columns, kernel);

    return matrixResul;
}

DNN::SquareMatrix DNN::SquareMatrix::identity(int N, std::shared_ptr<CLMatrixSetup> setup) {
    SquareMatrix identity(N, 0.f, setup);
    for(int i = 0; i < N; ++i)
        identity[i][i] = 1.f;
    return identity;
}

DNN::SquareMatrix::SquareMatrix(int N, cl::Buffer *existingBuffer, std::shared_ptr<CLMatrixSetup> setup) : Matrix(N, N, existingBuffer, setup) {
    SquareMatrix::setCLSetup(CLSetup);
}

DNN::SquareMatrix::SquareMatrix(int N, cl::vector<float> *existingVector, std::shared_ptr<CLMatrixSetup> setup) : Matrix(N, N, existingVector, setup) {
    SquareMatrix::setCLSetup(CLSetup);
}

void DNN::SquareMatrix::opPow(const SquareMatrix &A, unsigned int exp, SquareMatrix &R) {
    //We treat small case specifically to avoid useless copies...
    if(exp == 0) { R = identity(A.rows, A.CLSetup); return; }
    if(exp == 1) { R = A; return; }

    SquareMatrix *result[2] = {new SquareMatrix(A.rows, (cl::Buffer *) nullptr, A.CLSetup), 
        &R
    };
    SquareMatrix power[2] = {{A.rows,
        new cl::Buffer (A.CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*A.rows*A.columns),
        A.CLSetup
    }, {A.rows,
        new cl::Buffer (A.CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*A.rows*A.columns),
        A.CLSetup
    }};
    uint8_t resultParity = 0, powerParity  = 0;
    bool resultFirst  = true;

    if(exp % 2) {
        *result[0] = A; //ENH : How to remove this copy ?
        resultFirst = false;
    }
    opMul(A, A, power[powerParity]);
    exp >>= 1;

    while(exp > 0) {
        if(exp % 2) {
            if(resultFirst) {
                *result[0] = power[powerParity]; //ENH : How to remove this copy ?
                resultFirst = false;
            }
            else {
                opMul(power[powerParity], *result[resultParity], *result[resultParity^1]);
                resultParity ^= 1;

                result[resultParity]->promptStateMutex.lock();
                power[powerParity].TS_lastUploadEvent = result[resultParity]->TS_lastComputationEvent; 
                //No upload ever happen on power so the event can be used to note the previous reading event...
                result[resultParity]->promptStateMutex.unlock();
            }
        }
        opMul(power[powerParity], power[powerParity], power[powerParity^1]);
        powerParity ^= 1;
        exp >>= 1;
    }

    R = *result[resultParity];
    delete result[0];
}

void DNN::SquareMatrix::setCLSetup(std::shared_ptr<CLMatrixSetup> newSetup) {
    newSetup->addKernelsFromSource(libFile, 
        {},
        libCode
    );
    Matrix::setCLSetup(newSetup);
}
