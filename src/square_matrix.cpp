#include "square_matrix.hpp"

DNN::SquareMatrix::SquareMatrix(int N, float expr, std::shared_ptr<CLMatrixSetup> setup) : Matrix(N, N, expr, setup) {
    SquareMatrix::setCLSetup(CLSetup);
}

DNN::SquareMatrix::SquareMatrix(const cl::vector<cl::vector<float>> &initializer, bool transposed, std::shared_ptr<CLMatrixSetup> setup) : Matrix(initializer, transposed, setup){
    SquareMatrix::setCLSetup(CLSetup);
}

DNN::SquareMatrix::SquareMatrix(const SquareMatrix &toCopy) : Matrix(toCopy) {}

DNN::SquareMatrix::SquareMatrix(SquareMatrix &&toMove) noexcept : Matrix((Matrix &&) toMove) {}

DNN::SquareMatrix::SquareMatrix(const Matrix &toCopy) : Matrix(toCopy) {
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
        opInv(*this, matrixResult);
        opPow(matrixResult, (unsigned int) -exp, matrixResult);
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

DNN::SquareMatrix DNN::SquareMatrix::IDENTITY(int N, std::shared_ptr<CLMatrixSetup> setup) {
    return SCALAR(N, 1.f, setup);
}

DNN::SquareMatrix DNN::SquareMatrix::SCALAR(int N, float lambda, std::shared_ptr<CLMatrixSetup> setup) {
    SquareMatrix identity(N, 0.f, setup);
    for(int i = 0; i < N; ++i) identity[i][i] = lambda;
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
    if(exp == 0) { R = IDENTITY(A.rows, A.CLSetup); return; }
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

void DNN::SquareMatrix::opInv(const DNN::SquareMatrix &A, DNN::SquareMatrix &R) {
    SquareMatrix T(A); //Absolutely needed as A must be modified
    std::cout << T;
    T.downloadData();

    //Basc variables preparation...
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int> kernelCheck(T.CLSetup->getKernel("square_inversionStepCheck"));
    cl::KernelFunctor<cl::Buffer, cl::Buffer> kernelPivoting(T.CLSetup->getKernel("square_rowTransposition"));
    cl::KernelFunctor<cl::Buffer, int, int> kernelMain(T.CLSetup->getKernel("square_inversionStepMain"));
    cl::KernelFunctor<cl::Buffer, int, int> kernelRow(T.CLSetup->getKernel("square_inversionStepRow"));
    cl::KernelFunctor<cl::Buffer, int, int> kernelCol(T.CLSetup->getKernel("square_inversionStepCol"));
    cl::KernelFunctor<cl::Buffer, int, int> kernelFinal(T.CLSetup->getKernel("square_inversionStepFinal"));

    cl::vector<cl::Event> eventsFst;
    cl::vector<cl::Event> eventsSnd;
    eventsFst.reserve(4);
    eventsSnd.reserve(4);

    cl::CommandQueue queue = T.CLSetup->getQueue();

    const int N = T.rows;
    cl::Buffer permutation(T.CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(int)*N);
    cl::Buffer transposition(T.CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(int)*2);

    cl::vector<int> preparation(N, 0);
    for(int j = 0; j < N; j++) preparation[j] = j;

    T.CLSetup->getQueue().enqueueWriteBuffer(
            permutation, true, 0, sizeof(int)*N,
            (void *) preparation.data(), nullptr,
            nullptr //In this function the previous cl_event is correctly released...
    );

    for(int step = 0; step < A.rows; step++) {
        //Prepare events
        T.data->addBufferEvent();
        T.manageBeforeComputation(eventsFst, true); //WARNING : It locks the promptMutex !!!!

        //Prepare result
        T.TS_stateFlags &= oppFlag(StateFlags::INTERNAL_FLAGS);
        T.TS_stateFlags |= StateFlags::COMPUTATION_EXECUTING | StateFlags::DATA_DOWNLOADED;

        /// Actual computations...

        ///Pre-kernels : Check and pivoting
        eventsSnd.push_back(kernelCheck(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(1)),
                *T.data->TS_buffer,
                transposition,
                permutation,
                N, step
        ));
        eventsFst.clear();

        eventsFst.push_back(kernelPivoting(
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(N)),
                *T.data->TS_buffer,
                transposition
        ));
        eventsSnd.clear();

        /// First kernel : row of the current step
        eventsSnd.push_back(kernelRow(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(step, 0), cl::NDRange(1, step), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        ));
        eventsSnd.push_back(kernelRow(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(step, step + 1), cl::NDRange(1, N - step - 1), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        ));
        eventsFst.clear();

        /// Second kernel : elimination in all rows except the current
        eventsFst.push_back(kernelMain( //Update to B_00
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(0, 0), cl::NDRange(step, step), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        ));
        eventsFst.push_back(kernelMain( //Update to B_20
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(step + 1, 0), cl::NDRange(N - step - 1, step), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        ));
        eventsFst.push_back(kernelMain( //Update to A_02
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(0, step + 1), cl::NDRange(step, N - step - 1), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        ));
        eventsFst.push_back(kernelMain( //Update to A_22
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(step + 1, step + 1), cl::NDRange(N - step - 1, N - step - 1), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        ));
        eventsSnd.clear();

        /// Third kernel : columns of the current step
        eventsSnd.push_back(kernelCol(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(0, step), cl::NDRange(step, 1), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        ));
        eventsSnd.push_back(kernelCol(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(step+1, step), cl::NDRange(N - step - 1, 1), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        ));
        eventsFst.clear();

        /// Fourth kernel : central element of the step
        T.TS_lastComputationEvent = kernelFinal( //Here the previous cl_event will be correctly released...
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(step, step), cl::NDRange(1, 1), cl::NullRange),
                *T.data->TS_buffer,
                N, step
        );
        eventsSnd.clear();

        //End of computation
        addDataCallbackTo(T.TS_lastComputationEvent, computationCallback, T.data);
        T.promptStateMutex.unlock();
    }

    std::cout << T;
    cl::vector<int> test(N, 0);
    T.CLSetup->getQueue().enqueueReadBuffer(
            permutation, true, 0, sizeof(int)*N,
            (void *) test.data(), nullptr,
            nullptr //In this function the previous cl_event is correctly released...
    );
    for(int j = 0; j < N; j++)
        std::cout << test[j] << std::endl;

    CLMatrixSetup::PerKerType kernelRes(T.CLSetup->getKernel("square_colInvPermutation"));
    basicUnaryOp<cl::Buffer &>(T, R, T.transpose, N, N, kernelRes, permutation);
}

void DNN::SquareMatrix::setCLSetup(std::shared_ptr<CLMatrixSetup> newSetup) {
    newSetup->addKernelsFromSource(libFile, 
        {
            "square_rowDilation",       "square_colDilation",
            "square_rowTransposition",  "square_colTransposition",
            "square_rowTransvection",   "square_colTransvection",
            "square_rowPermutation",    "square_colPermutation",
            "square_rowInvPermutation", "square_colInvPermutation",
            "square_inversionStepCheck", "square_inversionStepCol", "square_inversionStepRow", "square_inversionStepMain", "square_inversionStepFinal"
        },
        libCode
    );
    Matrix::setCLSetup(newSetup);
}
