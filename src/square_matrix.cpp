#include "square_matrix.hpp"

void DNN::SquareMatrix::_pow(const SquareMatrix &A, unsigned int exp, SquareMatrix &R) {
    auto setup = A.getCLSetup();
    auto N = A.getRowCount();

    //We treat small case specifically to avoid useless copies...
    if(exp == 0) { R = IDENTITY(A.getRowCount(), setup); return; }
    if(exp == 1) { R = A; return; }

    SquareMatrix *result[2] = {new SquareMatrix(setup), &R};
    SquareMatrix power[2] = {{setup}, {setup}};

    uint8_t resultParity = 0, powerParity  = 0;
    bool resultFirst  = true;

    if(exp % 2) {
        *result[0] = A; //ENH : Here a shallow copy is more than enough
        resultFirst = false;
    }
    _mul(A, A, power[powerParity]);
    exp >>= 1;

    while(exp > 0) {
        if(exp % 2) {
            if(resultFirst) {
                *result[0] = power[powerParity]; //ENH : How to remove this copy ?
                resultFirst = false;
            }
            else {
                _mul(power[powerParity], *result[resultParity], *result[resultParity ^ 1]);
                resultParity ^= 1;
                getBase(power[powerParity]).registerAdditionalComputing(getBase(*result[resultParity]));
            }
        }
        _mul(power[powerParity], power[powerParity], power[powerParity ^ 1]);
        powerParity ^= 1;
        exp >>= 1;
    }

    R = *result[resultParity];
    delete result[0];
}

void DNN::SquareMatrix::_pow(const DNN::SquareMatrix &A, int exp, DNN::SquareMatrix &R) {
    if(exp >= 0) {
        SquareMatrix::_pow(A, (unsigned int) exp, R);
    } else {
        SquareMatrix::_inv(A, R);
        SquareMatrix::_pow(R, (unsigned int) -exp, R);
    }
}

void DNN::SquareMatrix::_inv(const SquareMatrix &A, SquareMatrix &R) {
    SquareMatrix TM(A); //Absolutely needed as A must be modified
    BaseMatrix &T = getBase(TM);
    std::shared_ptr<CLMatrixSetup> setup = T.getCLSetup();
    const int N = T.getRowCount();

    //Basc variables preparation...
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int> kernelCheck(setup->getKernel("square_inversionStepCheck"));
    cl::KernelFunctor<cl::Buffer, cl::Buffer> kernelPivoting(setup->getKernel("square_rowTransposition"));
    cl::KernelFunctor<cl::Buffer, int, int> kernelMain(setup->getKernel("square_inversionStepMain"));
    cl::KernelFunctor<cl::Buffer, int, int> kernelRow(setup->getKernel("square_inversionStepRow"));
    cl::KernelFunctor<cl::Buffer, int, int> kernelCol(setup->getKernel("square_inversionStepCol"));
    cl::KernelFunctor<cl::Buffer, int, int> kernelFinal(setup->getKernel("square_inversionStepFinal"));

    cl::vector<cl::Event> eventsFst;
    cl::vector<cl::Event> eventsSnd;
    eventsFst.reserve(4);
    eventsSnd.reserve(4);

    cl::CommandQueue queue = setup->getQueue();
    cl::Buffer permutation(setup->getContext(), CL_MEM_READ_WRITE, sizeof(int)*N);
    cl::Buffer transposition(setup->getContext(), CL_MEM_READ_WRITE, sizeof(int)*2);
    {
        cl::vector<int> preparation(N, 0);
        for (int j = 0; j < N; j++) preparation[j] = j;

        eventsFst.push_back(cl::Event());
        setup->getQueue().enqueueWriteBuffer(
                permutation, true, 0, sizeof(int) * N,
                (void *) preparation.data(), nullptr,
                &eventsFst[0] //In this function the previous cl_event is correctly released...
        );
    }

    for(int step = 0; step < N; step++) {
        //Prepare events
        T.manageBeforeComputing(eventsFst, N, N, true); //WARNING : It locks the promptMutex !!!!

        ///Pre-kernels : Check and pivoting
        eventsSnd.push_back(kernelCheck(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(1)),
                T.getCLData(),
                transposition,
                permutation,
                N, step
        ));
        eventsFst.clear();

        eventsFst.push_back(kernelPivoting(
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(N)),
                T.getCLData(),
                transposition
        ));
        eventsSnd.clear();

        /// First kernel : row of the current step
        eventsSnd.push_back(kernelRow(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(step, 0), cl::NDRange(1, step), cl::NullRange),
                T.getCLData(),
                N, step
        ));
        eventsSnd.push_back(kernelRow(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(step, step + 1), cl::NDRange(1, N - step - 1), cl::NullRange),
                T.getCLData(),
                N, step
        ));
        eventsFst.clear();

        /// Second kernel : elimination in all rows except the current
        eventsFst.push_back(kernelMain( //Update to B_00
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(0, 0), cl::NDRange(step, step), cl::NullRange),
                T.getCLData(),
                N, step
        ));
        eventsFst.push_back(kernelMain( //Update to B_20
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(step + 1, 0), cl::NDRange(N - step - 1, step), cl::NullRange),
                T.getCLData(),
                N, step
        ));
        eventsFst.push_back(kernelMain( //Update to A_02
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(0, step + 1), cl::NDRange(step, N - step - 1), cl::NullRange),
                T.getCLData(),
                N, step
        ));
        eventsFst.push_back(kernelMain( //Update to A_22
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(step + 1, step + 1), cl::NDRange(N - step - 1, N - step - 1), cl::NullRange),
                T.getCLData(),
                N, step
        ));
        eventsSnd.clear();

        /// Third kernel : columns of the current step
        eventsSnd.push_back(kernelCol(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(0, step), cl::NDRange(step, 1), cl::NullRange),
                T.getCLData(),
                N, step
        ));
        eventsSnd.push_back(kernelCol(
                cl::EnqueueArgs(queue, eventsFst, cl::NDRange(step+1, step), cl::NDRange(N - step - 1, 1), cl::NullRange),
                T.getCLData(),
                N, step
        ));
        eventsFst.clear();

        /// Fourth kernel : central element of the step
        eventsFst.push_back(kernelFinal( //Here the previous cl_event will be correctly released...
                cl::EnqueueArgs(queue, eventsSnd, cl::NDRange(step, step), cl::NDRange(1, 1), cl::NullRange),
                T.getCLData(),
                N, step
        ));

        //End of computation
        T.finishComputing(eventsFst[0]);
        eventsFst.clear();
        eventsSnd.clear();
    }

    std::cout << TM << std::endl;

    cl::vector<int> test(N, 0);
    setup->getQueue().enqueueReadBuffer(
            permutation, true, 0, sizeof(int)*N,
            (void *) test.data(), nullptr,
            nullptr //In this function the previous cl_event is correctly released...
    );
    for(int j = 0; j < N; j++)
        std::cout << test[j] << std::endl;

    cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &> kernelRes(setup->getKernel("square_colInvPermutation"));
    BaseMatrix::basicUnaryOp<cl::Buffer &>(T, getBase(R), N, N, kernelRes, permutation);
}

DNN::SquareMatrix DNN::SquareMatrix::SCALAR(int N, float lambda, std::shared_ptr<CLMatrixSetup> setup) {
    SquareMatrix identity(N, 0.f, setup);
    for(int i = 0; i < N; ++i) identity[i][i] = lambda;
    return identity;
}

void DNN::SquareMatrix::buildCLSetup() {
    getCLSetup()->addKernelsFromSource(libFile,
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
}

DNN::SquareMatrix DNN::operator^(const DNN::SquareMatrix &A, int exp) {
    SquareMatrix matrixResult(A.getCLSetup());
    SquareMatrix::_pow(A, exp, matrixResult);
    return matrixResult;
}
