#include "vector.hpp"

DNN::Vector::Vector(int nbRow, float expr) : Matrix(nbRow, 1, expr) {
    Vector::setCLSetup(CLSetup);
}

DNN::Vector::Vector(const cl::vector<float> &initialiser) : Matrix(initialiser.size(), 1, new cl::vector<float>(initialiser)) {
    Vector::setCLSetup(CLSetup);
}

DNN::Vector::Vector(cl::vector<float> &&initialiser) : Matrix(initialiser.size(), 1, new cl::vector<float>((cl::vector<float> &&) initialiser)) {
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
    Vector vectorResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows)
    );
    opAdd(*this, operand, vectorResult);

    return vectorResult;
}

DNN::Vector DNN::Vector::operator-(Vector &operand) {
    Vector vectorResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows)
    );
    opSub(*this, operand, vectorResult);

    return vectorResult;
}

DNN::Vector DNN::Vector::operator-() {
    Vector vectorResult(rows,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows)
    );
    opOpp(*this, vectorResult);

    return vectorResult;
}

DNN::Matrix DNN::Vector::addOverMatrix(Matrix &operand) {
    Matrix matrixResult(operand.getRowCount(), operand.getColumnCount(), //As R.transpose = A.transpose
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*operand.rows*operand.columns)
    );
    opAOM(*this, operand, matrixResult);

    return matrixResult;
}

DNN::Matrix DNN::Vector::subOverMatrix(Matrix &operand) {
    Matrix matrixResult(operand.getRowCount(), operand.getColumnCount(), //As R.transpose = A.transpose
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*operand.rows*operand.columns)
    );
    opSOM(*this, operand, matrixResult);

    return matrixResult;
}

void DNN::Vector::setCLSetup(CLMatrixSetup *newSetup) {
    newSetup->addKernelsFromSource(libFile, 
        {"vector_additionOM", "vector_transAdditionOM", "vector_substractionOM", "vector_transSubstractionOM"},
        libCode
    );
    Matrix::setCLSetup(newSetup);
}

DNN::Vector::Vector(int nbRow, cl::Buffer *existingBuffer) : Matrix(nbRow, 1, existingBuffer) {
    Vector::setCLSetup(CLSetup);
}

DNN::Vector::Vector(int nbRow, cl::vector<float> *existingVector) : Matrix(nbRow, 1, existingVector) {
    Vector::setCLSetup(CLSetup);
}

void DNN::Vector::opAOM(Vector &A, Matrix &B, Matrix &R) {
    assert(A.isValid() && B.isValid());
    assert(A.getRowCount() == B.getRowCount());

    //Prepare result and transposition (no need for TS behavior, see constructors)
    CLMatrixSetup::AddKerType kernel(B.transpose ? A.CLSetup->getKernel("vector_additionOM") : A.CLSetup->getKernel("vector_transAdditionOM"));

    R.TS_stateFlags |= StateFlags::COMPUTATION_EXECUTING;
    R.transpose = false; //We do not care about vector transposition (changes nothing...)
    R.rows      = B.getRowCount();
    R.columns   = B.getColumnCount();
    R.data->addBufferEvent();

    //Prepare operands
    cl::vector<cl::Event> events;
    events.reserve(2);
    A.mangageBeforeComputation(events); //WARNING : It locks the promptMutex !!!!
    B.mangageBeforeComputation(events);

    //Actual computations...
    std::lock_guard<std::recursive_mutex> lockRes(R.promptStateMutex);
    R.TS_lastComputationEvent = kernel( //Here the previous cl_event will be correctly released...
        cl::EnqueueArgs(A.CLSetup->getQueue(), events, cl::NDRange(R.rows, R.columns)),
        *A.data->TS_buffer,
        *B.data->TS_buffer,
        *R.data->TS_buffer
    );

    //Callbacks
    addDataCallbackTo(R.TS_lastComputationEvent, readCallback, A.data);
    addDataCallbackTo(R.TS_lastComputationEvent, readCallback, B.data);
    addDataCallbackTo(R.TS_lastComputationEvent, computationCallback, R.data);

    A.promptStateMutex.unlock();
    B.promptStateMutex.unlock();
}

void DNN::Vector::opSOM(Vector &A, Matrix &B, Matrix &R) {
    assert(A.isValid() && B.isValid());
    assert(A.getRowCount() == B.getRowCount());

    //Prepare result and transposition (no need for TS behavior, see constructors)
    CLMatrixSetup::AddKerType kernel(B.transpose ? A.CLSetup->getKernel("vector_substractionOM") : A.CLSetup->getKernel("vector_transSubstractionOM"));

    R.TS_stateFlags |= StateFlags::COMPUTATION_EXECUTING;
    R.transpose = false; //We do not care about vector transposition (changes nothing...)
    R.rows      = B.getRowCount();
    R.columns   = B.getColumnCount();
    R.data->addBufferEvent();

    //Prepare operands
    cl::vector<cl::Event> events;
    events.reserve(2);
    A.mangageBeforeComputation(events); //WARNING : It locks the promptMutex !!!!
    B.mangageBeforeComputation(events);

    //Actual computations...
    std::lock_guard<std::recursive_mutex> lockRes(R.promptStateMutex);
    R.TS_lastComputationEvent = kernel( //Here the previous cl_event will be correctly released...
        cl::EnqueueArgs(A.CLSetup->getQueue(), events, cl::NDRange(R.rows, R.columns)),
        *A.data->TS_buffer,
        *B.data->TS_buffer,
        *R.data->TS_buffer
    );

    //Callbacks
    addDataCallbackTo(R.TS_lastComputationEvent, readCallback, A.data);
    addDataCallbackTo(R.TS_lastComputationEvent, readCallback, B.data);
    addDataCallbackTo(R.TS_lastComputationEvent, computationCallback, R.data);

    A.promptStateMutex.unlock();
    B.promptStateMutex.unlock();
}

DNN::Vector DNN::operator*(Matrix &AL, Vector &X) {
    Vector vectorResult(AL.getRowCount(),
        new cl::Buffer (AL.getCLSetup()->getContext(), CL_MEM_READ_WRITE, sizeof(float)*AL.getRowCount()) //ENH : Setup is not usual !!!
    );
    Matrix::opMul(AL, X, vectorResult);

    return vectorResult;
}
