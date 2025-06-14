#include "matrix.hpp"

std::atomic<int> DNN::BufferLinkManager::DEBUG_created(0);
std::atomic<int> DNN::BufferLinkManager::DEBUG_destroyed(0);

std::shared_ptr<DNN::CLMatrixSetup> DNN::CLMatrixSetup::defaultCLSetup;

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

DNN::CLMatrixSetup::CLMatrixSetup() : context(cl::Context::getDefault()), queue(context)  {}

std::shared_ptr<DNN::CLMatrixSetup> DNN::CLMatrixSetup::getDefault() {
    if(!defaultCLSetup) defaultCLSetup.reset(new CLMatrixSetup());
    return defaultCLSetup;
}

DNN::CLMatrixSetup::CLMatrixSetup(cl::Context _context) : context(_context), queue(context) {}

DNN::CLMatrixSetup::CLMatrixSetup(cl::Context _context, cl::CommandQueue _queue) : context(_context), queue(_queue)  {}

bool DNN::CLMatrixSetup::addKernelsFromSource(const char *file, cl::vector<cl::string> kernels, int8_t libCode) {
    if(libCode & includedLibraries) return false;

    cl::Program program(context, cl::util::read_text_file(file) , true );
    for(auto key : kernels) {
        if(internalKernelLib.find(key) == internalKernelLib.end())
            internalKernelLib[key] = cl::Kernel(program, key);
    }
    includedLibraries |= libCode;
    return true;
}

bool DNN::CLMatrixSetup::addKernelsFromProgram(cl::Program program, cl::vector<cl::string> kernels, int8_t libCode) {
    if(libCode & includedLibraries) return false;

    for(auto key : kernels) {
        if(internalKernelLib.find(key) == internalKernelLib.end())
            internalKernelLib[key] = cl::Kernel(program, key);
    }
    includedLibraries |= libCode;
    return true;
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

DNN::BufferLinkManager::~BufferLinkManager() {
    internalLinkMutex.lock();
    assert(TS_bufferAccess == 0 && TS_vectorAccess == 0 && TS_tobeDeleted);
    ++DEBUG_destroyed;

    if(TS_vector != nullptr) delete TS_vector;
    if(TS_buffer != nullptr) delete TS_buffer;

    internalLinkMutex.unlock();
}

void DNN::BufferLinkManager::registerForDeletion() {
    internalLinkMutex.lock();

    TS_tobeDeleted = true; 
    TS_holder = nullptr;

    if(TS_bufferAccess == 0 && TS_buffer != nullptr){
        delete TS_buffer;
        TS_buffer = nullptr;
    }

    if(TS_vectorAccess == 0 && TS_vector != nullptr){
        delete TS_vector;
        TS_vector = nullptr;
    }

    if(TS_bufferAccess == 0 && TS_vectorAccess == 0) {
        internalLinkMutex.unlock();
        delete this; return; //Both instructions CANNOT be separated !!
    }

    internalLinkMutex.unlock();
}

void DNN::BufferLinkManager::waitForBufferEvents() const {
    internalLinkMutex.lock();
    while (TS_bufferAccess > 0) {
        internalLinkMutex.unlock();
        internalLinkMutex.lock();
    }
    internalLinkMutex.unlock();
}

DNN::Matrix::Matrix(int nbRow, int nbCol, float expr, std::shared_ptr<CLMatrixSetup> setup) : rows(nbRow), columns(nbCol),
    data(new BufferLinkManager(this)) {
    Matrix::setCLSetup(setup);

    data->TS_vector = new cl::vector<float>(rows*columns, expr);
    TS_stateFlags |= StateFlags::DATA_UPLOADED;
    //Host side creation :
    // - no need for buffer now
    // - no thread unsafe danger.
}

DNN::Matrix::Matrix(const cl::vector<cl::vector<float>> &initializer, bool transposed, std::shared_ptr<CLMatrixSetup> setup) : rows(initializer.size()), columns(initializer[0].size()), transpose(transposed),
    data(new BufferLinkManager(this)) {
    Matrix::setCLSetup(setup);

    bool _checkDim = true;
    for(int i = 1; i < rows; ++i)
        if(initializer[i].size() != columns) _checkDim = false;
    assert(_checkDim);

    data->TS_vector = new cl::vector<float>(rows*columns);
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < columns; ++j)
            (*data->TS_vector)[i*columns + j] = initializer[i][j];

    TS_stateFlags |= StateFlags::DATA_UPLOADED;
    //Host side creation :
    // - no need for buffer now
    // - no thread unsafe danger.
}

DNN::Matrix::Matrix(const Matrix &toCopy) : rows(toCopy.rows), columns(toCopy.columns) {
    *this = toCopy; //Using copy affectation as it should behave the same...
}

DNN::Matrix::Matrix(Matrix &&toMove) noexcept  : rows(toMove.rows), columns(toMove.columns) {
    assert(toMove.isValid());
    *this = (Matrix &&) toMove; //Using move affectation as it should behave the same...
}

DNN::Matrix::~Matrix() {
    std::cout << "Destroy Matrix !!" << std::endl;
    if(data == nullptr) return;

    data->registerForDeletion(); //Ensure that no thread own any of the matrix's mutexes.
}

//ENH : Should we really delete all the previous data ??
DNN::Matrix &DNN::Matrix::operator=(const Matrix &toCopy) {
    assert(getRowCount() == toCopy.getRowCount() && getColumnCount() == toCopy.getColumnCount());
    if(this == &toCopy || data == toCopy.data || !toCopy.isValid()) 
        return *this;

    //Preparation for the copy (affectation behaviour, TS as no data registered for deletion)
    if(data != nullptr) data->registerForDeletion();
    data = new BufferLinkManager(this);
    TS_stateFlags = StateFlags::NO_FLAG;
    TS_lastComputationEvent = (cl_event) nullptr; //Here the previous cl_event will be correctly released...
    TS_lastDownloadEvent    = (cl_event) nullptr;
    TS_lastUploadEvent      = (cl_event) nullptr;

    transpose = toCopy.transpose; //We keep the same internal representation of the matrix
    rows = toCopy.rows;
    columns = toCopy.columns;
    this->setCLSetup(toCopy.CLSetup); //To ensure it calls the override function...

    //Effective smart copy... (never copy both, if such a behaviour is wanted the user should use the static copy function)
    toCopy.promptStateMutex.lock();
    if(toCopy.TS_stateFlags & StateFlags::DATA_UPLOADED) {
        toCopy.promptStateMutex.unlock(); //Using TS_vector is safe...

        data->TS_vector = new cl::vector<float>(*toCopy.data->TS_vector);
        TS_stateFlags |= StateFlags::DATA_UPLOADED;
    }
    else if (toCopy.TS_stateFlags & StateFlags::DATA_DOWNLOADED) {
        toCopy.promptStateMutex.unlock(); 
        
        /// Mimic computation from now...

        cl::vector<cl::Event> events;
        events.reserve(1);
        toCopy.manageBeforeReading();
        data->addBufferEvent();

        toCopy.manageBeforeComputation(events); //WARNING : It locks the promptMutex !!!!
        manageBeforeComputation(events, true); // ENH : In this context it is mainly useless...

        data->TS_buffer =  new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns);
        TS_stateFlags |= StateFlags::COMPUTATION_EXECUTING | StateFlags::DATA_DOWNLOADED;

        CLSetup->getQueue().enqueueCopyBuffer(
            *toCopy.data->TS_buffer, *data->TS_buffer,
            0, 0, sizeof(float)*rows*columns, 
            &events, &TS_lastComputationEvent            
        );

        //Callbacks
        addDataCallbackTo(TS_lastComputationEvent, readCallback, toCopy.data);
        addDataCallbackTo(TS_lastComputationEvent, computationCallback, data);

        toCopy.promptStateMutex.unlock();
        promptStateMutex.unlock();
    }
    else assert(false); //Invalid state

    return *this;
}

DNN::Matrix &DNN::Matrix::operator=(Matrix &&toMove) noexcept {
    assert(getRowCount() == toMove.getRowCount() && getColumnCount() == toMove.getColumnCount());
    if(this == &toMove || data == toMove.data || !toMove.isValid()) 
        return *this;

    transpose = toMove.transpose; //We keep the same internal representation of the matrix
    rows = toMove.rows;
    columns = toMove.columns;
    this->setCLSetup(toMove.CLSetup); //To ensure it calls the override function...

    //Zero case : me or you (steal, complete steal)
    if(!isValid()) {
        if(data != nullptr) data->registerForDeletion();
        data = toMove.data; toMove.data = nullptr;

        //TS part
        std::lock_guard<std::recursive_mutex> lockData(data->internalLinkMutex);
        std::lock_guard<std::recursive_mutex> lockThis(promptStateMutex); //Should be useless but good practice...
        std::lock_guard<std::recursive_mutex> lockMove(toMove.promptStateMutex);

        TS_stateFlags = toMove.TS_stateFlags;
        TS_lastComputationEvent     = (cl::Event &&) toMove.TS_lastComputationEvent;
        TS_lastUploadEvent          = (cl::Event &&) toMove.TS_lastUploadEvent;
        TS_lastDownloadEvent        = (cl::Event &&) toMove.TS_lastDownloadEvent;

        data->TS_holder = this;
    }

    //First case : complete move
    else if(toMove.data->TS_buffer != nullptr && toMove.data->TS_vector != nullptr) {
        //No need to protect as previous events and flags are erased...
        data->registerForDeletion();
        data = toMove.data; toMove.data = nullptr;

        //TS part
        std::lock_guard<std::recursive_mutex> lockData(data->internalLinkMutex);
        std::lock_guard<std::recursive_mutex> lockThis(promptStateMutex);
        std::lock_guard<std::recursive_mutex> lockMove(toMove.promptStateMutex);

        TS_stateFlags = toMove.TS_stateFlags;
        TS_lastComputationEvent = (cl::Event &&) toMove.TS_lastComputationEvent;
        TS_lastComputationEvent = (cl::Event &&) toMove.TS_lastUploadEvent;
        TS_lastComputationEvent = (cl::Event &&) toMove.TS_lastDownloadEvent;
        
        data->TS_holder = this;
    }

    //Second case : host side affectation (waits for everything)
    else if(toMove.data->TS_vector != nullptr) {
        waitForResults();
        toMove.waitForResults(); //Just security should return immediately...

        data->internalLinkMutex.lock();
        toMove.data->internalLinkMutex.lock();
        
        delete data->TS_vector;
        data->TS_vector = toMove.data->TS_vector;
        toMove.data->TS_vector = nullptr;
        data->TS_vectorAccess = 0;//Should be useless...
        toMove.data->TS_vectorAccess = 0;

        data->internalLinkMutex.unlock();
        toMove.data->internalLinkMutex.unlock();
        
        std::lock_guard<std::recursive_mutex> lockThis(promptStateMutex);
        TS_stateFlags &= oppFlag(StateFlags::INT_DOWNLOAD_FLAGS);
        TS_stateFlags |= StateFlags::DATA_UPLOADED; //Should be useless...
    }

    //Third case : device side affectation (wait the less possible)
    else if(toMove.data->TS_buffer != nullptr) {
        //No need to protect as previous events and flags are erased...
        toMove.data->TS_vector = data->TS_vector;
        data->TS_vector = nullptr;
        data->registerForDeletion(); //No issue with external actions as TS_vector is saved...
        data = toMove.data; toMove.data = nullptr;

        std::lock_guard<std::recursive_mutex> lockOldData(data->internalLinkMutex);
        std::lock_guard<std::recursive_mutex> lockNewData(toMove.data->internalLinkMutex);
        std::lock_guard<std::recursive_mutex> lockThis(promptStateMutex);
        std::lock_guard<std::recursive_mutex> lockMove(toMove.promptStateMutex);

        //External creation
        if(TS_stateFlags & (StateFlags::DATA_DOWNLOADING | StateFlags::EXTERNAL_DOWNLOADING)) { //Includes potential external download
            TS_stateFlags |= EXTERNAL_DOWNLOADING;
            data->addVectorEvent(); //TODO : How to avoid locking this mutex ??
            addDataCallbackTo(TS_lastDownloadEvent, externalDownloadCallback, data);
        }
        if(TS_stateFlags & (StateFlags::DATA_UPLOADING | StateFlags::EXTERNAL_UPLOADING)) { //Includes potential external upload
            TS_stateFlags |= EXTERNAL_UPLOADING;
            data->addVectorEvent(); //TODO : How to avoid locking this mutex ??
            addDataCallbackTo(TS_lastUploadEvent, externalUploadCallback, data);
        }

        //General TS changes
        TS_stateFlags &= oppFlag(StateFlags::INTERNAL_FLAGS);
        TS_stateFlags |= toMove.TS_stateFlags & StateFlags::INTERNAL_FLAGS;
        TS_lastComputationEvent = (cl::Event &&) toMove.TS_lastComputationEvent; //Will serve as deletion of the previous event

        data->TS_holder = this;
    }

    return *this;
}

DNN::Matrix DNN::Matrix::operator+(const Matrix &operand) const {
    Matrix matrixResult(rows, columns, //As R.transpose = A.transpose
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opAdd(*this, operand, matrixResult);

    return matrixResult;
}

DNN::Matrix DNN::Matrix::operator-(const Matrix &operand) const {
    Matrix matrixResult(rows, columns, //As R.transpose = A.transpose
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opSub(*this, operand, matrixResult);

    return matrixResult;
}

DNN::Matrix DNN::Matrix::operator*(const Matrix &operand) const {
    Matrix matrixResult(
        (!transpose || !operand.transpose) ? getRowCount() : operand.rows,
        (!transpose || !operand.transpose) ? operand.getColumnCount() : columns,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*getRowCount()*operand.getColumnCount()),
        CLSetup
    );
    opMul(*this, operand, matrixResult);

    return matrixResult;
}

DNN::Matrix DNN::Matrix::operator-() const {
    Matrix matrixResult(rows, columns, 
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opOpp(*this, matrixResult);

    return matrixResult;
}

DNN::Matrix DNN::Matrix::hadamardProduct(const Matrix &operand) const {
    Matrix matrixResult(rows, columns,
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    opHad(*this, operand, matrixResult);

    return matrixResult;
}

DNN::Matrix DNN::Matrix::executeKernel(cl::KernelFunctor<cl::Buffer &, cl::Buffer &> &kernel) const {
    assert(isValid());

    //Prepare result (no need for TS behavior, see constructors)
    Matrix matrixResult(rows, columns, 
        new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*rows*columns),
        CLSetup
    );
    basicUnaryOp(*this, matrixResult, transpose, rows, columns, kernel);

    return matrixResult;
}

float &DNN::Matrix::getLValueElement(cl::size_type row, cl::size_type col) {
    assert(row < getRowCount() && col < getColumnCount() && isValid());
    waitForResults();

    promptStateMutex.lock();
    TS_stateFlags &= oppFlag(
        StateFlags::DATA_DOWNLOADED | 
        StateFlags::DATA_DOWNLOADING
    );
    TS_stateFlags |= StateFlags::DATA_UPLOADED; //Should be useless...
    promptStateMutex.unlock();
    
    const size_t index = transpose ? col * columns + row : row * columns + col;
    return (*data->TS_vector)[index]; //Thread safe because it can't be registered for deletion...
}

float DNN::Matrix::getRValueElement(cl::size_type row, cl::size_type col) const {
    assert(row < getRowCount() && col < getColumnCount() && isValid());
    waitForConstResults();

    const size_t index = transpose ? col * columns + row : row * columns + col;
    return (*data->TS_vector)[index]; //Thread safe because it can't be registered for deletion...
}

void DNN::Matrix::askForResults() const {
    if(!isValid()) return;

    uploadData();
}

void DNN::Matrix::waitForResults() const {
    if(!isValid()) return;

    //Basically waits for any possible event (except readings !!)...
    uploadData();
    waitForExternal(); //Should not be necessary as an upload is requested...
    waitForDownload(); //In case the upload is skipped...
    waitForUpload();
}

void DNN::Matrix::waitForConstResults() const {
    if(!isValid()) return;

    uploadData();
    waitForExternal(); //Should not be necessary as an upload is requested...
    waitForUpload(); 
}

void DNN::Matrix::setCLSetup(std::shared_ptr<CLMatrixSetup> newSetup) {
    newSetup->addKernelsFromSource(libFile, 
        {
            "matrix_addition", "matrix_transAddition",
            "matrix_subtraction", "matrix_transSubtraction",
            "matrix_product", "matrix_transLProduct", "matrix_transRProduct",
            "matrix_hadamard", "matrix_transHadamard",
            "matrix_opposite"
        },
        libCode
    );
    CLSetup = newSetup;
}

// ENH : Should result matrix be checked ??
void DNN::Matrix::opAdd(const Matrix &A, const Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount() && A.getColumnCount() == B.getColumnCount());

    CLMatrixSetup::AddKerType kernel((A.transpose == B.transpose) ? A.CLSetup->getKernel("matrix_addition") : A.CLSetup->getKernel("matrix_transAddition"));
    basicBinaryOp(A, B, R, A.transpose, A.rows, A.columns, kernel);
}

void DNN::Matrix::opSub(const Matrix &A, const Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount() && A.getColumnCount() == B.getColumnCount());

    //Prepare result and transposition (no need for TS behavior, see constructors)
    CLMatrixSetup::SubKerType kernel((A.transpose == B.transpose) ? A.CLSetup->getKernel("matrix_subtraction") : A.CLSetup->getKernel("matrix_transSubtraction"));
    basicBinaryOp(A, B, R, A.transpose, A.rows, A.columns, kernel);
}

void DNN::Matrix::opMul(const Matrix &A, const Matrix &B, Matrix &R) {
    assert(A.isValid() && B.isValid());
    assert(A.getColumnCount() == B.getRowCount());

    //Prepare result (no need for TS behavior, see constructors)
    CLMatrixSetup::ProdKerType kernel((A.transpose == B.transpose) ? 
        A.CLSetup->getKernel("matrix_product") : 
        A.transpose ? A.CLSetup->getKernel("matrix_transLProduct") : A.CLSetup->getKernel("matrix_transRProduct")
    );
    R.transpose = A.transpose && B.transpose; //Avoids creating a useless variable...

    basicBinaryOp(A, B, R, R.transpose,
        R.transpose ? B.getColumnCount() : A.getRowCount(), 
        R.transpose ? A.getRowCount() : B.getColumnCount(), 
        kernel, A.getColumnCount()
    );
}

void DNN::Matrix::opHad(const Matrix &A, const Matrix &B, Matrix &R) {
    assert(A.getRowCount() == B.getRowCount() && A.getColumnCount() == B.getColumnCount());

    //Prepare result and transposition (no need for TS behavior, see constructors)
    CLMatrixSetup::SubKerType kernel((A.transpose == B.transpose) ? A.CLSetup->getKernel("matrix_hadamard") : A.CLSetup->getKernel("matrix_transHadamard"));
    basicBinaryOp(A, B, R, A.transpose, A.rows, A.columns, kernel);
}

void DNN::Matrix::opOpp(const Matrix &A, Matrix &R) {
    //Prepare result (no need for TS behavior, see constructors)
    CLMatrixSetup::OppKerType kernel(A.CLSetup->getKernel("matrix_opposite"));

    basicUnaryOp(A, R, A.transpose, A.rows, A.columns, kernel);
}

DNN::Matrix::Matrix(int nbRow, int nbCol, cl::Buffer *existingBuffer, std::shared_ptr<CLMatrixSetup> setup) : rows(nbRow), columns(nbCol),
    data(new BufferLinkManager(this)) {
    Matrix::setCLSetup(setup);
    
    data->TS_buffer = existingBuffer;
    TS_stateFlags |= StateFlags::DATA_DOWNLOADED;
    //Buffer side creation : no thread unsafe danger
}

DNN::Matrix::Matrix(int nbRow, int nbCol, cl::vector<float> *existingVector, std::shared_ptr<CLMatrixSetup> setup)  : rows(nbRow), columns(nbCol),
    data(new BufferLinkManager(this)) {
    Matrix::setCLSetup(setup);
    
    data->TS_vector = existingVector;
    TS_stateFlags |= StateFlags::DATA_UPLOADED;
    //Buffer side creation : no thread unsafe danger
}

void DNN::Matrix::manageBeforeReading(bool download) const {
    assert(isValid());

    //Data management
    if(download) downloadData();
    data->addBufferEvent();
}

void DNN::Matrix::manageBeforeComputation(cl::vector<cl::Event> &requiredEvents, bool includeUpload) const {
    //Event management
    promptStateMutex.lock();
    if(TS_stateFlags & StateFlags::COMPUTATION_EXECUTING)
        requiredEvents.push_back(TS_lastComputationEvent);
    else if(TS_stateFlags & StateFlags::DATA_DOWNLOADING)
        requiredEvents.push_back(TS_lastDownloadEvent);

    if(includeUpload && (TS_stateFlags & StateFlags::DATA_UPLOADING))
        requiredEvents.push_back(TS_lastUploadEvent);
}

void DNN::Matrix::waitForExternal() const {
    if(!isValid()) return;

    cl::vector<cl::Event> events;
    events.reserve(2);

    promptStateMutex.lock();

    if(TS_stateFlags & StateFlags::EXTERNAL_DOWNLOADING) {
        safelyWaitForEvent(TS_lastDownloadEvent, waitingDownloadMutex);
        TS_lastDownloadEvent = (cl_event) nullptr;//Previous cl_event is correctly released...
    }
    if(TS_stateFlags & StateFlags::EXTERNAL_UPLOADING) {
        safelyWaitForEvent(TS_lastUploadEvent, waitingUploadMutex);
        TS_lastUploadEvent = (cl_event) nullptr; //Previous cl_event is correctly released...
    }
    
    TS_stateFlags &= oppFlag(StateFlags::EXTERNAL_FLAGS);    
    promptStateMutex.unlock();
}

void DNN::Matrix::waitForDownload() const {
    if(!isValid()) return;

    promptStateMutex.lock();
    if(!(TS_stateFlags & StateFlags::EXTERNAL_DOWNLOADING) && TS_lastDownloadEvent.get() != nullptr)
        safelyWaitForEvent(TS_lastDownloadEvent, waitingDownloadMutex); // External downloads are not waited for...
    
    TS_lastDownloadEvent = (cl_event) nullptr;
    promptStateMutex.unlock();
}

void DNN::Matrix::waitForComputation() const {
    if(!isValid()) return;

    promptStateMutex.lock();
    if(TS_lastComputationEvent.get() != nullptr)
        safelyWaitForEvent(TS_lastComputationEvent, waitingComputationMutex, false);

    TS_lastComputationEvent = (cl_event) nullptr; //Previous cl_event is correctly released...
    promptStateMutex.unlock();
}

void DNN::Matrix::waitForUpload() const {
    if(!isValid()) return;

    promptStateMutex.lock();
    if(!(TS_stateFlags & StateFlags::EXTERNAL_UPLOADING) && TS_lastUploadEvent.get() != nullptr)
        safelyWaitForEvent(TS_lastUploadEvent, waitingUploadMutex);
    
    TS_lastUploadEvent = (cl_event) nullptr; //Previous cl_event is correctly released...
    promptStateMutex.unlock();
}

void DNN::Matrix::safelyWaitForEvent(cl::Event &event, std::mutex &waitingMutex, bool relock) const {
    if(!isValid()) return;

    waitingMutex.lock();
    promptStateMutex.unlock();

    event.wait(); //Supposed to be read only on the cl_event underlying pointer...

    waitingMutex.unlock();
    if(relock) promptStateMutex.lock();
}

void DNN::Matrix::downloadData() const {
    if(!isValid() || data->TS_vector == nullptr) return;

    promptStateMutex.lock();
    if(TS_stateFlags & StateFlags::INT_DOWNLOAD_FLAGS) {
        promptStateMutex.unlock();
        return;
    }
    promptStateMutex.unlock();

    //Data control (promptStateMutex must be unlocked)
    if(data->TS_buffer == nullptr)
        data->TS_buffer = new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_ONLY, sizeof(float)*rows*columns);
    
    data->waitForBufferEvents();
    data->addBufferEvent();
    data->addVectorEvent();

    //Event management
    std::lock_guard<std::recursive_mutex> lock(promptStateMutex);
    cl::vector<cl::Event> dependencies;
    dependencies.reserve(2);

    if(TS_stateFlags & StateFlags::EXTERNAL_DOWNLOADING)
        dependencies.push_back(TS_lastDownloadEvent);
    if(TS_stateFlags & StateFlags::EXTERNAL_UPLOADING)
        dependencies.push_back(TS_lastUploadEvent);

    //OpenCL request
    CLSetup->getQueue().enqueueWriteBuffer(
        *data->TS_buffer, false, 0, sizeof(float)*rows*columns, 
        (void *) data->TS_vector->data(), &dependencies,
        &TS_lastDownloadEvent //In this function the previous cl_event is correctly released...
    );
    TS_stateFlags &= oppFlag(StateFlags::EXTERNAL_DOWNLOADING);  //Only TS_lastDownloadEvent is modified...
    TS_stateFlags |= StateFlags::DATA_DOWNLOADING;

    addDataCallbackTo(TS_lastDownloadEvent, downloadCallback, data);
}

void DNN::Matrix::uploadData() const {
    if(!isValid() || data->TS_buffer == nullptr) return;

    promptStateMutex.lock();
    if(TS_stateFlags & (StateFlags::INT_UPLOAD_FLAGS)) {
        promptStateMutex.unlock();
        return;
    }
    promptStateMutex.unlock();

    //Data control (promptStateMutex must be unlocked)
    if(data->TS_vector == nullptr) //Create buffer if no buffer exist for now
        data->TS_vector = new cl::vector<float>(rows*columns);

    data->addBufferEvent();
    data->addVectorEvent();

    //Event management
    std::lock_guard<std::recursive_mutex> lock(promptStateMutex);
    cl::vector<cl::Event> dependencies;
    dependencies.reserve(3);

    if(TS_stateFlags & StateFlags::COMPUTATION_EXECUTING)
        dependencies.push_back(TS_lastComputationEvent);
    else if(TS_stateFlags & StateFlags::DATA_DOWNLOADING) //Should NEVER happen...
        dependencies.push_back(TS_lastDownloadEvent);

    if(TS_stateFlags & StateFlags::EXTERNAL_DOWNLOADING)
        dependencies.push_back(TS_lastDownloadEvent);
    if(TS_stateFlags & StateFlags::EXTERNAL_UPLOADING) //Just to be clearer
        dependencies.push_back(TS_lastUploadEvent);
    else if(TS_lastUploadEvent.get() != nullptr) //To avoid waiting for twice the same event...
        dependencies.push_back(TS_lastUploadEvent); //TODO : This one is included in computation event nah ?

    //OpenCL request
    CLSetup->getQueue().enqueueReadBuffer(
        *data->TS_buffer, false, 0, sizeof(float)*rows*columns, 
        (void *) data->TS_vector->data(), &dependencies,
        &TS_lastUploadEvent //In this function the previous cl_event is correctly released...
    );
    TS_stateFlags &= oppFlag(StateFlags::EXTERNAL_UPLOADING); //Only TS_lastUploadEvent is modified...
    TS_stateFlags |= StateFlags::DATA_UPLOADING;

    addDataCallbackTo(TS_lastUploadEvent, uploadCallback, data);
}

void DNN::Matrix::addDataCallbackTo(cl::Event &event, void (CL_CALLBACK *cb)(cl_event, cl_int, void *), BufferLinkManager *arg) {
    clRetainEvent(event.get());
    event.setCallback(CL_COMPLETE, cb, (void *) arg);
}

void CL_CALLBACK DNN::Matrix::computationCallback(cl_event event, cl_int, void *_linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    Matrix *holder = linkManager->TS_holder;
    if(holder != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(holder->promptStateMutex);
        if(holder->TS_lastComputationEvent.get() == event) {
            holder->TS_stateFlags |= StateFlags::COMPUTATION_EXECUTED;
            holder->TS_stateFlags &= oppFlag(StateFlags::COMPUTATION_EXECUTING);

            //Previous cl_event is correctly released...
            if(holder->waitingComputationMutex.try_lock()) {
                holder->TS_lastComputationEvent = (cl_event) nullptr;
                holder->waitingComputationMutex.unlock();
            }
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks(linkManager, true, false);
    clReleaseEvent(event);
    std::cout << "End computation" << std::endl;
}

void CL_CALLBACK DNN::Matrix::downloadCallback(cl_event event, cl_int, void *_linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    Matrix *holder = linkManager->TS_holder;
    if(holder != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(holder->promptStateMutex);
        if(holder->TS_lastDownloadEvent.get() == event) {//Security...
            holder->TS_stateFlags |= StateFlags::DATA_DOWNLOADED;
            holder->TS_stateFlags &= oppFlag(StateFlags::DATA_DOWNLOADING);

            //Previous cl_event is correctly released...
            if(holder->waitingDownloadMutex.try_lock()) {
                holder->TS_lastDownloadEvent = (cl_event) nullptr;
                holder->waitingDownloadMutex.unlock();
            }
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks(linkManager, true, true);
    clReleaseEvent(event);
    std::cout << "End download" << std::endl;
}

void CL_CALLBACK DNN::Matrix::uploadCallback(cl_event event, cl_int, void *_linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    Matrix *holder = linkManager->TS_holder;
    if(holder != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(holder->promptStateMutex);
        if(holder->TS_lastUploadEvent.get() == event) {//Security...
            holder->TS_stateFlags |= StateFlags::DATA_UPLOADED;
            holder->TS_stateFlags &= oppFlag(StateFlags::DATA_UPLOADING);

            //Previous cl_event is correctly released...
            if(holder->waitingUploadMutex.try_lock()) {
                holder->TS_lastUploadEvent = (cl_event) nullptr;
                holder->waitingUploadMutex.unlock();
            }
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks(linkManager, true, true);
    clReleaseEvent(event);
    std::cout << "End upload" << std::endl;
}

void CL_CALLBACK DNN::Matrix::readCallback(cl_event event, cl_int, void* _linkManager) {
    checkDeletionForCallbacks((BufferLinkManager *) _linkManager, true, false);
    clReleaseEvent(event);
}

void CL_CALLBACK DNN::Matrix::externalDownloadCallback(cl_event event, cl_int, void *_linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    Matrix *holder = linkManager->TS_holder;
    if(holder != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(holder->promptStateMutex);
        if(holder->TS_lastDownloadEvent.get() == event) {//Security...
            holder->TS_stateFlags &= oppFlag(StateFlags::EXTERNAL_DOWNLOADING); //No issue with multiple externals (for one holder only the last is registered)

            //Previous cl_event is correctly released...
            if(holder->waitingDownloadMutex.try_lock()) {
                holder->TS_lastDownloadEvent = (cl_event) nullptr;
                holder->waitingDownloadMutex.unlock();
            }
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks((BufferLinkManager *) _linkManager, false, true);
    clReleaseEvent(event);
}

void CL_CALLBACK DNN::Matrix::externalUploadCallback(cl_event event, cl_int, void* _linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    Matrix *holder = linkManager->TS_holder;
    if(holder != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(holder->promptStateMutex);
        if(holder->TS_lastUploadEvent.get() == event) {//Security...
            holder->TS_stateFlags &= oppFlag(StateFlags::EXTERNAL_UPLOADING); //No issue with multiple externals (for one holder only the last is registered)

            //Previous cl_event is correctly released...
            if(holder->waitingUploadMutex.try_lock()) {
                holder->TS_lastUploadEvent = (cl_event) nullptr;
                holder->waitingUploadMutex.unlock();
            }
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks((BufferLinkManager *) _linkManager, false, true);
    clReleaseEvent(event);
}

void CL_CALLBACK DNN::Matrix::checkDeletionForCallbacks(BufferLinkManager *linkManager, bool buffer, bool vector) {
    linkManager->internalLinkMutex.lock();
    linkManager->TS_bufferAccess -= (buffer && linkManager->TS_bufferAccess > 0) ? 1 : 0;
    linkManager->TS_vectorAccess -= (vector && linkManager->TS_vectorAccess > 0) ? 1 : 0;

    if(linkManager->TS_tobeDeleted) {
        if(linkManager->TS_bufferAccess == 0 && linkManager->TS_buffer != nullptr) {
            delete linkManager->TS_buffer;
            linkManager->TS_buffer = nullptr;
        }
        if(linkManager->TS_vectorAccess == 0 && linkManager->TS_vector != nullptr) {
            delete linkManager->TS_vector;
            linkManager->TS_vector = nullptr;
        }
        if(linkManager->TS_vectorAccess == 0 && linkManager->TS_bufferAccess == 0) {
            linkManager->internalLinkMutex.unlock();
            delete linkManager; return;
        }
    }

    linkManager->internalLinkMutex.unlock();
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
