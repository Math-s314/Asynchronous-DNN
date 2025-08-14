#include "base.hpp"

std::atomic<int> DNN::BufferLinkManager::DEBUG_created(0);
std::atomic<int> DNN::BufferLinkManager::DEBUG_destroyed(0);
std::shared_ptr<DNN::CLMatrixSetup> DNN::CLMatrixSetup::defaultCLSetup(nullptr);


/// CLMatrixSetup class definitions

std::shared_ptr<DNN::CLMatrixSetup> DNN::CLMatrixSetup::getDefault() {
    if(!defaultCLSetup) {
        defaultCLSetup.reset(new CLMatrixSetup());
    }
    return defaultCLSetup;
}

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


/// BufferLinkManager class definitions

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


/// BaseMatrix class definitions

//Constructors

DNN::BaseMatrix::BaseMatrix(const cl::vector<cl::vector<float>> &initializer, std::shared_ptr<CLMatrixSetup> setup) :
    rows(initializer.size()), columns(initializer[0].size()), data(new BufferLinkManager(this)), CLSetup(setup) {
    bool _checkDim = true;
    for(int i = 1; i < rows; ++i)
        if(initializer[i].size() != columns) _checkDim = false;
    assert(_checkDim);

    data->TS_vector = new cl::vector<float>(rows*columns);
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < columns; ++j)
            (*data->TS_vector)[i*columns + j] = initializer[i][j];

    TS_stateFlags |= StateFlags::DATA_UPLOADED;
}

DNN::BaseMatrix::BaseMatrix(int nbRow, int nbCol, cl::Buffer *existingBuffer, std::shared_ptr<CLMatrixSetup> setup) :
    rows(nbRow), columns(nbCol), data(new BufferLinkManager(this)), CLSetup(setup) {
    data->TS_buffer = existingBuffer;
    TS_stateFlags |= StateFlags::DATA_DOWNLOADED;
    //Buffer side creation : no thread unsafe danger
}

DNN::BaseMatrix::BaseMatrix(int nbRow, int nbCol, cl::vector<float> *existingVector, std::shared_ptr<CLMatrixSetup> setup)  :
    rows(nbRow), columns(nbCol), data(new BufferLinkManager(this)), CLSetup(setup) {
    data->TS_vector = existingVector;
    TS_stateFlags |= StateFlags::DATA_UPLOADED;
    //Buffer side creation : no thread unsafe danger
}

void DNN::BaseMatrix::move(DNN::BaseMatrix &&From, DNN::BaseMatrix &To, bool forceNonBlocking) noexcept {
    if(&From == &To || From.data == To.data || From.isEmpty()) return;

    const bool buffer = From.data->TS_buffer != nullptr;
    const bool vector = From.data->TS_vector != nullptr;
    if(To.getSize()  != From.getSize() || (buffer && vector) || (vector && forceNonBlocking)) { //First case : complete steal
        if(To.data != nullptr) To.data->registerForDeletion();
        To.data = From.stealData();

        //TS part
        To.lock();
        From.lock();

        To.TS_stateFlags = From.TS_stateFlags;
        To.TS_lastComputationEvent     = (cl::Event &&) From.TS_lastComputationEvent;
        To.TS_lastUploadEvent          = (cl::Event &&) From.TS_lastUploadEvent;
        To.TS_lastDownloadEvent        = (cl::Event &&) From.TS_lastDownloadEvent;

        To.data->TS_holder = &To;

        To.unlock();
        From.unlock();
    }
    else if(vector) { //Second case : host side affectation (waits for everything)
        To.waitForExternal();
        To.waitForUpload();
        To.waitForDownload();

        To.lock();
        From.lock();

        if(To.data->TS_vector != nullptr) delete To.data->TS_vector;
        To.data->TS_vector = From.data->TS_vector;

        To.TS_stateFlags &= oppFlag(StateFlags::INT_DOWNLOAD_FLAGS);
        To.TS_stateFlags |= StateFlags::DATA_UPLOADED;

        From.data->TS_vectorAccess = 0;
        From.data->TS_vector = nullptr;
        From.data->registerForDeletion();
        From.stealData();

        To.unlock();
        From.unlock();
    }
    else if(buffer) { //Third case : device side affectation (wait the less possible)
        //No need to protect as previous events and flags are erased...
        From.data->TS_vector = To.data->TS_vector;
        To.data->TS_vector = nullptr;
        To.data->registerForDeletion(); //No issue with external actions as TS_vector is saved...
        To.data = From.stealData();

        To.lock();
        From.lock();

        //External creation
        if(To.TS_stateFlags & (StateFlags::DATA_DOWNLOADING | StateFlags::EXTERNAL_DOWNLOADING)) { //Includes potential external download
            To.TS_stateFlags |= EXTERNAL_DOWNLOADING;
            To.data->addVectorEvent(); //TODO : How to avoid locking this mutex ??
            To.addDataCallback(To.TS_lastDownloadEvent, externalDownloadCallback);
        }
        if(To.TS_stateFlags & (StateFlags::DATA_UPLOADING | StateFlags::EXTERNAL_UPLOADING)) { //Includes potential external upload
            To.TS_stateFlags |= EXTERNAL_UPLOADING;
            To.data->addVectorEvent(); //TODO : How to avoid locking this mutex ??
            To.addDataCallback(To.TS_lastUploadEvent, externalUploadCallback);
        }

        //General TS changes
        To.TS_stateFlags &= oppFlag(StateFlags::INTERNAL_FLAGS);
        To.TS_stateFlags |= From.TS_stateFlags & StateFlags::INTERNAL_FLAGS;
        To.TS_lastComputationEvent = (cl::Event &&) From.TS_lastComputationEvent; //Will serve as deletion of the previous event

        To.data->TS_holder = &To;

        To.unlock();
        From.unlock();
    }

    //Basic finalisation, in every case From has become Empty...
    To.rows = From.rows;
    To.columns = From.columns;
    To.CLSetup = From.CLSetup;
}

void DNN::BaseMatrix::copy(const BaseMatrix &From, BaseMatrix &To) {
    if(&To == &From || To.data == From.data || From.isEmpty()) return;

    //Preparation for the copy (affectation behaviour, TS as no data registered for deletion)
    //ENH : Should we really delete all the previous data ??
    if(To.data != nullptr) To.data->registerForDeletion();
    To.data = new BufferLinkManager(&To);
    To.TS_stateFlags = StateFlags::NO_FLAG;
    To.TS_lastComputationEvent = (cl_event) nullptr; //Here the previous cl_event will be correctly released...
    To.TS_lastDownloadEvent    = (cl_event) nullptr;
    To.TS_lastUploadEvent      = (cl_event) nullptr;

    To.rows = From.rows;
    To.columns = From.columns;
    To.CLSetup = From.CLSetup; //To ensure it calls the override function...

    //Effective smart copy... (never copy both, if such a behaviour is wanted the user should use the static copy function)
    From.lock();
    if(From.TS_stateFlags & StateFlags::DATA_UPLOADED) {
        From.unlock(); //Using TS_vector is safe...

        To.data->TS_vector = new cl::vector<float>(*From.data->TS_vector);
        To.TS_stateFlags |= StateFlags::DATA_UPLOADED;
    }
    else if (From.TS_stateFlags & StateFlags::DATA_DOWNLOADED) {
        From.unlock();

        /// Mimic computation from now...
        cl::vector<cl::Event> events;
        events.reserve(1);
        From.manageBeforeReading(events);

        To.lock();
        To.data->TS_buffer =  new cl::Buffer (To.CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*From.getSize());
        To.data->addBufferEvent();
        To.TS_stateFlags |= StateFlags::COMPUTATION_EXECUTING | StateFlags::DATA_DOWNLOADED;

        cl::Event copyEvent;
        To.CLSetup->getQueue().enqueueCopyBuffer(
                *From.data->TS_buffer, *To.data->TS_buffer,
                0, 0, sizeof(float)*To.rows*To.columns,
                &events, &copyEvent
        );

        //Callbacks
        From.finishReading(copyEvent);
        To.finishComputing(copyEvent);
    }
    else assert(false); //Invalid state
}

//Data Access

float &DNN::BaseMatrix::getLValueElement(cl::size_type row, cl::size_type col) {
    assert(row < getRowCount() && col < getColumnCount());
    waitForResults();

    lock();
    TS_stateFlags &= oppFlag(
        StateFlags::DATA_DOWNLOADED | 
        StateFlags::DATA_DOWNLOADING
    );
    TS_stateFlags |= StateFlags::DATA_UPLOADED; //Should be useless...
    unlock();

    return (*data->TS_vector)[row * columns + col]; //Thread safe because it can't be registered for deletion...
}

float DNN::BaseMatrix::getRValueElement(cl::size_type row, cl::size_type col) const {
    assert(row < getRowCount() && col < getColumnCount());
    waitForConstResults();

    return (*data->TS_vector)[row * columns + col]; //Thread safe because it can't be registered for deletion...
}

//Data Management

void DNN::BaseMatrix::askForResults() const {
    if(isEmpty()) return;
    uploadData();
}

void DNN::BaseMatrix::waitForResults() const {
    if(isEmpty()) return;

    //Basically waits for any possible event (except readings !!)...
    uploadData();
    waitForExternal(); //Should not be necessary as an upload is requested...
    waitForDownload(); //In case the upload is skipped...
    waitForUpload();
}

void DNN::BaseMatrix::waitForConstResults() const {
    if(isEmpty()) return;

    uploadData();
    waitForExternal(); //Should not be necessary as an upload is requested...
    waitForUpload();
}

//Computation Management

void DNN::BaseMatrix::manageBeforeReading(cl::vector<cl::Event> &requiredEvents, bool download) const {
    assert(!isEmpty());

    //Data management
    if(download) downloadData();
    data->addBufferEvent();

    //Event management
    lock();
    if(TS_stateFlags & StateFlags::COMPUTATION_EXECUTING)
        requiredEvents.push_back(TS_lastComputationEvent);
    else if(TS_stateFlags & StateFlags::DATA_DOWNLOADING)
        requiredEvents.push_back(TS_lastDownloadEvent);
}

void DNN::BaseMatrix::manageBeforeComputing(cl::vector<cl::Event> &requiredEvents, int resRow, int resCol, bool download) {
    //BUG : Does not check reading event
    //BUG : Does not check CLSetup

    //Data management
    if(data == nullptr || getSize() != resRow * resCol) {
        if(data != nullptr) data->registerForDeletion();
        data = new BufferLinkManager(this);
        data->TS_buffer = new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float)*resCol*resRow);
    }
    else if(data->TS_buffer == nullptr) {
        data->TS_buffer = new cl::Buffer(CLSetup->getContext(), CL_MEM_READ_WRITE, sizeof(float) * resCol * resRow);
    }

    if(download) downloadData();
    data->addBufferEvent();

    //Event management
    lock();
    if(TS_stateFlags & StateFlags::COMPUTATION_EXECUTING)
        requiredEvents.push_back(TS_lastComputationEvent);
    else if(TS_stateFlags & StateFlags::DATA_DOWNLOADING)
        requiredEvents.push_back(TS_lastDownloadEvent);

    if(TS_stateFlags & StateFlags::DATA_UPLOADING)
        requiredEvents.push_back(TS_lastUploadEvent);

    if(additionalEvent.get() != nullptr) {
        requiredEvents.push_back(additionalEvent);
        additionalEvent = nullptr;
    }

    //State preparation
    TS_stateFlags &= oppFlag(StateFlags::INTERNAL_FLAGS);
    TS_stateFlags |= StateFlags::COMPUTATION_EXECUTING | StateFlags::DATA_DOWNLOADED;
    rows = resRow;
    columns = resCol;
}

//Flow Behavior

void DNN::BaseMatrix::downloadData() const {
    //TODO : Recheck up/download to see if it fits in the new architecture
    if(isEmpty() || data->TS_vector == nullptr) return;

    lock();
    if(TS_stateFlags & StateFlags::INT_DOWNLOAD_FLAGS) {
        unlock();
        return;
    }
    unlock();

    //Data control (matrix must be unlocked)
    if(data->TS_buffer == nullptr)
        data->TS_buffer = new cl::Buffer (CLSetup->getContext(), CL_MEM_READ_ONLY, sizeof(float)*rows*columns);

    data->waitForBufferEvents();
    data->addBufferEvent();
    data->addVectorEvent();

    //Event management
    lock();
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

    addDataCallback(TS_lastDownloadEvent, downloadCallback);
    unlock();
}

void DNN::BaseMatrix::uploadData() const {
    if(isEmpty() || data->TS_buffer == nullptr) return;

    lock();
    if(TS_stateFlags & (StateFlags::INT_UPLOAD_FLAGS)) {
        unlock();
        return;
    }
    unlock();

    //Data control
    if(data->TS_vector == nullptr) //Create buffer if no buffer exist for now
        data->TS_vector = new cl::vector<float>(rows*columns);

    data->addBufferEvent();
    data->addVectorEvent();

    //Event management
    lock();
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

    addDataCallback(TS_lastUploadEvent, uploadCallback);
    unlock();
}

void DNN::BaseMatrix::waitForExternal() const {
    if(isEmpty()) return;

    lock();
    if(TS_stateFlags & StateFlags::EXTERNAL_DOWNLOADING) {
        safelyWaitForEvent(TS_lastDownloadEvent, waitingDownloadMutex);
        TS_lastDownloadEvent = (cl_event) nullptr;//Previous cl_event is correctly released...
    }
    if(TS_stateFlags & StateFlags::EXTERNAL_UPLOADING) {
        safelyWaitForEvent(TS_lastUploadEvent, waitingUploadMutex);
        TS_lastUploadEvent = (cl_event) nullptr; //Previous cl_event is correctly released...
    }
    
    TS_stateFlags &= oppFlag(StateFlags::EXTERNAL_FLAGS);    
    unlock();
}

void DNN::BaseMatrix::waitForDownload() const {
    if(isEmpty()) return;

    lock();
    if(!(TS_stateFlags & StateFlags::EXTERNAL_DOWNLOADING) && TS_lastDownloadEvent.get() != nullptr)
        safelyWaitForEvent(TS_lastDownloadEvent, waitingDownloadMutex); // External downloads are not waited for...
    
    TS_lastDownloadEvent = (cl_event) nullptr;
    unlock();
}

void DNN::BaseMatrix::waitForComputation() const {
    if(isEmpty()) return;

    lock();
    if(TS_lastComputationEvent.get() != nullptr)
        safelyWaitForEvent(TS_lastComputationEvent, waitingComputationMutex, false);

    TS_lastComputationEvent = (cl_event) nullptr; //Previous cl_event is correctly released...
    unlock();
}

void DNN::BaseMatrix::waitForUpload() const {
    if(isEmpty()) return;

    lock();
    if(!(TS_stateFlags & StateFlags::EXTERNAL_UPLOADING) && TS_lastUploadEvent.get() != nullptr)
        safelyWaitForEvent(TS_lastUploadEvent, waitingUploadMutex);
    
    TS_lastUploadEvent = (cl_event) nullptr; //Previous cl_event is correctly released...
    unlock();
}

void DNN::BaseMatrix::safelyWaitForEvent(cl::Event &event, std::mutex &waitingMutex, bool relock) const {
    if(isEmpty()) return;

    waitingMutex.lock();
    unlock();

    event.wait(); //Supposed to be read only on the cl_event underlying pointer...

    waitingMutex.unlock();
    if(relock) lock();
}

//Device Callbacks

void DNN::BaseMatrix::addDataCallback(cl::Event &event, void (CL_CALLBACK *cb)(cl_event, cl_int, void *)) const {
    clRetainEvent(event.get());
    event.setCallback(CL_COMPLETE, cb, (void *) data);
}

void CL_CALLBACK DNN::BaseMatrix::computationCallback(cl_event event, cl_int, void *_linkManager) {
    //TODO : Avoid C-style cast
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    BaseMatrix *holder = linkManager->TS_holder;
    if(holder != nullptr && holder->TS_lastComputationEvent.get() == event) {
        holder->TS_stateFlags |= StateFlags::COMPUTATION_EXECUTED;
        holder->TS_stateFlags &= oppFlag(StateFlags::COMPUTATION_EXECUTING);

        //Previous cl_event is correctly released...
        if(holder->waitingComputationMutex.try_lock()) {
            holder->TS_lastComputationEvent = (cl_event) nullptr;
            holder->waitingComputationMutex.unlock();
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks(linkManager, true, false);
    clReleaseEvent(event);
    std::cout << "End computation" << std::endl;
}

void CL_CALLBACK DNN::BaseMatrix::downloadCallback(cl_event event, cl_int, void *_linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    BaseMatrix *holder = linkManager->TS_holder;
    if(holder != nullptr && holder->TS_lastDownloadEvent.get() == event) {
        holder->TS_stateFlags |= StateFlags::DATA_DOWNLOADED;
        holder->TS_stateFlags &= oppFlag(StateFlags::DATA_DOWNLOADING);

        //Previous cl_event is correctly released...
        if(holder->waitingDownloadMutex.try_lock()) {
            holder->TS_lastDownloadEvent = (cl_event) nullptr;
            holder->waitingDownloadMutex.unlock();
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks(linkManager, true, true);
    clReleaseEvent(event);
    std::cout << "End download" << std::endl;
}

void CL_CALLBACK DNN::BaseMatrix::uploadCallback(cl_event event, cl_int, void *_linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    BaseMatrix *holder = linkManager->TS_holder;
    if(holder != nullptr && holder->TS_lastUploadEvent.get() == event) {
        holder->TS_stateFlags |= StateFlags::DATA_UPLOADED;
        holder->TS_stateFlags &= oppFlag(StateFlags::DATA_UPLOADING);

        //Previous cl_event is correctly released...
        if(holder->waitingUploadMutex.try_lock()) {
            holder->TS_lastUploadEvent = (cl_event) nullptr;
            holder->waitingUploadMutex.unlock();
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks(linkManager, true, true);
    clReleaseEvent(event);
    std::cout << "End upload" << std::endl;
}

void CL_CALLBACK DNN::BaseMatrix::readCallback(cl_event event, cl_int, void* _linkManager) {
    checkDeletionForCallbacks((BufferLinkManager *) _linkManager, true, false);
    clReleaseEvent(event);
}

void CL_CALLBACK DNN::BaseMatrix::externalDownloadCallback(cl_event event, cl_int, void *_linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    BaseMatrix *holder = linkManager->TS_holder;
    if(holder != nullptr && holder->TS_lastDownloadEvent.get() == event) {
        holder->TS_stateFlags &= oppFlag(StateFlags::EXTERNAL_DOWNLOADING); //No issue with multiple externals (for one holder only the last is registered)

        //Previous cl_event is correctly released...
        if(holder->waitingDownloadMutex.try_lock()) {
            holder->TS_lastDownloadEvent = (cl_event) nullptr;
            holder->waitingDownloadMutex.unlock();
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks((BufferLinkManager *) _linkManager, false, true);
    clReleaseEvent(event);
}

void CL_CALLBACK DNN::BaseMatrix::externalUploadCallback(cl_event event, cl_int, void* _linkManager) {
    BufferLinkManager *linkManager = (BufferLinkManager *) _linkManager;

    linkManager->internalLinkMutex.lock();
    BaseMatrix *holder = linkManager->TS_holder;
    if(holder != nullptr && holder->TS_lastUploadEvent.get() == event) {
        holder->TS_stateFlags &= oppFlag(StateFlags::EXTERNAL_UPLOADING); //No issue with multiple externals (for one holder only the last is registered)

        //Previous cl_event is correctly released...
        if(holder->waitingUploadMutex.try_lock()) {
            holder->TS_lastUploadEvent = (cl_event) nullptr;
            holder->waitingUploadMutex.unlock();
        }
    }
    linkManager->internalLinkMutex.unlock();

    checkDeletionForCallbacks((BufferLinkManager *) _linkManager, false, true);
    clReleaseEvent(event);
}

void CL_CALLBACK DNN::BaseMatrix::checkDeletionForCallbacks(BufferLinkManager *linkManager, bool buffer, bool vector) {
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
