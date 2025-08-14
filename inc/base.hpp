#pragma once

#include <CL/opencl.hpp>
#include <CL/Utils/Utils.hpp>

#include <cassert>
#include <memory>
#include <mutex>
#include <atomic>
#include <iostream>
#include <unordered_map>

namespace DNN {
    class BaseMatrix;

    //ENH : Make it an inner class of the BaseMatrix class
    class CLMatrixSetup {
    public :
        static std::shared_ptr<CLMatrixSetup> getDefault();

        CLMatrixSetup(cl::Context _context)  : context(_context), queue(context) {}
        CLMatrixSetup(cl::Context _context, cl::CommandQueue _queue) : context(_context), queue(_queue)  {}

        bool addKernelsFromSource(const char *file, cl::vector<cl::string> kernels, int8_t libCode = 0);
        bool addKernelsFromProgram(cl::Program program, cl::vector<cl::string> kernels, int8_t libCode = 0);

        cl::Context getContext() { return context; } //Copy is not an issue as it will only copy the wrapped pointer...
        cl::CommandQueue getQueue() { return queue; }
        cl::Kernel getKernel(cl::string key) { return internalKernelLib[key]; }
    protected:
        cl::Context context;
        cl::CommandQueue queue;

        std::unordered_map<cl::string, cl::Kernel> internalKernelLib;
        uint8_t includedLibraries = 0;

        //Default singleton management
        CLMatrixSetup() : context(cl::Context::getDefault()), queue(context)  {}
        static std::shared_ptr<CLMatrixSetup> defaultCLSetup;
    };

    class BufferLinkManager {
    public:
        BufferLinkManager() { ++DEBUG_created; };
        BufferLinkManager(BaseMatrix *holder) : TS_holder(holder) { ++DEBUG_created; }
        BufferLinkManager(BufferLinkManager  &toCopy) = delete; //Copy must be done in correct conditions, directly by the BaseMatrix...
        BufferLinkManager(BufferLinkManager &&toMove) = delete; //Move must be done in correct conditions, directly by the BaseMatrix...
        ~BufferLinkManager();

        void registerForDeletion();
        void addBufferEvent() const { std::lock_guard<std::recursive_mutex> lock(internalLinkMutex); ++TS_bufferAccess; }
        void addVectorEvent() const { std::lock_guard<std::recursive_mutex> lock(internalLinkMutex); ++TS_vectorAccess; }
        void waitForBufferEvents() const;

        static std::atomic<int> DEBUG_created;   
        static std::atomic<int> DEBUG_destroyed;   
    private:
        bool TS_tobeDeleted  = false;
        mutable volatile unsigned long long TS_bufferAccess = 0;
        mutable volatile unsigned long long TS_vectorAccess = 0;

        BaseMatrix * volatile TS_holder        = nullptr;
        cl::Buffer * volatile TS_buffer        = nullptr; //TS unless registered for deletion...
        cl::vector<float> * volatile TS_vector = nullptr; //TS unless registered for deletion...

        //The BaseMatrix is in charge of locking this mutex when making direct access...
        mutable std::recursive_mutex internalLinkMutex;

        friend class BaseMatrix;
    };

    //ENH : Add mutable and in link to what is seen by the user...
    //ENH : Replace assert by exception
    //ENH : avoid assert in private methods and maybe also for protected nah ?
    //TODO : Sort needed public functions and method that must remain private
    //BUG : Empty must not lead to any crash of the program...
    class BaseMatrix {
    public:
        //Constructors
        BaseMatrix(std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        BaseMatrix(int nbRow, int nbCol, float expr = 0.0, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        BaseMatrix(const cl::vector<cl::vector<float>> &initializer, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

        BaseMatrix(int nbRow, int nbCol, cl::Buffer *existingBuffer       , std::shared_ptr<CLMatrixSetup> setup);  //Internal device side creation
        BaseMatrix(int nbRow, int nbCol, cl::vector<float> *existingVector, std::shared_ptr<CLMatrixSetup> setup);  //Internal host side creation (for derived classes)

        BaseMatrix(const BaseMatrix  &toCopy) { copy(toCopy, *this); }
        BaseMatrix(BaseMatrix &&toMove) noexcept { move((BaseMatrix &&) toMove, *this); }
        ~BaseMatrix();

        static void move(BaseMatrix &&From, BaseMatrix &To, bool forceNonBlocking = false) noexcept;
        static void copy(const BaseMatrix &From, BaseMatrix &To);

        //Data Access
        int getRowCount()    const {return rows; }
        int getColumnCount() const {return columns; }
        int getSize()        const {return columns * rows;}

        float &getLValueElement(cl::size_type row, cl::size_type col);
        float getRValueElement(cl::size_type row, cl::size_type col) const;

        std::shared_ptr<CLMatrixSetup> getCLSetup() const { return CLSetup; }
        cl::Buffer &getCLData() { return *data->TS_buffer; } //BUG : What if one of this thing is null ?

        //Data Management
        //BUG : Not thread safe
        bool areComputationsFinished() const;
        bool areConstResultsAvailable() const;
        bool areResultsAvailable() const;
        bool isEmpty() const;

        void askForResults() const;// Ask for upload if not uploaded
        void waitForResults() const; //Ask for upload if not uploaded and wait for it
        void waitForConstResults() const;

        //Computation Management
        void manageBeforeReading(cl::vector<cl::Event> &requiredEvents, bool download = true) const;
        void manageBeforeComputing(cl::vector<cl::Event> &requiredEvents, int resRow, int resCol, bool download = false); //BUG : Does not take in account reading events for R !!

        void finishReading(cl::Event &event) const;
        void finishComputing(cl::Event &event);

        void registerAdditionalComputing(const BaseMatrix &res) { res.lock(); additionalEvent = res.TS_lastComputationEvent; res.unlock(); }

        template<typename... Ts>
        static void basicUnaryOp(const BaseMatrix &A, BaseMatrix &R, int rows, int columns, cl::KernelFunctor<cl::Buffer &, cl::Buffer &, Ts...> kernel, Ts... args);
        template<typename... Ts>
        static void basicUnaryOp(const BaseMatrix &A, BaseMatrix &R, int rows, int columns, cl::string kernelName, Ts... args);
        template<typename... Ts>
        static void basicBinaryOp(const BaseMatrix &A, const BaseMatrix &B, BaseMatrix &R, int rows, int columns, cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, Ts...> kernel, Ts... args);
        template<typename... Ts>
        static void basicBinaryOp(const BaseMatrix &A, const BaseMatrix &B, BaseMatrix &R, int rows, int columns, cl::string kernelName, Ts... args);

    private:
        ///Flow Behavior

        //Data exchanges
        void downloadData() const;
        void uploadData() const;

        //Waiting
        void waitForExternal() const;
        void waitForDownload() const;
        void waitForComputation() const;
        void waitForUpload() const;
        void safelyWaitForEvent(cl::Event &event, std::mutex &waitingMutex, bool relock = true) const;

        //Threads management
        void lock() const { if(data) data->internalLinkMutex.lock(); }
        void unlock() const {if(data) data->internalLinkMutex.unlock(); }

        /// Host Behavior

        //Size
        int rows    = 0;
        int columns = 0;

        //State
        enum StateFlags : uint8_t {
            NO_FLAG                 = 0,
            DATA_DOWNLOADED         = 1 << 0, //Data from host vector is accessible in buffer.
            DATA_DOWNLOADING        = 1 << 1, //Data from host vector has been requested for the buffer.
            DATA_UPLOADED           = 1 << 2, //Data from **all** computations is accessible in host vector.
            DATA_UPLOADING          = 1 << 3, //Data from buffer has been requested and no new computations since.
            COMPUTATION_EXECUTED    = 1 << 4,
            COMPUTATION_EXECUTING   = 1 << 5,
            EXTERNAL_DOWNLOADING    = 1 << 6, //External means an external **buffer** (and not an external vector...)
            EXTERNAL_UPLOADING      = 1 << 7,

            //Flag groups
            INT_DOWNLOAD_FLAGS      = DATA_DOWNLOADED         | DATA_DOWNLOADING      ,
            INT_UPLOAD_FLAGS        = DATA_UPLOADED           | DATA_UPLOADING        ,
            INT_COMPUTATION_FLAGS   = COMPUTATION_EXECUTED    | COMPUTATION_EXECUTING ,
            INTERNAL_FLAGS          = INT_DOWNLOAD_FLAGS      | INT_UPLOAD_FLAGS      | INT_COMPUTATION_FLAGS,
            EXTERNAL_FLAGS          = EXTERNAL_DOWNLOADING    | EXTERNAL_UPLOADING
        };
        static constexpr uint8_t oppFlag(uint8_t flags) { return 255 - flags; };
        mutable volatile uint8_t TS_stateFlags = StateFlags::NO_FLAG;

        ///Device Behavior

        //Data
        std::shared_ptr<CLMatrixSetup> CLSetup = nullptr;
        BufferLinkManager *data = nullptr;
        BufferLinkManager *stealData();

        //Events : the wrapped cl_event should be null if no corresponding command is executed
        mutable cl::Event TS_lastComputationEvent;
        mutable cl::Event TS_lastUploadEvent;
        mutable cl::Event TS_lastDownloadEvent;

        mutable std::mutex waitingUploadMutex;
        mutable std::mutex waitingDownloadMutex;
        mutable std::mutex waitingComputationMutex;
        mutable std::mutex waitingReadingMutex;

        mutable cl::Event additionalEvent;

        //Callbacks
        //ENH : For all callback affecting flags -> the code is the same, make a specific function...
        void addDataCallback(cl::Event &event, void (CL_CALLBACK* cb) (cl_event, cl_int, void *)) const;

        static void CL_CALLBACK computationCallback(cl_event event, cl_int, void* _linkManager);//Release the event
        static void CL_CALLBACK downloadCallback(cl_event event, cl_int, void* _linkManager);//Release the event
        static void CL_CALLBACK uploadCallback(cl_event event, cl_int, void* _linkManager);//Release the event
        static void CL_CALLBACK readCallback(cl_event event, cl_int, void* _linkManager);//Release the event
        static void CL_CALLBACK externalDownloadCallback(cl_event event, cl_int, void* _linkManager);
        static void CL_CALLBACK externalUploadCallback(cl_event event, cl_int, void* _linkManager);
        static void CL_CALLBACK checkDeletionForCallbacks(BufferLinkManager *linkManager, bool buffer, bool vector);
    };

    ///Definitions

    //Template
    template <typename... Ts>
    inline void BaseMatrix::basicUnaryOp(const BaseMatrix &A, BaseMatrix &R, int rows, int columns, cl::KernelFunctor<cl::Buffer &, cl::Buffer &, Ts...> kernel, Ts... args) {
        //Prepare events
        cl::vector<cl::Event> events;
        events.reserve(4);
        A.manageBeforeReading(events); //WARNING : It locks the matrix !!!!
        R.manageBeforeComputing(events, rows, columns);

        //Actual computations...
        cl::CommandQueue queue = A.CLSetup->getQueue();
        cl::Event computeEvent = kernel( //Here the previous cl_event will be correctly released...
            cl::EnqueueArgs(queue, events, cl::NDRange(R.rows, R.columns)),
            *A.data->TS_buffer,
            *R.data->TS_buffer,
            args...
        );

        //Callbacks
        A.finishReading(computeEvent);
        R.finishComputing(computeEvent);
    }

    template <typename... Ts>
    inline void BaseMatrix::basicUnaryOp(const BaseMatrix &A, BaseMatrix &R, int rows, int columns, cl::string kernelName, Ts... args) {
        basicUnaryOp(A, R, rows, columns, cl::KernelFunctor<cl::Buffer &, cl::Buffer &, Ts...>(A.getCLSetup()->getKernel(kernelName)), args...);
    }

    template <typename... Ts>
    inline void BaseMatrix::basicBinaryOp(const BaseMatrix &A, const BaseMatrix &B, BaseMatrix &R, int rows, int columns, cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, Ts...> kernel, Ts... args) {
        //Prepare operands
        cl::vector<cl::Event> events;
        events.reserve(5);
        A.manageBeforeReading(events);
        B.manageBeforeReading(events);
        R.manageBeforeComputing(events, rows, columns);

        //Actual computations...
        cl::CommandQueue queue = A.CLSetup->getQueue();
        cl::Event computeEvent = kernel( //Here the previous cl_event will be correctly released...
            cl::EnqueueArgs(queue, events, cl::NDRange(R.rows, R.columns)),
            *A.data->TS_buffer,
            *B.data->TS_buffer,
            *R.data->TS_buffer,
            args...
        );

        //Callbacks
        A.finishReading(computeEvent);
        B.finishReading(computeEvent);
        R.finishComputing(computeEvent);
    }

    template <typename... Ts>
    inline void BaseMatrix::basicBinaryOp(const BaseMatrix &A, const BaseMatrix &B, BaseMatrix &R, int rows, int columns, cl::string kernelName, Ts... args) {
        basicBinaryOp(A, B, R, rows, columns, cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, Ts...>(A.getCLSetup()->getKernel(kernelName)), args...);
    }

    //Inline
    inline DNN::BaseMatrix::BaseMatrix(std::shared_ptr<CLMatrixSetup> setup) : rows(0), columns(0), data(nullptr), CLSetup(setup) {
        TS_stateFlags |= StateFlags::DATA_UPLOADED | StateFlags::DATA_DOWNLOADED;
    }
    inline DNN::BaseMatrix::BaseMatrix(int nbRow, int nbCol, float expr, std::shared_ptr<CLMatrixSetup> setup) :
        rows(nbRow), columns(nbCol), data(new BufferLinkManager(this)), CLSetup(setup) {
        data->TS_vector = new cl::vector<float>(rows*columns, expr);
        TS_stateFlags |= StateFlags::DATA_UPLOADED;
    }
    inline DNN::BaseMatrix::~BaseMatrix() {
        if(data != nullptr) data->registerForDeletion();
    }

    inline bool BaseMatrix::areComputationsFinished() const {
        return TS_stateFlags & StateFlags::COMPUTATION_EXECUTED;
    }
    inline bool BaseMatrix::areConstResultsAvailable() const{
        return !(TS_stateFlags & StateFlags::DATA_UPLOADED) && !(TS_stateFlags & StateFlags::EXTERNAL_FLAGS);
    }
    inline bool BaseMatrix::areResultsAvailable() const {
        return areConstResultsAvailable() && !(TS_stateFlags & StateFlags::DATA_DOWNLOADING);
    }
    inline bool BaseMatrix::isEmpty() const {
        return data == nullptr || (data->TS_vector == nullptr && data->TS_buffer == nullptr);
    }

    inline void BaseMatrix::finishReading(cl::Event &event) const {
        addDataCallback(event, readCallback);
        unlock();
    }
    inline void BaseMatrix::finishComputing(cl::Event &event) {
        TS_lastComputationEvent = event;
        addDataCallback(event, computationCallback);
        unlock();
    }

    inline BufferLinkManager *BaseMatrix::stealData() {
        BufferLinkManager * const temp = data;
        data = nullptr;
        return temp;
    }
}
