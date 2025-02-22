#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/opencl.hpp>
#include <CL/Utils/Utils.hpp>

#include <cassert>
#include <mutex>
#include <atomic>
#include <iostream>
#include <iomanip>

namespace DNN {
    class Matrix;

    struct RowAccesser {
        RowAccesser() = delete;
        RowAccesser(int rowIndex, Matrix &matrix) : row(rowIndex), linkedMatrix(matrix) {}
        float &operator[](cl::size_type col);

        const cl::size_type row;
        Matrix &linkedMatrix;      
    };

    struct CLMatrixSetup {
        CLMatrixSetup();

        cl::Context context;
        cl::CommandQueue queue;
        cl::Program program;

        cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &> addKer;
        cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &> transAddKer;

        cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &> subKer;
        cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &> transSubKer;

        cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, int> prodKer;
        cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, int> transLProdKer;
        cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, int> transRProdKer;

        cl::KernelFunctor<cl::Buffer &, cl::Buffer &> oppKer;
    };

    //ENH : Make prepare string constexpr
    //ENH : A big think about how is organised the CL setup must be done...
    class VectorisedFunction {
        public:
            VectorisedFunction() = delete;
            VectorisedFunction(const cl::string &operation, CLMatrixSetup &setup);

            Matrix operator()(Matrix &arg);
            static cl::string prepareString(const cl::string &operation);
        protected:
            CLMatrixSetup &setup;
            cl::KernelFunctor<cl::Buffer &, cl::Buffer &> kernel;

            static const cl::string preKernelStr;
            static const cl::string postKernelStr;
            static const cl::string indicator;
    };

    class BufferLinkManager {
    public:
        BufferLinkManager() { ++DEBUG_created; };
        BufferLinkManager(Matrix *holder) : TS_holder(holder) { ++DEBUG_created; }
        BufferLinkManager(BufferLinkManager  &toCopy) = delete; //Copy must be done in correct conditions, directly by the matrix...
        BufferLinkManager(BufferLinkManager &&toMove) = delete; //Move must be done in correct conditions, directly by the matrix...
        ~BufferLinkManager();

        void registerForDeletion();
        inline void addBufferEvent() { std::lock_guard<std::recursive_mutex> lock(internalLinkMutex); ++TS_bufferAccess; }
        inline void addVectorEvent() { std::lock_guard<std::recursive_mutex> lock(internalLinkMutex); ++TS_vectorAccess; }
        void waitForBufferEvents();

        static std::atomic<int> DEBUG_created;   
        static std::atomic<int> DEBUG_destroyed;   
    private:
        mutable bool TS_tobeDeleted  = false;
        mutable volatile unsigned long long TS_bufferAccess = 0;
        mutable volatile unsigned long long TS_vectorAccess = 0;

        Matrix *TS_holder                      = nullptr;
        cl::Buffer * volatile TS_buffer        = nullptr; //TS unless registred for deletion...
        cl::vector<float> * volatile TS_vector = nullptr; //TS unless registred for deletion...

        //The matrix is in charge of locking this mutex when making direct access...
        std::recursive_mutex internalLinkMutex;

        friend class Matrix;
    };

    //ENH : Add mutable and in link to what is seen by the user...
    //ENH : Replace assert by exception
    //TODO : Check every usage of rows/columns, it should be real dimension of the matrix
    //TODO : Check every usage of get...
    //TODO : avoid assert in private methods and maybe also for protected nah ? 
    class Matrix {
    public:
        //Host side creation
        Matrix() = delete; //Dimensions must be fixed...
        Matrix(int nbRow = 0, int nbCol = 0, float expr = 0.0);
        Matrix(const cl::vector<cl::vector<float>> &initialiser);

        //Affectation creation (behaves smartly...)
        Matrix(Matrix  &toCopy);
        Matrix(Matrix &&toMove) noexcept;
        virtual ~Matrix();

        Matrix &operator=(Matrix &toCopy);
        Matrix &operator=(Matrix &&toMove) noexcept;

        //Operations' public library (we don't want them to be virtual : adapted return type)
        Matrix operator+(Matrix &operand); //ENH : Add rvalue version ?
        Matrix operator-(Matrix &operand);
        Matrix operator*(Matrix &operand);
        
        //ENH : Add new operators...
        // Matrix operator^(int power);
        // Matrix operator+(float scalar);
        // Matrix operator-(float scalar);
        // Matrix operator*(float scalar);

        Matrix operator-();

        Matrix executeKernel(cl::KernelFunctor<cl::Buffer &, cl::Buffer &> kernel);

        //Data access (we dont want them to be virtual : adapted return type and parameters)
        virtual inline int getRowCount()    const {return transpose ? columns : rows; }
        virtual inline int getColumnCount() const {return transpose ? rows : columns; }

        inline RowAccesser operator[](cl::size_type row) { return RowAccesser(row, *this); }
        float &getLValueElement(cl::size_type row, cl::size_type col);
        float getRValueElement(cl::size_type row, cl::size_type col);

        //Public data management
        void askForResults();// Ask for upload if not uploaded
        void waitForResults(); //Ask for upload if not uploaded and wait for it
        void waitForConstResults();
        
        inline bool areComputationsFinished() 
            { return TS_stateFlags & StateFlags::COMPUTATION_EXECUTED; }
        inline bool areResultsAvailable() 
            { return TS_stateFlags & StateFlags::DATA_UPLOADED; }
        inline bool isValid() const
            { return data != nullptr && (data->TS_vector != nullptr || data->TS_buffer != nullptr); }
        inline operator bool() { return isValid(); }

        //Calculation management
        static CLMatrixSetup *defaultCLSetup; 
                
    protected:
        //Operations' library (to allow any derived type as return without copy)
        static void opAdd(Matrix &A, Matrix &B, Matrix &R);
        static void opSub(Matrix &A, Matrix &B, Matrix &R);
        static void opMul(Matrix &A, Matrix &B, Matrix &R);
        static void opOpp(Matrix &A, Matrix &R);

        //Automatic data management
        Matrix(int nbRow, int nbCol, cl::Buffer *existingBuffer);  //Internal device side creation
        Matrix(int nbRow, int nbCol, cl::vector<float> *existingVector);  //Internal host side creation (for derived classes)
        void mangageBeforeComputation(cl::vector<cl::Event> &requiredEvents);

        void waitForExternal();
        void waitForDownload();
        void waitForComputation();
        void waitForUpload();
        void safelyWaitForEvent(cl::Event &event, std::mutex &waitingMutex, bool relock = true);

        void downloadData();
        void uploadData();

        bool transpose = false;
        int rows    = 0;
        int columns = 0;
        
        CLMatrixSetup *CLSetup = defaultCLSetup; //ENH : Risky nah ? How should it behave on copy ??

        //OpenCL Callbacks
        static void addDataCallbackTo(cl::Event &event, void (* cb) (cl_event, cl_int, void *), BufferLinkManager *arg);

        //TODO : For all callback affecting flags -> the code is the same, make a specific function...
        static void CL_CALLBACK computationCallback(cl_event event, cl_int, void* _linkManager);//Release the event
        static void CL_CALLBACK downloadCallback(cl_event event, cl_int, void* _linkManager);//Release the event
        static void CL_CALLBACK uploadCallback(cl_event event, cl_int, void* _linkManager);//Release the event
        static void CL_CALLBACK readCallback(cl_event event, cl_int, void* _linkManager);//Release the event
        static void CL_CALLBACK externaDownloadlCallback(cl_event event, cl_int, void* _linkManager);
        static void CL_CALLBACK externalUploadCallback(cl_event event, cl_int, void* _linkManager);
        static void CL_CALLBACK checkDeletionForCallbacks(BufferLinkManager *linkManager, bool buffer, bool vector);

    private: 
        //Host behavior
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
            INT_DOWNLOAD_FLAGS      =
                DATA_DOWNLOADED         | DATA_DOWNLOADING      ,
            INT_UPLOAD_FLAGS        =
                DATA_UPLOADED           | DATA_UPLOADING        ,
            INT_COMPUTATION_FLAGS   =
                COMPUTATION_EXECUTED    | COMPUTATION_EXECUTING ,
            INTERNAL_FLAGS          = 
                INT_DOWNLOAD_FLAGS      | INT_UPLOAD_FLAGS      | INT_COMPUTATION_FLAGS,
            EXTERNAL_FLAGS          = 
                EXTERNAL_DOWNLOADING    | EXTERNAL_UPLOADING    
        };
        static inline constexpr uint8_t oppFlag(uint8_t flags) { return 255 - flags; };
        volatile uint8_t TS_stateFlags = StateFlags::NO_FLAG;

        //OpenCL behavior
        //The wrapped cl_event should be null if no corresponding command is executed (includes externals).
        mutable cl::Event TS_lastComputationEvent;
        mutable cl::Event TS_lastUploadEvent;
        mutable cl::Event TS_lastDownloadEvent;

        BufferLinkManager *data = nullptr;
        mutable std::recursive_mutex promptStateMutex; //To protect TS_... variable when they are not used in blocking CL calls.
        mutable std::mutex waitingUploadMutex;
        mutable std::mutex waitingDownloadMutex;
        mutable std::mutex waitingComputationMutex;

        friend class VectorisedFunction;
    };


    //Other operators for the Matrix class
    std::ostream &operator<<(std::ostream &output, DNN::Matrix &matrix);
    // Matrix operator*(float scalar, Matrix &operand);
};

