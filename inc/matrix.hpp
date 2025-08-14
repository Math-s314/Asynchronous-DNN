#pragma once

#include <CL/opencl.hpp>
#include <CL/Utils/Utils.hpp>

#include "base.hpp"

#include <cassert>
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

    //ENH : Make prepare string constexpr
    class VectorisedFunction {
    public:
        VectorisedFunction() = delete;

        VectorisedFunction(const cl::string &operation, std::weak_ptr<CLMatrixSetup> setup);

        Matrix operator()(const Matrix &arg) const;

        static cl::string prepareString(const cl::string &operation);

    protected:
        std::weak_ptr<CLMatrixSetup> setup;
        mutable cl::KernelFunctor<cl::Buffer &, cl::Buffer &> kernel;

        static const cl::string preKernelStr;
        static const cl::string postKernelStr;
        static const cl::string indicator;
    };

    //ENH : Add mutable and in link to what is seen by the user...
    //ENH : Replace assert by exception
    //ENH : avoid assert in private methods and maybe also for protected nah ?
    class Matrix {
    public:
        //Constructors
        Matrix(std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        Matrix(int nbRow, int nbCol, float expr = 0.0, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        Matrix(const cl::vector<cl::vector<float>> &initializer, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

        Matrix(const Matrix &toCopy) = default;
        Matrix(Matrix &&toMove) noexcept = default;
        virtual ~Matrix() = default;

        Matrix &operator=(const Matrix &toCopy);
        Matrix &operator=(Matrix &&toMove) noexcept;

        [[deprecated]]
        Matrix executeKernel(cl::KernelFunctor<cl::Buffer &, cl::Buffer &> &kernel) const { return Matrix(); }

        //Data Access
        int getRowCount()    const { return internalMatrix.getRowCount(); }
        int getColumnCount() const { return internalMatrix.getColumnCount(); }

        RowAccesser operator[](cl::size_type row) { return RowAccesser(row, *this); }
        float &getLValueElement(cl::size_type row, cl::size_type col) { return internalMatrix.getLValueElement(row, col); }
        float getRValueElement(cl::size_type row, cl::size_type col) const { return internalMatrix.getRValueElement(row, col); }

        //Data Management
        void askForResults()       const { internalMatrix.askForResults(); }// Ask for upload if not uploaded
        void waitForResults()      const { internalMatrix.waitForResults(); } //Ask for upload if not uploaded and wait for it
        void waitForConstResults() const { internalMatrix.waitForConstResults(); }

        bool areComputationsFinished()  const { return internalMatrix.areComputationsFinished(); }
        bool areConstResultsAvailable() const { return internalMatrix.areConstResultsAvailable(); }
        bool areResultsAvailable()      const { return internalMatrix.areResultsAvailable(); }
        bool isEmpty()                  const { return internalMatrix.isEmpty(); }
        operator bool()                 const { return internalMatrix.isEmpty(); }

        std::shared_ptr<CLMatrixSetup> getCLSetup() const { return internalMatrix.getCLSetup(); }

        /// Operations' library

        //In-place operations
        Matrix &neg_IP();
        Matrix &hadamardProd_IP(const Matrix &operand);
        Matrix &operator+=(const Matrix &operand);
        Matrix &operator-=(const Matrix &operand);

        //Internal versions (avoiding copies)
        static void _add(const Matrix &A, const Matrix &B, Matrix &R);
        static void _sub(const Matrix &A, const Matrix &B, Matrix &R);
        static void _mul(const Matrix &A, const Matrix &B, Matrix &R);
        static void _hadamard(const Matrix &A, const Matrix &B, Matrix &R);
        static void _neg(const Matrix &A, Matrix &R);

    protected:
        //Kernel library
        static constexpr uint8_t libCode = 1 << 0;
        static constexpr char libFile[] = "ocl/matrix.ocl";
        virtual void buildCLSetup();

        static BaseMatrix &getBase(Matrix &target) { return target.internalMatrix; }
        static const BaseMatrix &getBase(const Matrix &target) { return target.internalMatrix; }
        BaseMatrix internalMatrix;

        //For derived class
        //TODO : Add flat constructors to avoid dealing with pointers at this level
        Matrix(int row, int col, cl::vector<float> *existingData, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
    };

    /// Operators

    //Arithmetic symbols
    Matrix operator+(const Matrix &A, const Matrix &B);
    Matrix operator-(const Matrix &A, const Matrix &B);
    Matrix operator*(const Matrix &A, const Matrix &B);
    Matrix operator-(const Matrix &A);

    //Functions
    Matrix hadamardProduct(const Matrix &A, const Matrix &B);

    //Others
    std::ostream &operator<<(std::ostream &output, DNN::Matrix &matrix);


    /// Inline definitions

    inline Matrix::Matrix(std::shared_ptr<CLMatrixSetup> setup) : internalMatrix(setup) {
        Matrix::buildCLSetup();
    }
    inline Matrix::Matrix(int nbRow, int nbCol, float expr, std::shared_ptr<CLMatrixSetup> setup) : internalMatrix(nbRow, nbCol, expr, setup) {
        Matrix::buildCLSetup();
    }
    inline Matrix::Matrix(const cl::vector<cl::vector<float>> &initializer, std::shared_ptr<CLMatrixSetup> setup) : internalMatrix(initializer, setup) {
        Matrix::buildCLSetup();
    }
    inline Matrix::Matrix(int row, int col, cl::vector<float> *existingData, std::shared_ptr<CLMatrixSetup> setup)  : internalMatrix(row, col, existingData, setup) {
        Matrix::buildCLSetup();
    }
    inline Matrix &DNN::Matrix::operator=(const Matrix &toCopy) {
        BaseMatrix::copy(toCopy.internalMatrix, internalMatrix);
        return *this;
    }
    inline Matrix &DNN::Matrix::operator=(Matrix &&toMove) noexcept {
        BaseMatrix::move((BaseMatrix &&) toMove.internalMatrix, internalMatrix);
        return *this;
    }

    inline Matrix &Matrix::neg_IP() {
        _neg(*this, *this);
        return *this;
    }
    inline Matrix &Matrix::hadamardProd_IP(const Matrix &operand) {
        _hadamard(*this, operand, *this);
        return *this;
    }
    inline Matrix &Matrix::operator+=(const Matrix &operand) {
        _add(*this, operand, *this);
        return *this;
    }
    inline Matrix &Matrix::operator-=(const Matrix &operand) {
        _sub(*this, operand, *this);
        return *this;
    }
}