#pragma once

#include "matrix.hpp"

namespace DNN {
    //This class does not guaranty that transpose = false (it is an internal thing we do not care about)
    class Vector : public Matrix {
    public :
        //Host side creation
        Vector() = delete;
        explicit Vector(int nbRow, float expr = 0.0, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        Vector(const cl::vector<float> &initializer, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        Vector(cl::vector<float> &&initializer, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

        //Affectation creation (behaves smartly...)
        Vector(Vector  &toCopy);
        Vector(Vector &&toMove) noexcept;
        Vector(Matrix  &toCopy);
        Vector(Matrix &&toMove) noexcept;
        virtual ~Vector() = default;
        
        //Affectation operators
        inline Vector &operator=(Vector  &toCopy)           { return *this = (Matrix  &) toCopy; }
        inline Vector &operator=(Vector &&toMove) noexcept  { return *this = (Matrix &&) toMove; }
        inline Vector &operator=(Matrix  &toCopy)           { this->Matrix::operator=((Matrix  &) toCopy); return *this; }
        inline Vector &operator=(Matrix &&toMove) noexcept  { this->Matrix::operator=((Matrix &&) toMove); return *this; }

        //Public operations' library
        Vector operator+(Vector &operand);
        Vector operator-(Vector &operand);
        Vector operator-();

        Vector hadamardProduct(Vector &operand);
        Vector executeKernel(cl::KernelFunctor<cl::Buffer &, cl::Buffer &> kernel);

        Matrix addOverMatrix(Matrix &operand);
        Matrix subOverMatrix(Matrix &operand); 

        //Data access
        inline float &operator[](cl::size_type row) { return getLValueElement(row); }
        inline float &getLValueElement(cl::size_type row) {return Matrix::getLValueElement(row, 0); }
        inline float  getRValueElement(cl::size_type row) {return Matrix::getRValueElement(row, 0); }

    protected:
        Vector(int nbRow, cl::Buffer *existingBuffer       , std::shared_ptr<CLMatrixSetup> setup);  //Internal device side creation
        Vector(int nbRow, cl::vector<float> *existingVector, std::shared_ptr<CLMatrixSetup> setup);  //Internal host side creation (for derived classes)

        //Calculation management
        static constexpr uint8_t libCode = 1 << 1;
        static constexpr char libFile[] = "ocl/vector.ocl";
        virtual void setCLSetup(std::shared_ptr<CLMatrixSetup> newSetup) override;

        //Operations' library (to allow any derived type as return without copy)
        static void opAOM(Vector &A, Matrix &B, Matrix &R);
        static void opSOM(Vector &A, Matrix &B, Matrix &R);

        friend Vector operator*(Matrix &AL, Vector &X);
    };

    //Other DNN operators
    Vector operator*(Matrix &AL, Vector &X);
}
