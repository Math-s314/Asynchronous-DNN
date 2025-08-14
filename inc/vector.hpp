#pragma once

#include "matrix.hpp"

namespace DNN {
    class Vector : public Matrix {
    public :
        //Constructors
        Vector(std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        Vector(int nbRow, float expr = 0.0, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        Vector(const cl::vector<float> &initializer, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        Vector(cl::vector<float> &&initializer, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

        Vector(const Vector  &toCopy) : Matrix(toCopy) {}
        Vector(Vector &&toMove) noexcept : Matrix((Matrix &&) toMove) {}
        ~Vector() override = default;

        Vector &operator=(const Vector  &toCopy)           { return *this = (Matrix  &) toCopy; }
        Vector &operator=(Vector       &&toMove) noexcept  { return *this = (Matrix &&) toMove; }
        Vector &operator=(const Matrix  &toCopy)           { this->Matrix::operator=((Matrix  &) toCopy); return *this; }
        Vector &operator=(Matrix       &&toMove) noexcept  { this->Matrix::operator=((Matrix &&) toMove); return *this; }

        // Data Access
        float &operator[](cl::size_type row) { return getLValueElement(row); }
        float &getLValueElement(cl::size_type row)       {return Matrix::getLValueElement(row, 0); }
        float  getRValueElement(cl::size_type row) const {return Matrix::getRValueElement(row, 0); }

        /// Operations' library

        //In-place operations
        Vector &neg_IP() { return static_cast<Vector &>(Matrix::neg_IP()); }
        Vector &hadamardProd_IP(const Vector &operand) { return static_cast<Vector &>(Matrix::hadamardProd_IP(operand)); }
        Vector &operator+=(const Vector &operand) { return static_cast<Vector &>(Matrix::operator+=(operand)); }
        Vector &operator-=(const Vector &operand) { return static_cast<Vector &>(Matrix::operator-=(operand)); }

        //Internal versions (avoiding copies)
        static void _addOver(const Vector &A, const Matrix &B, Matrix &R);
        static void _subOver(const Vector &A, const Matrix &B, Matrix &R);

    protected:
        //Calculation management
        static constexpr uint8_t libCode = 1 << 3;
        static constexpr char libFile[] = "ocl/vector.ocl";
        virtual void buildCLSetup() override;
    };

    /// Operators

    // Arithmetic symbols
    Vector operator+(const Vector &A, const Vector &B);
    Vector operator-(const Vector &A, const Vector &B);
    Vector operator*(const Vector &A, const Vector &B);
    Vector operator*(const Matrix &A, const Vector &X);
    Vector operator-(const Vector &A);

    // Functions
    Vector hadamardProduct(const Vector &A, const Vector &B);


    /// Inline Definitions

    inline Vector::Vector(std::shared_ptr<CLMatrixSetup> setup) : Matrix(setup) {
        Vector::buildCLSetup();
    }
    inline Vector::Vector(int nbRow, float expr, std::shared_ptr<CLMatrixSetup> setup) : Matrix(nbRow, 1, expr, setup) {
        Vector::buildCLSetup();
    }
    inline Vector::Vector(const cl::vector<float> &initializer, std::shared_ptr<CLMatrixSetup> setup) : Matrix(initializer.size(), 1, new cl::vector<float>(initializer), setup) {
        Vector::buildCLSetup();
    }
    inline Vector::Vector(cl::vector<float> &&initializer, std::shared_ptr<CLMatrixSetup> setup) : Matrix(initializer.size(), 1, new cl::vector<float>((cl::vector<float> &&) initializer), setup) {
        Vector::buildCLSetup();
    }

    inline Vector operator+(const Vector &A, const Vector &B) {
        return static_cast<Vector>((Matrix &) A + (Matrix &) B);
    }
    inline Vector operator-(const Vector &A, const Vector &B) {
        return static_cast<Vector>((Matrix &) A - (Matrix &) B);
    }
    inline Vector operator*(const Vector &A, const Vector &B) {
        return static_cast<Vector>((Matrix &) A * (Matrix &) B);
    }
    inline Vector operator*(const Matrix &A, const Vector &X) {
        return static_cast<Vector>(A * (Matrix &) X);
    }
    inline Vector operator-(const Vector &A) {
        return static_cast<Vector>(-(Matrix &) A);
    }
}
