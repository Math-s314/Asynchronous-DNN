#pragma once

#include "matrix.hpp"

namespace DNN {
    class Scalar : public Matrix {
    public :
        //Constructors
        Scalar(std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        Scalar(float expr, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

        Scalar(const Scalar  &toCopy) : Matrix(toCopy) {}
        Scalar(Scalar &&toMove) noexcept : Matrix((Matrix &&) toMove) {}
        virtual ~Scalar() = default;

        Scalar &operator=(const Scalar  &toCopy)           { return *this = (Matrix  &) toCopy; }
        Scalar &operator=(Scalar       &&toMove) noexcept  { return *this = (Matrix &&) toMove; }
        Scalar &operator=(const Matrix  &toCopy)           { this->Matrix::operator=((Matrix  &) toCopy); return *this; }
        Scalar &operator=(Matrix       &&toMove) noexcept  { this->Matrix::operator=((Matrix &&) toMove); return *this; }

        // Data Access
        float &operator[](cl::size_type row) = delete;
        float &getLValueElement()       {return Matrix::getLValueElement(0, 0); }
        float  getRValueElement() const {return Matrix::getRValueElement(0, 0); }

        /// Operations' library

        //In-place operations
        Scalar &inv_IP();
        Scalar &neg_IP() { return static_cast<Scalar &>(Matrix::neg_IP()); }
        Scalar &hadamardProd_IP(const Scalar &operand) { return static_cast<Scalar &>(Matrix::hadamardProd_IP(operand)); }
        Scalar &operator+=(const Scalar &operand) { return static_cast<Scalar &>(Matrix::operator+=(operand)); }
        Scalar &operator-=(const Scalar &operand) { return static_cast<Scalar &>(Matrix::operator-=(operand)); }
        Scalar &operator*=(const Scalar &operand);
        Scalar &operator/=(const Scalar &operand);

        //Internal versions (avoiding copies)
        static void _mul(const Scalar &A, const Scalar &B, Scalar &R) { _hadamard(A, B, R); }
        static void _div(const Scalar &A, const Scalar &B, Scalar &R);
        static void _inv(const Scalar &A, Scalar &R);
        static void _addOver(const Scalar &A, const Matrix &B, Matrix &R);
        static void _subOver(const Scalar &A, const Matrix &B, Matrix &R);
        static void _mulOver(const Scalar &A, const Matrix &B, Matrix &R);
        static void _divOver(const Scalar &A, const Matrix &B, Matrix &R);

    protected:
        //Calculation management
        static constexpr uint8_t libCode = 1 << 1;
        static constexpr char libFile[] = "ocl/scalar.ocl";
        virtual void buildCLSetup() override;
    };
    

    /// Operators

    // Arithmetic symbols
    Scalar operator+(const Scalar &A, const Scalar &B);
    Matrix operator+(const Matrix &A, const Scalar &B);
    Matrix operator+(const Scalar &A, const Matrix &B);

    Scalar operator-(const Scalar &A, const Scalar &B);
    Matrix operator-(const Matrix &A, const Scalar &B);

    Scalar operator*(const Scalar &A, const Scalar &B);
    Matrix operator*(const Matrix &A, const Scalar &X);
    Matrix operator*(const Scalar &A, const Matrix &X);

    Scalar operator/(const Scalar &A, const Scalar &B);
    Matrix operator/(const Matrix &A, const Scalar &X);

    Scalar operator-(const Scalar &A);

    // Functions
    Scalar hadamardProduct(const Scalar &A, const Scalar &B);
    Scalar inv(const Scalar &A);

    /// Inline Definitions

    inline Scalar::Scalar(std::shared_ptr<CLMatrixSetup> setup) : Matrix(setup) {
        Scalar::buildCLSetup();
    }
    inline Scalar::Scalar(float expr, std::shared_ptr<CLMatrixSetup> setup) : Matrix(1, 1, expr, setup) {
        Scalar::buildCLSetup();
    }

    inline Scalar &Scalar::inv_IP() {
        _inv(*this, *this);
        return *this;
    }
    inline Scalar &Scalar::operator*=(const Scalar &operand) {
        _mul(*this, operand, *this);
        return *this;
    }
    inline Scalar &Scalar::operator/=(const Scalar &operand) {
        _div(*this, operand, *this);
        return *this;
    }

    inline Scalar operator+(const Scalar &A, const Scalar &B) {
        return static_cast<Scalar>((Matrix &) A + (Matrix &) B);
    }
    inline Scalar operator-(const Scalar &A, const Scalar &B) {
        return static_cast<Scalar>((Matrix &) A - (Matrix &) B);
    }
    inline Scalar operator*(const Scalar &A, const Scalar &X) {
        return static_cast<Scalar>(hadamardProduct(A, X));
    }
    inline Scalar operator-(const Scalar &A) {
        return static_cast<Scalar>(-(Matrix &) A);
    }
    inline Matrix operator+(const Scalar &A, const Matrix &B) {
        return B + A;
    }
    inline Matrix operator*(const Scalar &A, const Matrix &B) {
        return B * A;
    }

    inline Scalar hadamardProduct(const Scalar &A, const Scalar &B) {
        return static_cast<Scalar>(hadamardProduct((const Matrix &) A, (const Matrix &) B));
    }
}
