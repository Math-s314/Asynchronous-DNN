#include "matrix.hpp"

namespace DNN {
    class SquareMatrix : public Matrix {
    public :
        // Constructors

        SquareMatrix(std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault()) : Matrix(setup) {}
        SquareMatrix(int N, float expr = 0.0, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        SquareMatrix(const cl::vector<cl::vector<float>> &initializer, bool transposed, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

        SquareMatrix(const SquareMatrix  &toCopy) : Matrix(toCopy) {}
        SquareMatrix(SquareMatrix &&toMove) noexcept : Matrix((Matrix &&) toMove) {}
        virtual ~SquareMatrix() = default;

        SquareMatrix &operator=(const SquareMatrix  &toCopy)           { return *this = (Matrix  &) toCopy; }
        SquareMatrix &operator=(SquareMatrix       &&toMove) noexcept  { return *this = (Matrix &&) toMove; }
        SquareMatrix &operator=(const Matrix        &toCopy)           { this->Matrix::operator=((Matrix  &) toCopy); return *this; }
        SquareMatrix &operator=(Matrix             &&toMove) noexcept  { this->Matrix::operator=((Matrix &&) toMove); return *this; }

        /// Operations' library

        //In-place operations
        SquareMatrix &neg_IP() { return static_cast<SquareMatrix &>(Matrix::neg_IP()); }
        SquareMatrix &hadamardProd_IP(const SquareMatrix &operand) { return static_cast<SquareMatrix &>(Matrix::hadamardProd_IP(operand)); }
        SquareMatrix &operator+=(const SquareMatrix &operand) { return static_cast<SquareMatrix &>(Matrix::operator+=(operand)); }
        SquareMatrix &operator-=(const SquareMatrix &operand) { return static_cast<SquareMatrix &>(Matrix::operator-=(operand)); }
        SquareMatrix &pow_IP(int exp) { _pow(*this, exp, *this); return *this; }

        //Internal versions (avoiding copies)
        static void _pow(const SquareMatrix &A, unsigned int exp, SquareMatrix &R);
        static void _pow(const SquareMatrix &A, int exp, SquareMatrix &R);
        static void _inv(const SquareMatrix &A, SquareMatrix &R);

        //Constants
        static SquareMatrix IDENTITY(int N, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault()) { return SCALAR(N, 1.f, setup); }
        static SquareMatrix SCALAR(int N, float lambda, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

    protected:
        // Calculation management
        static constexpr uint8_t libCode = 1 << 2;
        static constexpr char libFile[] = "ocl/square_matrix.ocl";
        virtual void buildCLSetup() override;
    };

    /// Operators

    //Arithmetic symbols
    SquareMatrix operator+(const SquareMatrix &A, const SquareMatrix &B);
    SquareMatrix operator-(const SquareMatrix &A, const SquareMatrix &B);
    SquareMatrix operator*(const SquareMatrix &A, const SquareMatrix &B);
    SquareMatrix operator-(const SquareMatrix &A);

    SquareMatrix operator^(const SquareMatrix &A, int exp);

    //Functions
    SquareMatrix hadamardProduct(const SquareMatrix &A, const SquareMatrix &B);


    /// Inline Definitions

    inline SquareMatrix::SquareMatrix(int N, float expr, std::shared_ptr<CLMatrixSetup> setup) : Matrix(N, N, expr, setup) {
        SquareMatrix::buildCLSetup();
    }
    inline SquareMatrix::SquareMatrix(const cl::vector<cl::vector<float>> &initializer, bool transposed, std::shared_ptr<CLMatrixSetup> setup) : Matrix(initializer, setup){
        SquareMatrix::buildCLSetup();
    }

    inline SquareMatrix operator+(const DNN::SquareMatrix &A, const DNN::SquareMatrix &B) {
        return static_cast<SquareMatrix>((const Matrix &) A + (const Matrix &) B);
    }
    inline SquareMatrix operator-(const DNN::SquareMatrix &A, const DNN::SquareMatrix &B) {
        return static_cast<SquareMatrix>((const Matrix &) A - (const Matrix &) B);
    }
    inline SquareMatrix operator*(const DNN::SquareMatrix &A, const DNN::SquareMatrix &B) {
        return static_cast<SquareMatrix>((const Matrix &) A * (const Matrix &) B);
    }
    inline SquareMatrix operator-(const DNN::SquareMatrix &A) {
        return static_cast<SquareMatrix>(-(const Matrix &) A);
    }
    inline SquareMatrix hadamardProduct(const DNN::SquareMatrix &A, const DNN::SquareMatrix &B) {
        return static_cast<SquareMatrix>(hadamardProduct((const Matrix &) A, (const Matrix &) B));
    }
}