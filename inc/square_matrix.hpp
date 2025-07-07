#include "matrix.hpp"

namespace DNN {
    class SquareMatrix : public Matrix {
    public :
        //Host side creation
        SquareMatrix() = delete;
        SquareMatrix(int N, float expr = 0.0, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        SquareMatrix(const cl::vector<cl::vector<float>> &initializer, bool transposed, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

        //Affectation creation (behaves smartly...)
        SquareMatrix(const SquareMatrix  &toCopy);
        SquareMatrix(SquareMatrix &&toMove) noexcept;
        SquareMatrix(const Matrix  &toCopy);
        SquareMatrix(Matrix &&toMove) noexcept;
        virtual ~SquareMatrix() = default;
        
        //Affectation operators
        inline SquareMatrix &operator=(const SquareMatrix  &toCopy)           { return *this = (Matrix  &) toCopy; }
        inline SquareMatrix &operator=(SquareMatrix       &&toMove) noexcept  { return *this = (Matrix &&) toMove; }
        inline SquareMatrix &operator=(const Matrix        &toCopy)           { this->Matrix::operator=((Matrix  &) toCopy); return *this; }
        inline SquareMatrix &operator=(Matrix             &&toMove) noexcept  { this->Matrix::operator=((Matrix &&) toMove); return *this; }

        //Public operations' library
        SquareMatrix operator+(const SquareMatrix &operand) const;
        SquareMatrix operator-(const SquareMatrix &operand) const;
        SquareMatrix operator*(const SquareMatrix &operand) const;

        SquareMatrix operator^(int exp) const;
        SquareMatrix operator-() const;

        SquareMatrix hadamardProduct(const SquareMatrix &operand) const;
        SquareMatrix executeKernel(cl::KernelFunctor<cl::Buffer &, cl::Buffer &> kernel) const;

        //Constants
        static SquareMatrix IDENTITY(int N, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        static SquareMatrix SCALAR(int N, float lambda, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
    protected:
        SquareMatrix(int N, cl::Buffer *existingBuffer       , std::shared_ptr<CLMatrixSetup> setup);  //Internal device side creation
        SquareMatrix(int N, cl::vector<float> *existingVector, std::shared_ptr<CLMatrixSetup> setup);  //Internal host side creation (for derived classes)

        //Operations' library (to allow any derived type as return without copy)
        static void opPow(const SquareMatrix &A, unsigned int exp, SquareMatrix &R);
        static void opInv(const SquareMatrix &A, SquareMatrix &R);

        //Calculation management
        static constexpr uint8_t libCode = 1 << 2;
        static constexpr char libFile[] = "ocl/square_matrix.ocl";
        virtual void setCLSetup(std::shared_ptr<CLMatrixSetup> newSetup) override;
    };
}