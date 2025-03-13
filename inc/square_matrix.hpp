#include "matrix.hpp"

namespace DNN {
    class SquareMatrix : public Matrix {
    public :
        //Host side creation
        SquareMatrix() = delete;
        SquareMatrix(int N, float expr = 0.0, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        SquareMatrix(const cl::vector<cl::vector<float>> &initializer, bool transposed, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());

        //Affectation creation (behaves smartly...)
        SquareMatrix(SquareMatrix  &toCopy);
        SquareMatrix(SquareMatrix &&toMove) noexcept;
        SquareMatrix(Matrix  &toCopy);
        SquareMatrix(Matrix &&toMove) noexcept;
        virtual ~SquareMatrix() = default;
        
        //Affectation operators
        inline SquareMatrix &operator=(SquareMatrix  &toCopy)           { return *this = (Matrix  &) toCopy; }
        inline SquareMatrix &operator=(SquareMatrix &&toMove) noexcept  { return *this = (Matrix &&) toMove; }
        inline SquareMatrix &operator=(Matrix        &toCopy)           { this->Matrix::operator=((Matrix  &) toCopy); return *this; }
        inline SquareMatrix &operator=(Matrix       &&toMove) noexcept  { this->Matrix::operator=((Matrix &&) toMove); return *this; }

        //Public operations' library
        SquareMatrix operator+(SquareMatrix &operand);
        SquareMatrix operator-(SquareMatrix &operand);
        SquareMatrix operator*(SquareMatrix &operand);

        SquareMatrix operator^(int exp);
        SquareMatrix operator-();

        SquareMatrix hadamardProduct(SquareMatrix &operand);
        SquareMatrix executeKernel(cl::KernelFunctor<cl::Buffer &, cl::Buffer &> kernel);

        static SquareMatrix identity(int N, std::shared_ptr<CLMatrixSetup> setup = CLMatrixSetup::getDefault());
        
    protected:
        SquareMatrix(int N, cl::Buffer *existingBuffer       , std::shared_ptr<CLMatrixSetup> setup);  //Internal device side creation
        SquareMatrix(int N, cl::vector<float> *existingVector, std::shared_ptr<CLMatrixSetup> setup);  //Internal host side creation (for derived classes)

        //Operations' library (to allow any derived type as return without copy)
        static void opPow(SquareMatrix &A, unsigned int exp, SquareMatrix &R);

        //Calculation management
        static constexpr uint8_t libCode = 1 << 0;
        static constexpr char libFile[] = "ocl/square_matrix.ocl";
        virtual void setCLSetup(std::shared_ptr<CLMatrixSetup> newSetup) override;
    };
}