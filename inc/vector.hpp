#pragma once

#include "matrix.hpp"

namespace DNN {
    //This class does not garanty that transpose = false (it is an internal thing we do not care about)
    class Vector : public Matrix {
    public :
        //Host side creation
        Vector() = delete;
        Vector(int nbRow, float expr = 0.0);
        Vector(const cl::vector<float> &initialiser);
        Vector(cl::vector<float> &&initialiser);

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

        Matrix addOverMatrix(Matrix &operand); //TODO : Need specific kernel (and the corresponding transpose...)
        Matrix subOverMatrix(Matrix &operand); 

        //Data access
        inline float &operator[](cl::size_type row) { return getLValueElement(row); }
        inline float &getLValueElement(cl::size_type row) {return Matrix::getLValueElement(row, 0); }
        inline float  getRValueElement(cl::size_type row) {return Matrix::getRValueElement(row, 0); }

    protected:
        Vector(int nbRow, cl::Buffer *existingBuffer);  //Internal device side creation
        Vector(int nbRow, cl::vector<float> *existingVector);  //Internal host side creation (for derived classes)

        friend Vector operator*(Matrix &AL, Vector &X);
    };

    //Other DNN operators
    Vector operator*(Matrix &AL, Vector &X);
};
