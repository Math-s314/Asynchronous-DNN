#include <iostream>
#include <iomanip>
#include "matrix.hpp"

#define LENGTH 20

int main(void) {
    DNN::Matrix A(9, 9, 10.f);
    DNN::Matrix B(9, 9, 10.f);
    A[0][0] = 1.f;
    A[1][1] = 1.f;
    A[2][2] = 1.f;
    A[3][3] = 1.f;
    A[4][4] = 1.f;
    A[5][5] = 1.f;
    A[6][6] = 1.f;
    A[7][7] = 1.f;
    A[8][8] = 1.f;

    B[0][0] = 1.f;
    B[1][1] = 1.f;
    B[2][2] = 1.f;
    B[3][3] = 1.f;
    B[4][4] = 1.f;
    B[5][5] = 1.f;
    B[6][6] = 1.f;
    B[7][7] = 1.f;
    B[8][8] = 1.f;
    std::cout << A;

    DNN::Matrix C(A+A);
    C.waitForResults();

    std::cout << C;
    std::cout << DNN::BufferLinkManager::DEBUG_created;

    return 0;
}