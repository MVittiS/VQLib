#pragma once

#include <array>
#include <vector>

template <typename T, size_t width>
using MatrixRow = std::array<T, width>;

template <typename T, size_t width, size_t height>
using FixMatrix = std::array<MatrixRow<T, width>, height>;

template <typename T, size_t width>
using FlexMatrix = std::vector<MatrixRow<T, width>>;

template <typename T, size_t width, size_t height>
FlexMatrix<T, width> FixToFlexMatrix (
    const FixMatrix<T, width, height>& fixMatrix) {
    
    FlexMatrix<T, width> flexResult(height);
    for (auto x = 0; x != height; ++x) {
        std::copy(fixMatrix[x].begin(), fixMatrix[x].end(), flexResult[x].begin());
    }
    return flexResult;
}
