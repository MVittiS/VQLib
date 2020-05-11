#pragma once

#include "VQDataTypes.h"
#include "VQArithmetic.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <type_traits>


template <typename T, size_t width, typename Index = uint32_t>
std::pair<FlexMatrix<T, width>, std::vector<Index>> VQGenerateDict(
    const FlexMatrix<T, width>& data,
    Index dictSize, 
    size_t iterationsMax = 300) {

    FlexMatrix<T, width> dict;
    dict.reserve(dictSize);

    std::vector<Index> indices(data.size());

    for (Index currentDictSize = 1; currentDictSize != dictSize; ++currentDictSize) {
        if (currentDictSize != 1) {
            indices = VQIndicesFromDict<T, width, Index>(data, dict);
        }
        else {
            std::fill(indices.begin(), indices.end(), T(0));
            dict = VQDictFromIndices<T, width, Index>(data, indices);
        }

        const auto mostFrequentIdx = ModeWithArray<Index>(indices, currentDictSize);

        // TODO: Use Covariance algorithm to split, instead of random
        const auto frand = [] {
            constexpr T invLimit = T(1) / T(RAND_MAX);
            return rand() * invLimit;
        };

        const auto randomVec = [&]{
            typename decltype(dict)::value_type randomVec;
            for (auto& elem : randomVec) {
                constexpr T small(1e-5);
                elem = (frand() - T(0.5)) * small;
            }
            return randomVec;
        }();

        dict.insert(dict.begin() + mostFrequentIdx, dict[mostFrequentIdx]);
        for (size_t x = 0; x != width; ++x) {
            dict[mostFrequentIdx    ][x] += randomVec[x];
            dict[mostFrequentIdx + 1][x] -= randomVec[x];
        }

        indices = VQIndicesFromDict<T, width, Index>(data, dict);

        bool indicesChanged = true;
        for (size_t iter = 0; iter != iterationsMax && indicesChanged; ++iter) {
            #ifdef VQLIB_VERBOSE_OUTPUT
                printf("Iteration %d.%d\n", int(currentDictSize), int(iter));
            #endif
            indicesChanged = false;

            dict = VQDictFromIndices<T, width, Index>(data, indices);
            const auto newIndices = VQIndicesFromDict<T, width, Index>(data, dict);

            if (indices != newIndices) {
                indicesChanged = true;
                indices = newIndices;
            }
        }
    }
    
    return std::make_pair(dict, indices);
}


template <typename T, size_t width, typename Index = uint32_t>
std::pair<FlexMatrix<T, width>, std::vector<Index>> VQGenerateDictFast(
    const FlexMatrix<T, width>& data,
    Index dictSize,
    size_t iterationsMax = 300) {

    FlexMatrix<T, width> dict;
    dict.reserve(dictSize);

    const auto randomIdx = randomSamples(data.size(), dictSize);

    for (auto idx : randomIdx) {
        dict.push_back(data[idx]);
    }

    std::vector<Index> indices(data.size());

    indices = VQIndicesFromDict<T, width, Index>(data, dict);

    bool indicesChanged = true;
    for (size_t iter = 0; iter != iterationsMax && indicesChanged; ++iter) {
        #ifdef VQLIB_VERBOSE_OUTPUT
            printf("Fast iteration %d\n", int(iter));
        #endif
        indicesChanged = false;

        dict = VQDictFromIndices<T, width, Index>(data, indices);
        const auto newIndices = VQIndicesFromDict<T, width, Index>(data, dict);

        if (indices != newIndices) {
            indicesChanged = true;
            indices = newIndices;
        }
    }

    return std::make_pair(dict, indices);
}