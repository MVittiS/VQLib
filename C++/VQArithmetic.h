#pragma once

#include "VQDataTypes.h"

#include <algorithm>
#include <random>

#include <cassert>
#include <cmath>

const auto randomSamples = [](size_t numElements, size_t howMany) {
        std::vector<size_t> randomSamples(numElements);
        std::iota(randomSamples.begin(), randomSamples.end(), 0);

        std::random_device device;
        std::mt19937 generator(device());
        std::shuffle(randomSamples.begin(), randomSamples.end(), generator);
        randomSamples.erase(randomSamples.begin() + howMany, randomSamples.end());

        return randomSamples;
};


template <typename T, size_t width, typename Index = uint32_t>
std::vector<Index> VQIndicesFromDict(
    const FlexMatrix<T, width>& data,
          FlexMatrix<T, width>  dict) {

    const auto dataSize = data.size();
    std::vector<Index> minIndices(dataSize);

    std::vector<size_t> histogram(dict.size());
    std::fill(histogram.begin(), histogram.end(), 0);

    for (auto dataIdx = 0; dataIdx != dataSize; ++dataIdx) {
        // The dictionary will be smaller in the majority of cases,
        //  so it makes sense to iterate over the thing that is
        //  likelier to still be in cache rather than the whole
        //  dataset, as opposed to the MatLAB code. 
        const auto dataEntry = data[dataIdx];
        const auto currentDictSize = dict.size();
        std::vector<float> distances(currentDictSize);

        for (auto dictIdx = 0; dictIdx != currentDictSize; ++dictIdx) {
            T rms(0);

            for (auto elem = 0; elem != width; ++elem) {
                const auto diff = dict[dictIdx][elem] - dataEntry[elem];
                rms += diff * diff;
                // if constexpr (std::is_floating_point_v<T>) {
                //     rms += std::fabs(diff);
                // } else {
                //     rms += std::abs(diff);
                // }
            }
            distances[dictIdx] = rms;
        }

        const auto minPos = std::min_element(distances.begin(), distances.end());
        const auto minIdx = (Index)std::distance(distances.begin(), minPos);
        minIndices[dataIdx] = minIdx;
        ++histogram[minIdx];
    }

    const bool allIndicesUsed = std::all_of(histogram.begin(), histogram.end(),
                                    [](auto x) { return x != 0; });
    if (!allIndicesUsed) {
        const auto missingEntries = 
            std::accumulate(histogram.begin(), histogram.end(), 0,
                            [](auto acc, auto h) { return acc + (h == 0); });
        const auto randomIdxs = randomSamples(data.size(), missingEntries);

        auto remainingIdx = 0;
        for (auto col = 0; col != dict.size(); ++col) {
            if (histogram[col] == 0) {
                // Force at least one sample to become a new dict element,
                //  which will be picked up during the next iteration
                minIndices[randomIdxs[remainingIdx++]] = col;
            }
        }
    }

    return minIndices;
}

#include <type_traits>

template <typename T, size_t width, typename Index = uint32_t>
FlexMatrix<T, width> VQDictFromIndices(
    const FlexMatrix<T, width>& data,
    const std::vector<Index>& indices,
    const Index maxElement) {

    assert(data.size() == indices.size());
    const auto dataSize = data.size();

    const auto histogram = [&] {
        std::vector<size_t> histogram(maxElement + 1);
        std::fill(histogram.begin(), histogram.end(), 0);
        for (auto elem : indices) {
            ++histogram[elem];
        }
        return histogram;
    }();

    const auto newDictSize = maxElement + 1;
    FlexMatrix<T, width> newDict(newDictSize);
    for (auto& row : newDict) {
        std::fill(row.begin(), row.end(), T(0));
    }

    for (auto index = 0; index != dataSize; ++index) {
        for (auto elem = 0; elem != width; ++elem) {
            newDict[indices[index]][elem] += data[index][elem];
        }
    }

    if constexpr (std::is_floating_point_v<T>) {
        const auto invHistogram = [&] {
            std::vector<float> invHistogram;
            invHistogram.reserve(histogram.size());
            for (auto elem : histogram) {
                invHistogram.push_back(T(1) / T(elem));
            }
            return invHistogram;
        }();

        for (auto row = 0; row != newDictSize; ++row) {
            for (auto elem = 0; elem != width; ++elem) {
                newDict[row][elem] *= invHistogram[row];
            }
        }
    }
    else {
        for (auto row = 0; row != newDictSize; ++row) {
            for (auto elem = 0; elem != width; ++elem) {
                newDict[row][elem] /= histogram[row];
            }
        }
    }

    return newDict;
}

template <typename T, size_t width, typename Index = uint32_t>
FlexMatrix<T, width> VQDictFromIndices(
    const FlexMatrix<T, width>& data,
    const std::vector<Index>& indices) {

    Index maxElement = *std::max_element(indices.begin(), indices.end());
    return VQDictFromIndices<T, width, Index>(data, indices, maxElement);
}

template<typename Index = uint32_t>
Index ModeWithArray(const std::vector<Index>& input, Index limit) {
    std::vector<Index> histogram(limit + 1);
    std::fill(histogram.begin(), histogram.end(), 0);

    for (const auto element : input) {
        ++histogram[element];
    }

    const auto maxPos = std::max_element(histogram.begin(), histogram.end());
    return (Index)std::distance(histogram.begin(), maxPos);
}

#include <unordered_map>

template<typename T, typename Index = uint32_t>
T ModeWithMap(const std::vector<T>& input) {
    std::unordered_map<T, Index> histogram;

    for (const auto element : input) {
        ++histogram[element];
    }

    return std::max_element(histogram.begin(), histogram.end(),
        [](const auto key1, const auto key2) -> bool {
        return key1.second() < key2.second();
    })->first;
}

template<typename T, size_t width, typename Index = uint32_t>
std::array<T, width> LargestCovariance(
    const FlexMatrix<T, width>& data,
    const std::vector<Index>& indices,
    const std::array<T, width> dictEntry,
    const Index selectedIndex) {

    const auto dataMinusDict = [&] {
        FlexMatrix<T, width> dataMinusDict;
        for (auto row = 0; row != data.size(); ++row) {
            if (indices[row] == selectedIndex) {
                auto rowMinusDict = data[row];
                for (auto elem = 0; elem != width; ++elem) {
                    rowMinusDict[elem] -= dictEntry[elem];
                }
                dataMinusDict.push_back(rowMinusDict);
            }
        }
        return dataMinusDict;
    }();

    // TO BENCHMARK: is it worth transposing this matrix
    //  before the next step, to avoid the strided memory
    //  access?

    const auto covMatrix = [&] {
        const T scaling = T(1) / T(dataMinusDict.size() - 1);
        std::array<std::array<T, width>, width> covMatrix;

        for (auto y = 0; y != width; ++y) {
            for (auto x = 0; x != width; ++x) {
                if (x < y) {
                    covMatrix[y][x] = covMatrix[x][y];
                }
                else {
                    T accumulator(0);
                    for (auto elem = 0; elem != dataMinusDict.size(); ++elem) {
                        accumulator += dataMinusDict[elem][x] * dataMinusDict[elem][y];
                    }
                    covMatrix[y][x] = accumulator;
                }
            }
        }
        return covMatrix;
    }();


}
