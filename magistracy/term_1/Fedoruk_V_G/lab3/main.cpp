#include "WTALayer.h"
#include <algorithm>
#include <cmath>

void GenerateElipsis2DPointSet(
        std::vector<std::vector<double>> *points,
        double a, double b,
        double xOffset, double yOffset,
        double rotation, double step) {
    a = (a < 0) ? -a : a;
    b = (b < 0) ? -b : b;
    double norm;

    for (double x = -a; x <= a; x += step) {
        for (double y = -b; y <= b; y += step) {
            if (((x * x) / (a * a) + (y * y) / (b * b)) <= 1) {
                std::vector<double> v(2);
                v[0] = (x * cos(rotation) + y * sin(rotation)) + xOffset;
                v[1] = (-x * sin(rotation) + y * cos(rotation)) + yOffset;
                // normalize radius-vectors to specified points
                norm = sqrt(v[0] * v[0] + v[1] * v[1]);
                v[0] /= norm;
                v[1] /= norm;
                points->emplace_back(std::move(v));
            }
        }
    }
}

std::vector<std::vector<double>> Generate2DTrainSet() {
    std::vector<std::vector<double>> trainSet;

    GenerateElipsis2DPointSet(&trainSet, 0.3, 0.2, 0.4, 0.5, -M_PI_4, 0.042);
    GenerateElipsis2DPointSet(&trainSet, 0.2, 0.1, -0.1, 0.5, M_PI_4, 0.022);

    std::random_shuffle(trainSet.begin(), trainSet.end());
    return trainSet;
}

void DumpPoints(std::ostream &output, const std::vector<std::vector<double>> &points) {
    for (size_t iPoint = 0; iPoint < points.size(); ++iPoint) {
        for (size_t iCoord = 0; iCoord < points[iPoint].size(); ++iCoord)
            output << points[iPoint][iCoord] << "\t";

        output << std::endl;
    }
}

int main() {
    const std::string WEIGHTS_DUMP_FILE = "weights.data";
    const std::string TRAIN_SET_FILE    = "points.data";
    
    WTALayer nnet;
    WTALayer::Config nnetConfig;

    nnetConfig.neurons  = 4;
    nnetConfig.inVecDim = 2;
    nnetConfig.trainEpochs = 2;
    nnetConfig.trainCoeff  = 0.1;
    nnetConfig.trainPenalty = 0.01;
    const std::vector<double> INITIAL_WEIGHTS(nnetConfig.neurons * nnetConfig.inVecDim, -1.0);

    nnet.Init(nnetConfig, false);

    if (!nnet.SetWeights(INITIAL_WEIGHTS))
        return 1;

    nnet.DumpWeights(std::cout, 6, "Initial weights:");

    std::ofstream trainSetDump(TRAIN_SET_FILE, std::ios::binary);
    std::ofstream weightsDump(WEIGHTS_DUMP_FILE, std::ios::binary);
    std::vector<std::vector<double>> trainSet(std::move(Generate2DTrainSet()));
    DumpPoints(trainSetDump, trainSet);
    nnet.Train(trainSet, weightsDump);
    nnet.DumpWeights(std::cout, 6, "Result weights:");
}
