#include "MLPApproxNet.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>

std::vector<TrainPair> GenerateTrainSet() {
    const double STEP_X = 0.005;
    const double INF_X = 0.0;
    const double SUP_X = 12.56;

    std::vector<TrainPair> trainSet;

    for (double x = INF_X; x <= SUP_X; x += STEP_X) {
        double y = x;
        double z = sin(y);
        std::cout << y << " " << z << " " << std::endl;
        trainSet.emplace_back(std::make_pair(std::vector<double>{y}, z));
    }

    std::random_shuffle(trainSet.begin(), trainSet.end());

    return trainSet;
}

void DumpPoints(const std::vector<TrainPair> &points, std::ostream &output) {
    for (size_t iPoint = 0; iPoint < points.size(); ++iPoint) {
        output << std::setprecision(15);
        output << points[iPoint].first[0] /*<< "\t" <<  points[iPoint].first[1]*/
            << "\t" << points[iPoint].second << std::endl;
    }
}

bool comparator(const TrainPair &t1, const TrainPair &t2) {
    return t1.first[0] > t2.first[0];
}

int main() {
    const std::string TRAIN_SET_FILE = "train.data";
    const std::string TEST_SET_FILE  = "test.data";

    MLPApproxNet::Config nnetConfig;
    nnetConfig.inputLayerNeurons = 6;
    nnetConfig.inputVecDim = 1;
    nnetConfig.epochs = 200;
    nnetConfig.sigma = 1.0;
    nnetConfig.trainCoeff = 0.063; // should be [0; 1]

    MLPApproxNet nnet;
    std::vector<TrainPair> pointSet(std::move(GenerateTrainSet()));
    const size_t half = pointSet.size() / 2;
    std::vector<TrainPair> trainSet(pointSet.begin(), pointSet.begin() + half);
    std::vector<TrainPair> testSet(pointSet.begin() + half, pointSet.end());

    std::ofstream trainSetDump(TRAIN_SET_FILE, std::ios::binary);
    std::ofstream testSetDump(TEST_SET_FILE, std::ios::binary);

    nnet.Init(nnetConfig);
    nnet.OnlineTrain(trainSet);

    std::sort(trainSet.begin(), trainSet.end());
    DumpPoints(trainSet, trainSetDump);

    double testVal;
    double maxTestError = 0.0;
    double testError;

    for (size_t iPoint = 0; iPoint < testSet.size(); ++iPoint) {
        testVal = nnet.Test(testSet[iPoint].first);
        testError = fabs(testVal - testSet[iPoint].second);

        if (testError > maxTestError)
            maxTestError = testError;
        
        testSet[iPoint].second = testVal; // for vizualization
    }

    std::cout << "Max approximation error: " << maxTestError << std::endl;
    std::sort(testSet.begin(), testSet.end(), comparator);
    DumpPoints(testSet, testSetDump);
    nnet.DumpSigmoids(testSet);

    return 0;
}
