#include "MLPApproxNet.h"
#include <iomanip>
#include <algorithm>
#include <cmath>

std::vector<TrainPair> GenerateTrainSet() {
    const double STEP_X = 0.002;//= 0.4; // 2500 points
    const double STEP_Y = 0.4;

    const double INF_X = 0.0;
    const double INF_Y = -10.0;

    const double SUP_X = 6.28;
    const double SUP_Y = 10.0;

    std::vector<TrainPair> trainSet;

    for (double x = INF_X; x <= SUP_X; x += STEP_X) {
        //for (double y = INF_Y; y <= SUP_Y; y += STEP_Y) {
            double z = sin(x);// * cos(y);
            trainSet.emplace_back(std::make_pair(std::vector<double>{x/*,y*/}, z));
        //}
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

int main() {
    const std::string TRAIN_SET_FILE = "train.data";
    const std::string SOURCE_SET_FILE  = "source.data";
    const std::string TEST_SET_FILE  = "test.data";

    MLPApproxNet nnet;
    MLPApproxNet::Config nnetConfig;

    nnetConfig.hiddenLayerNeurons = 32;
    nnetConfig.inVecDim = 1;
    nnetConfig.trainEpochs = 5000;
    nnetConfig.activationFunction.sigma = 1.0;
    nnetConfig.rprop.nuInit = 0.1;

    nnet.Init(nnetConfig);
    std::vector<TrainPair> pointSet(std::move(GenerateTrainSet()));
    const size_t half = pointSet.size() / 2;
    std::vector<TrainPair> trainSet(pointSet.begin(), pointSet.begin() + half);
    std::vector<TrainPair> testSet(pointSet.begin() + half, pointSet.end());

    std::ofstream trainSetDump(TRAIN_SET_FILE, std::ios::binary);
    std::ofstream srcSetDump(SOURCE_SET_FILE, std::ios::binary);
    std::ofstream testSetDump(TEST_SET_FILE, std::ios::binary);

    DumpPoints(trainSet, trainSetDump);
    DumpPoints(testSet, srcSetDump);
    nnet.Train(trainSet);

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
    DumpPoints(testSet, testSetDump);
}
