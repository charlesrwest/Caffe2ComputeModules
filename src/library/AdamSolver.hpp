#pragma once
#include "CompositeComputeModuleDefinition.hpp"

namespace GoodBot
{

struct AdamSolverParameters
{
std::string moduleName;
std::string learningRateBlobName = ""; //Makes a default value one if not given
std::string iteratorBlobName = ""; //Makes one if not given
std::vector<std::string> trainableParameterNames;
std::vector<std::vector<int64_t>> trainableParameterShapes;

double beta1 = .9;
double beta2 = .999;
double epsilon = 1e-5;
double learningRate = -.001; //Used if no learning rate blob is given
};

/**
This class is a compound compute module which makes a AdamModule for every given trainable blob.  This can be used to add adam solver training phase to pretty much any composite compute module.  This class also make a blob with a constant learning rate if a learning rate blob is not given and makes iterator blobs to keep track of the number of iterations for training. TODO: Make default iterator only be incremented in "TRAIN" mode. 
*/
class AdamSolver : public CompositeComputeModuleDefinition
{
public:
AdamSolver(const AdamSolverParameters& inputParameters);

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

/**
This function returns the name of blob it is using for the base learning rate (created if none provided).
@return: The name of the created blob for the learning rate (single scalar)
*/
std::string GetLearningRateBlobName() const;

/**
This function returns the name of blob it is using for the iteration count (created if none provided).
@return: The name of the created blob for the iteration count (single integer)
*/
std::string GetIteratorBlobName() const;

protected:
bool createLearningRateBlob;
double learningRate;
std::string learningRateBlobName;
std::string iteratorBlobName;
};
























}
