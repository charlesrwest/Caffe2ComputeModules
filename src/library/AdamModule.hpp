#pragma once
#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct AdamModuleParameters
{
std::string moduleName;
std::string blobToUpdateName;
std::vector<int64_t> blobToUpdateShape;
std::string gradientBlobName;
std::string learningRateBlobName; //Makes a default value one if not given
std::string iteratorBlobName; //Makes one if not given

double beta1 = .9;
double beta2 = .999;
double epsilon = 1e-5;
};

/**
This class is a more or less direct implementation of the "Adam" operator.  See the Adam operator documentation for parameter meanings and "ComputeModuleDefinition" for the meaning of the overriden functions.
*/
class AdamModule : public ComputeModuleDefinition
{
public:
AdamModule(const AdamModuleParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const override;

/**
This function returns the name of the moment1 blob which this operator will create upon initialization.
@return: The name of the created blob for moment1 (same shape as blobToUpdate)
*/
std::string GetMoment1BlobName() const;

/**
This function returns the name of the moment2 blob which this operator will create upon initialization.
@return: The name of the created blob for moment2 (same shape as blobToUpdate)
*/
std::string GetMoment2BlobName() const;

protected:
std::string blobToUpdateName;
std::vector<int64_t> blobToUpdateShape;
std::string gradientBlobName;
std::string learningRateBlobName; //Makes a default value one if not given
std::string iteratorBlobName; //Makes one if not given

double beta1;
double beta2;
double epsilon;
};


























} 
