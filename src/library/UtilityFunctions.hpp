#pragma once

#include<string>
#include<vector>
#include "caffe2/proto/caffe2.pb.h"

namespace GoodBot
{

/**
This function generates a (capitalized) hex string using rand() to select digits.
@inputLength:How many digits the string should have
@return: The generated string
*/
std::string GenerateRandomHexString(int64_t inputLength);

std::string MakeGradientOperatorBlobName(const std::string& inputOperatorBlobName);


std::vector<caffe2::OperatorDef> GetGradientOperatorsFromOperator(const caffe2::OperatorDef& inputOperator);













}
