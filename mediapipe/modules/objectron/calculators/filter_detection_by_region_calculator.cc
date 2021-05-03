// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/objectron/calculators/filter_detection_by_region_calculator.pb.h"

namespace mediapipe {

namespace {

constexpr char kDetectionsTag[] = "DETECTIONS";

using mediapipe::ContainsKey;
using Detections = std::vector<Detection>;
using Strings = std::vector<std::string>;

} // namespace

// Filters the entries in a Detection by location and size of the bounding box
//
// Example config:
// node {
//   calculator: "FilterDetectionByRegionCalculator"
//   input_stream: "DETECTIONS:detections"
//   output_stream: "DETECTIONS:filtered_detections"
//   options: {
//     [mediapipe.FilterDetectionByRegionCalculatorOptions.ext]: {
//       reject_border_width: 0.1
//     }
//   }
// }

bool isNearBorder(const Detection &detection, float rejectBorderWidth) {
  float xmin = detection.location_data().relative_bounding_box().xmin();
  float ymin = detection.location_data().relative_bounding_box().ymin();
  float dx = detection.location_data().relative_bounding_box().width();
  float dy = detection.location_data().relative_bounding_box().height();

  if (xmin < rejectBorderWidth || ymin < rejectBorderWidth ||
      (xmin + dx) > (1. - rejectBorderWidth) ||
      (ymin + dy) > (1. - rejectBorderWidth)) {
    return true;
  }
  return false;
}

class FilterDetectionByRegionCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status Process(CalculatorContext *cc) override;

private:
  // Stores filter thresholds
  FilterDetectionByRegionCalculatorOptions options_;
};
REGISTER_CALCULATOR(FilterDetectionByRegionCalculator);

absl::Status
FilterDetectionByRegionCalculator::GetContract(CalculatorContract *cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<Detections>();
    cc->Outputs().Tag(kDetectionsTag).Set<Detections>();
  }
  return absl::OkStatus();
}

absl::Status FilterDetectionByRegionCalculator::Open(CalculatorContext *cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<FilterDetectionByRegionCalculatorOptions>();
  return absl::OkStatus();
}

absl::Status FilterDetectionByRegionCalculator::Process(CalculatorContext *cc) {
  Detections detections;
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    detections = cc->Inputs().Tag(kDetectionsTag).Get<Detections>();
  }

  std::unique_ptr<Detections> outputs(new Detections);
  for (const auto &input : detections) {
    if (!isNearBorder(input, options_.reject_border_width())) {
      // std::cout << "in filter_detection_by_region_calculator "
      //           << input.detection_id() << std::endl;
      outputs->emplace_back(input);
    }
  }
  if (cc->Outputs().HasTag(kDetectionsTag) && !outputs->empty()) {
    cc->Outputs()
        .Tag(kDetectionsTag)
        .Add(outputs.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

} // namespace mediapipe
