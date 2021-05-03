// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/deps/json.hpp"
#include <fstream>

namespace mediapipe {

namespace {

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kFilenameTag[] = "FILENAME";

} // namespace
struct Box {
  int x;
  int y;
  int dx;
  int dy;
};

struct Instance {
  float score;
  std::string cat;
  Box box;
  std::vector<int> shape;

  Instance(float s, std::string c, Box b, std::vector<int> imageShape)
      : score(s), cat(c), box(b), shape(imageShape){};
};

// frame_id, a vector of instances
using FrameInstances = std::pair<int, std::vector<Instance>>;

std::vector<FrameInstances> loadFromJson(std::string filename) {
  std::ifstream in(filename);
  nlohmann::json j;
  in >> j;

  std::vector<FrameInstances> out;
  for (auto &element : j) {
    int frame_id = element["frame_id"];
    std::vector<int> imageShape{element["image_shape"][0],
                                element["image_shape"][1]};
    std::vector<Instance> instances;
    auto jsonInstances = element["instances"];
    for (auto &i : jsonInstances) {
      auto &jsonBox = i["x,y,dx,dy"];
      int x = jsonBox.at(0);
      int y = jsonBox.at(1);
      int dx = jsonBox.at(2);
      int dy = jsonBox.at(3);
      Box box{x, y, dx, dy};
      instances.push_back(Instance(i["score"], i["class"], box, imageShape));
    }
    out.emplace_back(frame_id, instances);
  }

  return out;
}

Detection createDetection(float score, std::string &cat, int x, int y, int dx,
                          int dy, int shape_x, int shape_y) {
  Detection detection;
  detection.add_score(score);
  detection.add_label(cat);

  LocationData *location_data = detection.mutable_location_data();
  LocationData::RelativeBoundingBox *bbox =
      location_data->mutable_relative_bounding_box();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

  float lx = static_cast<float>(shape_x), ly = static_cast<float>(shape_y);
  float xmin = static_cast<float>(x) / lx;
  float ymin = static_cast<float>(y) / ly;
  float delta_x = static_cast<float>(dx) / lx;
  float delta_y = static_cast<float>(dy) / ly;
  bbox->set_xmin(xmin);
  bbox->set_ymin(ymin);
  bbox->set_width(delta_x);
  bbox->set_height(delta_y);
  return detection;
}

// Load detections from a json file.
// The json file is expected to have the format
// [
//   {
//       "frame_id": 0,
//       "instances": [
//           {
//               "class": "chair",
//               "score": 0.9974355101585388,
//               "x,y,dx,dy": [
//                   760,
//                   357,
//                   135,
//                   220
//               ]
//           }, ...
//       ]
//   }
// ]
// Example config:
// node {
//   calculator: "DetectionsFromJsonCalculator"
//   input_stream: "DETECTIONS:detections"
//   output_stream: "DETECTIONS:output_detections"
// }
class DetectionsFromJsonCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract *cc) {
    RET_CHECK(cc->Inputs().HasTag(kImageFrameTag));
    RET_CHECK(cc->Outputs().HasTag(kDetectionsTag));
    RET_CHECK(cc->InputSidePackets().HasTag(kFilenameTag));
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
    cc->Outputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
    cc->InputSidePackets().Tag(kFilenameTag).Set<std::string>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext *cc) override {
    // Inform the framework that we always output at the same timestamp
    // as we receive a packet at.
    cc->SetOffset(mediapipe::TimestampDiff(0));
    std::string filename =
        cc->InputSidePackets().Tag(kFilenameTag).Get<std::string>();
    // "/home/chenyuf2/Downloads/challenge_data/iphone_test/2_fasterrcnn_d2.json";
    frameInstances_ = loadFromJson(filename);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext *cc) override;

private:
  int frame_id_ = -1;
  std::vector<FrameInstances> frameInstances_;
};

REGISTER_CALCULATOR(DetectionsFromJsonCalculator);

absl::Status DetectionsFromJsonCalculator::Process(CalculatorContext *cc) {
  std::cout << "outside frame_id_ " << frame_id_ << std::endl;
  if (cc->Inputs().HasTag(kImageFrameTag)) {
    frame_id_++;
    std::cout << "frame_id_ " << frame_id_ << std::endl;
    auto output_detections = absl::make_unique<std::vector<Detection>>();
    // if (frame_id_ > 5) {
    //   cc->Outputs()
    //       .Tag(kDetectionsTag)
    //       .Add(output_detections.release(), cc->InputTimestamp());
    //   return absl::OkStatus();
    // }
    // load_from_json

    // createDetection(float score, std::string &cat, int x, int y, int dx,
    // int dy)
    // std::string t = "chair";
    auto [_, instances] = frameInstances_[frame_id_];
    for (auto &i : instances) {
      // std::cout << i.score << " " << i.cat << " " << i.box.x << " " <<
      // i.box.y
      //           << " " << i.box.dx << " " << i.box.dy << std::endl;
      auto d = createDetection(i.score, i.cat, i.box.x, i.box.y, i.box.dx,
                               i.box.dy, i.shape[0], i.shape[1]);
      output_detections->emplace_back(d);
    }

    cc->Outputs()
        .Tag(kDetectionsTag)
        .Add(output_detections.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}

} // namespace mediapipe
