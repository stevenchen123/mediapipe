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
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/util/deps/json.hpp"
#include <fstream>
#include <iomanip>

namespace mediapipe {

namespace {

constexpr char kDetectionsTrackedTag[] = "TRACKEDDETECTIONS";
constexpr char kDetectionsFbfTag[] = "FBFDETECTIONS";
constexpr char kFrameAnnotationTag[] = "ANNOTATIONS";
constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kFilenameTag[] = "FILENAME";

using Detections = std::vector<Detection>;

} // namespace

nlohmann::json detectionToJson(const Detection &detection, bool isDetected) {
  nlohmann::json j;
  j["cat"] = detection.label(0);
  j["score"] = detection.score(0);
  j["track_id"] = detection.detection_id();
  j["xmin"] = detection.location_data().relative_bounding_box().xmin();
  j["ymin"] = detection.location_data().relative_bounding_box().ymin();
  j["width"] = detection.location_data().relative_bounding_box().width();
  j["height"] = detection.location_data().relative_bounding_box().height();
  j["isDetected"] = isDetected;
  return j;
}

// test whether detection is produced by a detector or propogated by a tracker
//     context: detection could be either produced by a fbf detection model
//              or propogated with a tracking module
bool ifDetected(const Detection &detection, const Detections fbfDetections) {
  float xmin = detection.location_data().relative_bounding_box().xmin();
  float ymin = detection.location_data().relative_bounding_box().ymin();
  float width = detection.location_data().relative_bounding_box().width();
  float height = detection.location_data().relative_bounding_box().height();

  for (const auto &d : fbfDetections) {
    if (abs(xmin - d.location_data().relative_bounding_box().xmin()) < 1e-3 &&
        abs(ymin - d.location_data().relative_bounding_box().ymin()) < 1e-3 &&
        abs(width - d.location_data().relative_bounding_box().width()) < 1e-3 &&
        abs(height - d.location_data().relative_bounding_box().height()) < 1e-3)
      return true;
  }
  return false;
}

nlohmann::json detectionsToJson(const Detections &trackedDetections,
                                const Detections &fbfDetections) {
  nlohmann::json j = nlohmann::json::array();
  for (const auto &trackedDetection : trackedDetections) {
    bool isDetected = ifDetected(trackedDetection, fbfDetections);
    j.push_back(detectionToJson(trackedDetection, isDetected));
  }
  return j;
}

nlohmann::json point3dToJson(const Point3D &p) {
  nlohmann::json j = nlohmann::json::array();
  j.push_back(p.x());
  j.push_back(p.y());
  j.push_back(p.z());
  return j;
}

nlohmann::json normalizedPoint2dToJson(const NormalizedPoint2D &p) {
  nlohmann::json j = nlohmann::json::array();
  j.push_back(p.x());
  j.push_back(p.y());
  j.push_back(p.depth());
  return j;
}

nlohmann::json annotationToJson(const ObjectAnnotation &anno) {
  nlohmann::json j;
  j["track_id"] = anno.object_id();
  j["rotation_3x3_row_major"] = {
      anno.rotation(0), anno.rotation(1), anno.rotation(2),
      anno.rotation(3), anno.rotation(4), anno.rotation(5),
      anno.rotation(6), anno.rotation(7), anno.rotation(8)};

  j["translation"] = {anno.translation(0), anno.translation(1),
                      anno.translation(2)};
  j["scale"] = {anno.scale(0), anno.scale(1), anno.scale(2)};
  j["9_3d_keypoints_xyz"] = nlohmann::json::array();
  j["9_normalized_keypoints_xyd"] = nlohmann::json::array();
  for (const auto &k : anno.keypoints()) {
    j["9_3d_keypoints_xyz"].push_back(point3dToJson(k.point_3d()));
    j["9_normalized_keypoints_xyd"].push_back(
        normalizedPoint2dToJson(k.point_2d()));
  }
  // std::cout << "anno anno.object_id()" << anno.object_id() << std::endl;
  // std::cout << "anno anno.visibility()" << anno.visibility() << std::endl;
  // std::cout << "anno anno.scale()" << anno.scale(0) << anno.scale(1)
  //           << anno.scale(2) << std::endl;
  // std::cout << "anno anno.translation()" << anno.translation(0)
  //           << anno.translation(1) << anno.translation(2) << std::endl;
  // std::cout << "anno anno.rotation()" << anno.rotation(0) << anno.rotation(1)
  //           << anno.rotation(2) << std::endl;

  return j;
}

nlohmann::json frameAnnotationsToJson(const FrameAnnotation &fAnno) {
  nlohmann::json jFrame;
  jFrame["timestamp"] = fAnno.timestamp();
  nlohmann::json j = nlohmann::json::array();
  for (const auto &anno : fAnno.annotations()) {
    j.push_back(annotationToJson(anno));
  }
  jFrame["detections"] = j;
  return jFrame;
}

// Keep track of detections and frame_annotations for the entire
// video then write the results to a json file
//
// The json file is expected to have the format
// [
//   {
//       "frame_id": 0,
//       "detections2d": [
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
//       "detections2d": [
//
//       ]
//   }
// ]
// Example config:
// node {
//   calculator: "DetectionsFromJsonCalculator"
//   input_stream: "DETECTIONS:detections"
//   input_stream: "IMAGES:images"
//   input_stream: "ANNOTATIONS:annotations"
// }
class WriteToJsonFileCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status Process(CalculatorContext *cc) override;
  absl::Status Close(CalculatorContext *cc) override;

private:
  int frame_id_ = -1;
  std::vector<Detections> tracked_detections_;
  std::vector<Detections> fbf_detections_;
  std::vector<FrameAnnotation> frameAnnotations_;
  std::string outJsonFilename_;
};

REGISTER_CALCULATOR(WriteToJsonFileCalculator);

absl::Status WriteToJsonFileCalculator::GetContract(CalculatorContract *cc) {
  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag));
  RET_CHECK(cc->Inputs().HasTag(kFrameAnnotationTag));
  RET_CHECK(cc->Inputs().HasTag(kDetectionsTrackedTag));
  RET_CHECK(cc->Inputs().HasTag(kDetectionsFbfTag));
  RET_CHECK(cc->InputSidePackets().HasTag(kFilenameTag));
  cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
  cc->Inputs().Tag(kFrameAnnotationTag).Set<FrameAnnotation>();
  cc->Inputs().Tag(kDetectionsTrackedTag).Set<Detections>();
  cc->Inputs().Tag(kDetectionsFbfTag).Set<Detections>();
  cc->InputSidePackets().Tag(kFilenameTag).Set<std::string>();
  return absl::OkStatus();
}

absl::Status WriteToJsonFileCalculator::Open(CalculatorContext *cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(mediapipe::TimestampDiff(0));
  outJsonFilename_ =
      cc->InputSidePackets().Tag(kFilenameTag).Get<std::string>();
  // "/home/chenyuf2/Downloads/challenge_data/iphone_test/2_fasterrcnn_d2.json";
  return absl::OkStatus();
}

absl::Status WriteToJsonFileCalculator::Process(CalculatorContext *cc) {
  std::cout << "in write_to_json_file_calculator; frame_id_" << frame_id_
            << std::endl;
  if (cc->Inputs().HasTag(kImageFrameTag)) {
    frame_id_++;
    // std::cout << "    frame_id_ " << frame_id_ << std::endl;

    Detections tracked_detections;
    if (cc->Inputs().HasTag(kDetectionsTrackedTag) &&
        !cc->Inputs().Tag(kDetectionsTrackedTag).IsEmpty()) {
      tracked_detections =
          cc->Inputs().Tag(kDetectionsTrackedTag).Get<Detections>();
    }
    Detections fbf_detections;
    if (cc->Inputs().HasTag(kDetectionsFbfTag) &&
        !cc->Inputs().Tag(kDetectionsFbfTag).IsEmpty()) {
      fbf_detections = cc->Inputs().Tag(kDetectionsFbfTag).Get<Detections>();
    }
    FrameAnnotation frameAnnotation;
    if (cc->Inputs().HasTag(kFrameAnnotationTag) &&
        !cc->Inputs().Tag(kFrameAnnotationTag).IsEmpty()) {
      frameAnnotation =
          cc->Inputs().Tag(kFrameAnnotationTag).Get<FrameAnnotation>();
    }
    // std::cout << "   tracked_detections.size() " << tracked_detections.size()
    //           << std::endl;
    // std::cout << "   frameAnnotation.annotations_size() "
    //           << frameAnnotation.annotations_size() << std::endl;
    tracked_detections_.emplace_back(tracked_detections);
    fbf_detections_.emplace_back(fbf_detections);
    frameAnnotations_.emplace_back(frameAnnotation);
    // printf
    // frameAnnotationsToJson(frameAnnotation);
  }

  return absl::OkStatus();
}

absl::Status WriteToJsonFileCalculator::Close(CalculatorContext *cc) {
  // convert tracked_detections_ and frameAnnotations_ to json and save to file
  int numFrames = tracked_detections_.size();
  nlohmann::json jsonOut = nlohmann::json::array();
  for (int i = 0; i < numFrames; ++i) {
    nlohmann::json jFrame;
    jFrame["frame_id"] = i;
    auto jDetections =
        detectionsToJson(tracked_detections_[i], fbf_detections_[i]);
    jFrame["detections_2d"] = jDetections;
    auto jAnnotation = frameAnnotationsToJson(frameAnnotations_[i]);
    jFrame["detections_3d"] = jAnnotation;
    jsonOut.push_back(jFrame);
  }
  std::ofstream o(outJsonFilename_);
  o << std::setw(4) << jsonOut << std::endl;

  return absl::OkStatus();
}

} // namespace mediapipe
