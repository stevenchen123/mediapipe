# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "renderer_cpu",
    graph = "renderer_cpu.pbtxt",
    register_as = "RendererSubgraph",
    visibility = ["//visibility:public"],
    deps = [
        ":renderer_multilandmarks_cpu",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "renderer_multilandmarks_cpu",
    graph = "renderer_multi_landmarks_cpu.pbtxt",
    register_as = "RendererMultiLandmarksSubgraph",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "renderer_tracking_cpu",
    graph = "renderer_tracking_cpu.pbtxt",
    register_as = "RendererTrackingSubgraphCpu",
    visibility = ["//visibility:public"],
    deps = [
        ":renderer_multilandmarks_cpu",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "object_detection_cpu_new",
    graph = "object_detection_cpu_new.pbtxt",
    register_as = "ObjectDetectionSubgraphCpuNew",
    deps = [
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/modules/objectron/calculators:filter_detection_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "object_detection_from_json",
    graph = "object_detection_from_json.pbtxt",
    register_as = "ObjectDetectionFromJsonSubgraph",
    deps = [
        "//mediapipe/calculators/util:detections_from_json_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/modules/objectron/calculators:filter_detection_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "object_tracking_cpu",
    graph = "object_tracking_cpu.pbtxt",
    register_as = "ObjectTrackingSubgraphCpu",
    deps = [
        "//mediapipe/calculators/util:detection_unique_id_calculator",
        "//mediapipe/calculators/util:detections_to_timed_box_list_calculator",
        "//mediapipe/calculators/video:tracked_detection_manager_calculator",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/graphs/tracking/subgraphs:box_tracking_cpu",
    ],
)

mediapipe_simple_subgraph(
    name = "object_detection_3d_cpu",
    graph = "object_detection_3d_cpu.pbtxt",
    register_as = "ObjectDetection3dCpuSubgraph",
    deps = [
        "//mediapipe/modules/objectron:box_landmark_cpu",
        "//mediapipe/modules/objectron:object_detection_oid_v4_cpu",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:association_norm_rect_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",
        "//mediapipe/modules/objectron/calculators:frame_annotation_to_rect_calculator",
        "//mediapipe/modules/objectron/calculators:landmarks_to_frame_annotation_calculator",
        "//mediapipe/modules/objectron/calculators:lift_2d_frame_annotation_to_3d_calculator",
        "//mediapipe/modules/objectron/calculators:association_detection_annotation_by_order_calculator",
    ],
)