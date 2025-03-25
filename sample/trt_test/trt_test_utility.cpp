

#include "./trt_test_utility.h"
#include <array>
using namespace cv;

const std::vector<std::string> cocolabels{
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
};

cv::Mat draw_rst(cv::Mat image, const Algo::BoxArray& boxes)
{
    for (auto& box : boxes) {
        cv::Scalar color(0, 255, 0);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);
        auto& name       = cocolabels[box.class_label];
        auto  caption    = cv::format("%s %.2f", name.c_str(), box.confidence);
        int   text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(box.left - 3, box.top - 33), cv::Point(box.left + text_width, box.top), color, -1);
        cv::putText(image, caption, cv::Point(box.left, box.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    return image;
}

cv::Mat draw_rst(cv::Mat image, const std::vector<stResultItem>& item_list)
{
    Algo::BoxArray box_array;
    for (auto item : item_list) {
        if (item.points.size() == 2) {
            float     left  = item.points[0].x;
            float     top   = item.points[0].y;
            float     right = item.points[1].x;
            float     down  = item.points[1].y;
            float     conf  = item.confidence;
            int       label = item.code;
            Algo::Box box(left, top, right, down, conf, label);
            box_array.push_back(box);
        }
    }
    draw_rst(image, box_array);
    return image;
}
