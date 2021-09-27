#include "boruvka_superpixel.h"
#include <iostream>
#include <algorithm>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void transposeAndSplit(unsigned char *out, unsigned char *in, int h, int w) {
  int hw = h * w, hw2 = 2 * h * w;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      out[x * h + y] = in[3 * (y * w + x) + 0];
      out[hw + x * h + y] = in[3 * (y * w + x) + 1];
      out[hw2 + x * h + y] = in[3 * (y * w + x) + 2];
    }
}

template <class T> void transpose(T *out, T *in, int h, int w) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      out[x * h + y] = in[y * w + x];
}

template <class T> void transposeT(T *out, T *in, int h, int w) {
  for (int x = 0; x < w; ++x)
    for (int y = 0; y < h; ++y)
      out[y * w + x] = in[x * h + y];
}

cv::Mat compute_sp(cv::Mat &img_in, int superpixel_count) {
  int nr_neighbor = 4;
  int h = img_in.rows;
  int w = img_in.cols;
  cv::Mat edge_in = cv::Mat::zeros(h, w, CV_8U);
  cv::GaussianBlur(img_in, img_in, cv::Size2i(7, 7), 3);
  cv::Canny(img_in, edge_in, 30, 50);
  unsigned char *img = new unsigned char[3 * h * w];
  unsigned char *imgt = new unsigned char[3 * h * w];
  unsigned char *color = new unsigned char[3 * h * w];
  unsigned char *edge = new unsigned char[h * w];
  unsigned char *edget = new unsigned char[h * w];
  //int *labelt = new int[h * w];
  BoruvkaSuperpixel seg;
  cv::Mat img_lab = cv::Mat::zeros(h, w, CV_8UC3);
  cvtColor(img_in, img_lab, cv::COLOR_BGR2Lab);
  seg.build_2d(h, w, 3, img_lab.data, edge_in.data);
  //seg.init(h, w, nr_neighbor);
  //transposeAndSplit(imgt, (unsigned char *)img_in.data, h, w);
  //transpose(edget, (unsigned char *)edge_in.data, h, w);

  //int nvertex, nregion, N;
  //seg.buildTree(imgt, edget);
  //int *label = seg.getLabel(superpixel_count);
  int *labelt = seg.label(superpixel_count);
  //transposeT(labelt, label, h, w);

  cv::Mat result(h, w, CV_32S, labelt);
  result = result.clone();
  int label_max = *std::max_element(labelt, labelt + (h * w));
  img_in.convertTo(img_in, CV_64FC3);
  cv::Mat M = cv::Mat::zeros(label_max + 1, 3, CV_64F);
  cv::Mat A = cv::Mat::ones(label_max + 1, 1, CV_64F);
  for (int i = 0; i < img_in.rows; ++i) {
    for (int j = 0; j < img_in.cols; ++j) {
      int id = result.at<int>(i, j);
      for (int c = 0; c < 3; ++c) {
        M.at<double>(id, c) =
            M.at<double>(id, c) + img_in.at<cv::Vec3d>(i, j)[c];
      }
      A.at<double>(id, 0) += 1;
    }
  }
  for (int i = 0; i < label_max; ++i) {
    for (int c = 0; c < 3; ++c) {
      M.at<double>(i, c) /= A.at<double>(i);
    }
  }
  cv::Mat colorImg = cv::Mat::zeros(h, w, CV_64FC3);
  for (int i = 0; i < img_in.rows; ++i) {
    for (int j = 0; j < img_in.cols; ++j) {
      int id = result.at<int>(i, j);
      for (int c = 0; c < 3; ++c) {
        colorImg.at<cv::Vec3d>(i, j)[c] = M.at<double>(id, c);
      }
    }
  }
  colorImg.convertTo(colorImg, CV_8UC3);
  return colorImg;
}

void sh_video(const std::string &input_video_path,
              const std::string &output_video_path, int superpixel_count) {
  cv::VideoCapture cap(input_video_path);
  auto width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  auto height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  auto fps = cap.get(cv::CAP_PROP_FPS);
  auto frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
  cv::VideoWriter writer(output_video_path, CV_FOURCC('H', '2', '6', '4'), fps,
                         cv::Size2i(width, height));
  cv::Mat img;
  int processed = 0;
  while (cap.read(img)) {
    auto sp_img = compute_sp(img, superpixel_count);
    writer.write(sp_img);
    std::cout << "Progress: " << processed++ << "/" << frame_count << std::endl;
  }
  writer.release();
}

void sh_image(const std::string &input_image_path,
              const std::string &output_image_path, int superpixel_count) {
  cv::Mat img = cv::imread(input_image_path, cv::IMREAD_COLOR);
  auto sp_img = compute_sp(img, superpixel_count);
  cv::imwrite(output_image_path, sp_img);
}

int main(int argc, char **argv) {
  if (argc >= 5) {
    std::string mode = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    int superpixel_count = std::atoi(argv[4]);
    if (mode == "image") {
      sh_image(input_path, output_path, superpixel_count);
    } else if (mode == "video") {
      sh_video(input_path, output_path, superpixel_count);
    } else {
      std::cerr << "Invalid processing type, please use 'image' or 'video'"
                << std::endl;
      std::exit(1);
    }
  } else {
    std::cerr
        << "Incorrect number of arguments. Usage: ./boruvkasupix "
	   "<type(image/video)> <input_path> <output_path> <superpixel_count>"
        << std::endl;
    std::exit(1);
  }

  return 0;
}
