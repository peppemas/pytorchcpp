#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

using namespace cv;

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 3

void PrintVersion() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "Torch OpenMP version: " << torch::get_openmp_version() << std::endl;
    std::cout << "Torch MKL version: " << torch::get_mkl_version() << std::endl;
    std::cout << "Torch MKL DNN version: " << torch::get_mkldnn_version() << std::endl;
}

int main(const int argc, char** argv ) {

    PrintVersion();

    if (argc != 4)
    {
        std::cout << "Usage: pytorchcpp <Model> <Label> <Image>" << std::endl;
        return -1;
    }

    // load torch model
    std::cout << "Loading " << argv[1] <<  std::endl;
    torch::jit::script::Module module = torch::jit::load(argv[1]);
    module.to(at::kCPU); // enable CPU (we are running on guest VM)
    std::cout << "Model " << argv[1] << " successfully loaded" << std::endl;

    // load labels
    std::vector<std::string> labels;
    std::cout << "Loading " << argv[2] <<  std::endl;
    std::ifstream ifs(argv[2]);
    if (!ifs) {
        std::cerr << "Error opening file " << argv[2] << std::endl;
        return -1;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        labels.push_back(line);
    }
    std::cout << labels.size() << " labels loaded" << std::endl;

    // load image
    std::cout << "Loading " << argv[3] <<  std::endl;
    const Mat image = imread(argv[3]);
    if (image.empty() || !image.data)
    {
        std::cerr << "Error opening file " << argv[3] << std::endl;
        return -1;
    }

    // image preprocessing for model inferencing
    Mat resized_image;
    try {
        std::cout << "\t\tBefore cvtColor image size: " << image.size() << std::endl;
        cvtColor(image, image, COLOR_BGR2RGB);
        std::cout << "\t\tAfter cvtColor image size: " << image.size() << std::endl;
        // scale image to fit
        Size scale(kIMAGE_SIZE, kIMAGE_SIZE);
        std::cout << "\t\tkIMAGE_SIZE: " << kIMAGE_SIZE << std::endl;
        std::cout << "\t\tBefore resize image size: " << image.size() << std::endl;
        resize(image, resized_image, scale);
        std::cout << "\t\tAfter resize image size: " << resized_image.size() << std::endl;
        // convert [unsigned int] to [float]
        resized_image.convertTo(resized_image, CV_32FC3, 1.0f / 255.0f);
    } catch (const std::exception& e) {
        std::cerr << "Error during image preprocessing: " << e.what() << std::endl;
        return -1;
    }

    // execute inferencing...
    const std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    auto input_tensor = torch::from_blob(resized_image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});

    // Dimension 0 (N - Batch) remains at position 0.
    // Dimension 3 (C - Channels) moves to position 1.
    // Dimension 1 (H - Height) moves to position 2.
    // Dimension 2 (W - Width) moves to position 3.
    input_tensor = input_tensor.permute({0, 3, 1, 2});

    // normalization of red,green,blue color channels
    input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
    input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
    input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

    input_tensor = input_tensor.to(at::kCPU);
    torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();
    const auto end = std::chrono::steady_clock::now();

    auto results = out_tensor.sort(-1, true);
    auto softmax_tensor = std::get<0>(results)[0].softmax(0);
    auto index_tensor = std::get<1>(results)[0];

    std::cout << "=== RESULTS ===" << std::endl;
    for (int i = 0; i < kTOP_K; ++i) {
        auto idx = index_tensor[i].item<int>();
        auto probability = softmax_tensor[i].item<float>() * 100.0f;
        std::cout << "top: " << i << " idx: " << idx << " label: " << labels[idx] << " (" << probability << " %)" << std::endl;
    }
    std::cout << "Inferencing in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << std::endl;

    /* just for debug...
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", resized_image);
    waitKey(0);
    */

    return 0;
}