#include <iostream>
#include <fstream>
#include <vector>
#include "Functions/loss.h"
#include "Tensor/Tensor.h"
#include "Optimizers/BaseOptimizer.h"
#include "NN/Modules/Linear.h"
#include "NN/Modules/ReLu.h"
#include "NN/Modules/Loss/MSELoss.h"

using namespace cortex;

std::vector<std::vector<f32_t>> Read_Boston() {
    std::ifstream inf;
    inf.open("../Dataset/BostonHousing/bostonHousing.txt");
    std::string line;

    std::vector<std::vector<f32_t>> _data;
    while (getline(inf, line)) {
        std::vector<f32_t> row;
        std::stringstream ss(line);
        std::string item;
        while (ss >> item) {
            row.push_back(std::stod(item));
        }

        _data.push_back(row);
    }

    inf.close();

    return _data;
}


void boston_housing_price_prediction(const Tensor& input, const Tensor& label) {
    Linear linear1(dtype::f32, DeviceType::cpu, 13, 30, true);
    ReLu relu1(dtype::f32, DeviceType::cpu);
    Linear linear2(dtype::f32, DeviceType::cpu, 30, 1, true);
    MSELoss mseLoss(dtype::f32, DeviceType::cpu);

    std::vector<std::shared_ptr<Tensor>> params = {linear1.get_weight(), linear1.get_bias(), linear2.get_weight(), linear2.get_bias()};

    SGD sgd(params, 0.02f);

    for (int i = 0; i < 50; i++) {
        Tensor l1 = relu1.forward(linear1.forward(input));
        Tensor l2 = linear2.forward(l1);

        Tensor loss = FMSELoss(label, l2, 2);
        loss.backward();
        sgd.step();
        sgd.zero_grads();

        std::cout << "Loss in Epoch " << i << ": " << loss.to_string() << std::endl;
    }
}

void train() {
    std::vector<std::vector<f32_t>> raw_data = Read_Boston();

    // Normalize the data
    for (auto& column : raw_data) {
        const double minVal = *std::ranges::min_element(column);
        const double maxVal = *std::ranges::max_element(column);

        for (auto& val : column) {
            val = (val - minVal) / (maxVal - minVal);
        }
    }

    // Split x and y
    std::vector<f32_t> input_r_data;
    std::vector<f32_t> label_r_data;
    for (int i = 0; i < raw_data.size(); i++) {
        for (int j = 0; j < raw_data[0].size(); j++) {
            if (j != raw_data[0].size() - 1) {
                input_r_data.push_back(raw_data[i][j]);
            }
            else {
                label_r_data.push_back(raw_data[i][j]);
            }
        }
    }

    Tensor input({static_cast<unsigned>(raw_data.size()), static_cast<unsigned>(raw_data[0].size() - 1)}, dtype::f32, DeviceType::cpu);
    Tensor labels({static_cast<unsigned>(raw_data.size()), 1}, dtype::f32, DeviceType::cpu);

    input.initialize_with(input_r_data);
    labels.initialize_with(label_r_data);

    boston_housing_price_prediction(input, labels);
}