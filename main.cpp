#include "Network.h"
#include <iostream>
#include <random>
#include <ctime>
#include <fstream>
#include <thread>
#include <mutex>

using namespace std;

const int max_threads = 16;

const int epoch_num = 1500;
const int num_hidden_neurons = 16;
const int num_layers = 4;
const int train_size = 20000;
double learning_rate = 10;
const int neurons[] = {784, num_hidden_neurons, num_hidden_neurons, 10};
const bool get_start = 0;

int progress = 1;
int image_size;
int total_train_size;
int total_test_size;

mutex mtx;

int reverse_int(int i) {
    return ((i & 255) << 24) + (((i >> 8) & 255) << 16) + (((i >> 16) & 255) << 8) + ((i >> 24) & 255);
}

void get_train_image(ifstream &file, vector<unsigned char> &arr) {
    for (int i = 0; i < image_size; i++) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(unsigned char));
        arr[i] = temp;
    }
}

void get_train_file(ifstream &img_file, ifstream &label_file) {
    int temp = 0;
    int n_rows = 0;
    int n_cols = 0;

    img_file.read((char*)&temp, sizeof(int));
    img_file.read((char*)&temp, sizeof(int));
    total_train_size = reverse_int(temp);

    img_file.read((char*)&n_rows, sizeof(int));
    n_rows = reverse_int(n_rows);
    img_file.read((char*)&n_cols, sizeof(int));
    n_cols = reverse_int(n_cols);

    label_file.read((char*)&temp, sizeof(int));
    label_file.read((char*)&temp, sizeof(int));

    image_size = n_rows * n_cols;
}

void run_epoch(Network &net, vector<double> &input_arr, int actual, vector<vector<double>>& delta_biases, vector<vector<vector<double>>>& delta_weights){
    net.calculate_cost(input_arr, actual, delta_biases, delta_weights);

    lock_guard<mutex> lock(mtx);
}

int main() {
    ifstream img_file("train-images.idx3-ubyte", ios::binary);
    ifstream label_file("train-labels.idx1-ubyte", ios::binary);

    int start = -1;
    int num_correct = -1;

    get_train_file(img_file, label_file);

    default_random_engine generator(time(NULL));
    uniform_int_distribution<int> distribution(0, total_train_size - 1);

    vector<vector<unsigned char>> image_arr(total_train_size);
    vector<unsigned char> label_arr = vector<unsigned char> (total_train_size, (unsigned char) 0);

    for(int i = 0; i < total_train_size; i++){

        vector<unsigned char> temp = vector<unsigned char> (image_size, (unsigned char) 0);
        get_train_image(img_file, temp);
        image_arr[i] = temp;

        unsigned char actual = 0;
        label_file.read((char*)&actual, sizeof(unsigned char));
        label_arr[i] = actual;
    }

    img_file.close();
    label_file.close();

    cout << "File Opened" << endl;

    Network net;

    for (int x = 0; x < epoch_num; x++) {
        double total_cost = 0;

        vector<vector<double>> delta_biases_total(num_layers - 1);
        vector<vector<vector<double>>> delta_weights_total(num_layers - 1);

        for (int i = 1; i < num_layers; i++) {
            delta_biases_total[i - 1] = vector<double>(neurons[i], 0.0);
            delta_weights_total[i - 1] = vector<vector<double>>(neurons[i], vector<double>(neurons[i - 1], 0.0));
        }

        vector<bool> used = vector<bool>(total_train_size, false);

        vector<thread> thread_list;

        int batch_size = train_size / max_threads;

        for (int i = 0; i < max_threads; i++) {
            thread_list.emplace_back([&net, &image_arr, &label_arr, &delta_biases_total, &delta_weights_total, batch_size, &generator, &distribution, &used]() {
                for (int j = 0; j < batch_size; j++) {
                    int index;

                    do {
                        index = distribution(generator);
                        if (!used[index]) {
                            used[index] = true;
                            break;
                        }
                    } while (true);

                    vector<double> input_arr(image_size, 0.0);
                    for (int k = 0; k < image_size; k++)
                        input_arr[k] = static_cast<double>(image_arr[index][k]) / 255.0;
                    int actual = label_arr[index];

                    run_epoch(net, input_arr, actual, delta_biases_total, delta_weights_total);
                }
            });
        }

        for (auto &t : thread_list) {
            t.join();
        }
        
        net.apply(delta_biases_total, delta_weights_total);

        num_correct = net.evaluate();

        if (start = -1) start = num_correct;

        cout << "Epoch (" << x << "): " << num_correct << "/10000" << endl;
    }

    if(num_correct > start)
        net.store();
}