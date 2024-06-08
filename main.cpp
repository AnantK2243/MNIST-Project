#include "network.h"
#include <iostream>
#include <random>
#include <ctime>
#include <fstream>

using namespace std;

const int epoch_num = 200;
const int num_hidden_neurons = 20;
const int num_layers = 3;
const int train_size = 5000;
double learning_rate = 2;
const int neurons[] = {784, num_hidden_neurons, 10};
const bool get_start = 1;

int progress = 0;
int image_size;
int total_train_size;
int total_test_size;

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

int main() {
    ifstream img_file("train-images.idx3-ubyte", ios::binary);
    ifstream label_file("train-labels.idx1-ubyte", ios::binary);

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

        for (int i = 0; i < train_size; i++) {
            int index;

            do{
                index = distribution(generator);

                if(used[index] == false){
                    used[index] = true;
                    break;
                }

            }while(true);

            vector<double> input_arr = vector<double>(image_size, 0.0);

            for(int j = 0; j < image_size; j++)
                input_arr[j] = (double) image_arr[index][j] / 255.0;
            int actual = label_arr[index];


            total_cost += net.calculate_cost(input_arr, actual, delta_biases_total, delta_weights_total);
        }

        net.apply(delta_biases_total, delta_weights_total);

        int num_correct = net.evaluate();

        cout << "Epoch (" << x << "): " << total_cost / (neurons[num_layers - 1]) << "\t - " << num_correct << "/10000" << endl;
    }

    net.store();
}