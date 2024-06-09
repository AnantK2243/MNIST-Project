#include "Network.h"
#include <cmath>
#include <ctime>
#include <iostream>

Network::Network() {
    biases.resize(num_layers - 1);
    weights.resize(num_layers - 1);

    if (get_start) {
        this->load_file();
        return;
    }

    default_random_engine generator(time(nullptr));
    normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 1; i < num_layers; ++i) {
        biases[i - 1].resize(neurons[i]);
        for (int j = 0; j < neurons[i]; ++j) {
            biases[i - 1][j] = distribution(generator);
        }
    }

    for (int i = 1; i < num_layers; ++i) {
        weights[i - 1].resize(neurons[i]);
        for (int j = 0; j < neurons[i]; ++j) {
            weights[i - 1][j].resize(neurons[i - 1]);
            for (int k = 0; k < neurons[i - 1]; ++k) {
                weights[i - 1][j][k] = distribution(generator);
            }
        }
    }
}

void Network::load_file() {
    ifstream w("weights.txt");
    ifstream b("biases.txt");

    for (int i = 0; i < num_layers - 1; i++) {
        biases[i].resize(neurons[i + 1]);
        weights[i].resize(neurons[i + 1]);
        for (int j = 0; j < neurons[i + 1]; j++) {
            string tempb = "";
            getline(b, tempb);
            double x = stod(tempb);
            biases[i][j] = x;
            
            weights[i][j].resize(neurons[i]);
            for (int k = 0; k < neurons[i]; k++) {
                string tempw = "";
                getline(w, tempw);
                weights[i][j][k] = stod(tempw);
            }
        }
    }

    w.close();
    b.close();
}

double Network::sigmoid(double val) {
    return 1.0 / (1.0 + exp(-val));
}

double Network::sigmoid_prime(double val) {
    double temp = sigmoid(val);
    return temp * (1 - temp);
}

void Network::feed_forward(vector<double> &curr, vector<double> &next, int layer) {
    int len = neurons[layer + 1];
    for (int i = 0; i < len; i++) {
        double total = 0;
        for (int j = 0; j < neurons[layer]; j++) {
            total += curr[j] * weights[layer][i][j];
        }
        total += biases[layer][i];
        next[i] = total;
    }
}

void Network::feed_forward_sigmoid(vector<double> &curr, vector<double> &next, int layer) {
    int len = neurons[layer + 1];
    for (int i = 0; i < len; i++) {
        double total = 0;
        for (int j = 0; j < neurons[layer]; j++) {
            total += curr[j] * weights[layer][i][j];
        }
        total += biases[layer][i];
        next[i] = sigmoid(total);
    }
}

double Network::calculate_cost(vector<double> &input_arr, int actual, vector<vector<double>>& delta_biases, vector<vector<vector<double>>>& delta_weights) {
    vector<double> hidden_arr(neurons[1], 0.0);
    vector<double> hidden_arr_z(neurons[1], 0.0);
    vector<double> hidden_arr_2(neurons[2], 0.0);
    vector<double> hidden_arr_z_2(neurons[2], 0.0);
    vector<double> res_z(neurons[3], 0.0);
    vector<double> res(neurons[3], 0.0);

    this->feed_forward(input_arr, hidden_arr_z, 0);
    for (int i = 0; i < neurons[1]; i++) hidden_arr[i] = sigmoid(hidden_arr_z[i]);

    this->feed_forward(hidden_arr, hidden_arr_z_2, 1);
    for (int i = 0; i < neurons[2]; i++) hidden_arr_2[i] = sigmoid(hidden_arr_z_2[i]);

    this->feed_forward(hidden_arr_2, res_z, 2);
    for (int i = 0; i < neurons[3]; i++) res[i] = sigmoid(res_z[i]);

    double cost = 0;
    for (int i = 0; i < neurons[num_layers - 1]; i++) {
        if (i == actual)
            cost += pow(res[i] - 1.0, 2);
        else
            cost += pow(res[i], 2);
    }

    vector<double> delta(neurons[num_layers - 1], 0.0);
    for (int j = 0; j < neurons[num_layers - 1]; j++) {
        double temp = (j == actual ? (res[j] - 1.0) : (res[j])) * sigmoid_prime(res_z[j]);
        delta[j] = temp;
        for (int k = 0; k < neurons[num_layers - 2]; k++) {
            delta_weights[num_layers - 2][j][k] += temp * hidden_arr_2[k];
        }
        delta_biases[num_layers - 2][j] += temp;
    }

    vector<double> delta2(neurons[num_layers - 2], 0.0);
    for (int j = 0; j < neurons[num_layers - 2]; j++) {
        double temp = 0;
        for (int m = 0; m < neurons[num_layers - 1]; m++) {
            temp += delta[m] * weights[num_layers - 2][m][j];
        }
        temp *= sigmoid_prime(hidden_arr_z_2[j]);
        delta2[j] = temp;
        for (int k = 0; k < neurons[num_layers - 3]; k++) {
            delta_weights[num_layers - 3][j][k] += temp * hidden_arr[k];
        }
        delta_biases[num_layers - 3][j] += temp;
    }

    for (int j = 0; j < neurons[num_layers - 3]; j++) {
        double temp = 0;
        for (int m = 0; m < neurons[num_layers - 2]; m++) {
            temp += delta2[m] * weights[num_layers - 3][m][j];
        }
        temp *= sigmoid_prime(hidden_arr_z[j]);
        for (int k = 0; k < neurons[num_layers - 4]; k++) {
            delta_weights[num_layers - 4][j][k] += temp * input_arr[k];
        }
        delta_biases[num_layers - 4][j] += temp;
    }

    return cost;
}

void Network::apply(vector<vector<double>>& delta_biases, vector<vector<vector<double>>>& delta_weights) {
    double lr = progress < 1000 ? learning_rate / (double) progress : 0.01;
    for (int i = 0; i < num_layers - 1; i++) {
        for (int j = 0; j < neurons[i + 1]; j++) {
            biases[i][j] -= lr * (delta_biases[i][j] / train_size);
            for (int k = 0; k < neurons[i]; k++) {
                weights[i][j][k] -= lr * (delta_weights[i][j][k] / train_size);
            }
        }
    }
}

vector<vector<double>> Network::get_biases() {
    return biases;
}

vector<vector<vector<double>>> Network::get_weights() {
    return weights;
}

int Network::reverse_int(int i) {
    return ((i & 255) << 24) + (((i >> 8) & 255) << 16) + (((i >> 16) & 255) << 8) + ((i >> 24) & 255);
}

int Network::evaluate() {
    ifstream img_file("t10k-images.idx3-ubyte", ios::binary);
    ifstream label_file("t10k-labels.idx1-ubyte", ios::binary);

    int img_magic_number = 0;
    int label_magic_number = 0;
    int label_number_of_items = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    img_file.read((char*)&img_magic_number, sizeof(int));
    img_magic_number = reverse_int(img_magic_number);
    img_file.read((char*)&number_of_images, sizeof(int));
    number_of_images = reverse_int(number_of_images);
    img_file.read((char*)&n_rows, sizeof(int));
    n_rows = reverse_int(n_rows);
    img_file.read((char*)&n_cols, sizeof(int));
    n_cols = reverse_int(n_cols);

    label_file.read((char*)&label_magic_number, sizeof(int));
    label_magic_number = reverse_int(label_magic_number);
    label_file.read((char*)&label_number_of_items, sizeof(int));
    label_number_of_items = reverse_int(label_number_of_items);

    int image_size = n_rows * n_cols;
    int total = 0;

    for (int n = 0; n < label_number_of_items; n++) {
        vector<double> input_arr(image_size, 0.0);

        for (int i = 0; i < image_size; i++) {
            unsigned char temp = 0;
            img_file.read((char*)&temp, sizeof(unsigned char));
            input_arr[i] = (double)temp / 255.0;
        }

        int actual = 0;
        label_file.read((char*)&actual, sizeof(unsigned char));

        vector<double> hidden_arr(neurons[1], 0.0);
        vector<double> hidden_arr_2(neurons[2], 0.0);
        vector<double> res(neurons[3], 0.0);
        this->feed_forward_sigmoid(input_arr, hidden_arr, 0);
        this->feed_forward_sigmoid(hidden_arr, hidden_arr_2, 1);
        this->feed_forward_sigmoid(hidden_arr_2, res, 2);

        double max = -1;
        int maxi = -1;
        for (int i = 0; i < neurons[3]; i++) {
            if (maxi == -1 || res[i] > max) {
                max = res[i];
                maxi = i;
            }
        }

        total += (maxi == actual);
    }

    img_file.close();
    label_file.close();

    return total;
}

void Network::store() {
    ofstream w("weights.txt", ios::out | ios::trunc);
    ofstream b("biases.txt", ios::out | ios::trunc);

    for (int i = 0; i < num_layers - 1; i++) {
        for (int j = 0; j < neurons[i + 1]; j++) {
            b << biases[i][j] << endl;
            for (int k = 0; k < neurons[i]; k++) {
                w << weights[i][j][k] << endl;
            }
        }
    }

    w.close();
    b.close();
}