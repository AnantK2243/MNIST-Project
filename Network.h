#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <random>
#include <fstream>

extern const int epoch_num;
extern const int num_hidden_neurons;
extern const int num_layers;
extern const int train_size;
extern double learning_rate;
extern const int neurons[];
extern const bool get_start;

extern int progress;
extern int image_size;
extern int total_train_size;
extern int total_test_size;

using namespace std;

class Network {
private:
    vector<vector<double>> biases;
    vector<vector<vector<double>>> weights;
    void load_file();
    double sigmoid(double val);
    double sigmoid_prime(double val);
    void feed_forward(vector<double> &curr, vector<double> &next, int layer);
    void feed_forward_sigmoid(vector<double> &curr, vector<double> &next, int layer);
    int reverse_int(int i);

public:
    Network();
    double calculate_cost(vector<double> &input_arr, int actual, vector<vector<double>>& delta_biases, vector<vector<vector<double>>>& delta_weights);
    void apply(vector<vector<double>>& delta_biases, vector<vector<vector<double>>>& delta_weights);
    vector<vector<double>> get_biases();
    vector<vector<vector<double>>> get_weights();
    int evaluate();
    void store();
};

#endif // NETWORK_H