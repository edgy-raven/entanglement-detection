#include <complex>
#include <random>
#include <vector>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <eigen3/Eigen/Dense>

#include <./operators.cpp>

std::vector<double> sample(const std::vector<Eigen::MatrixXcd> &b, int shard)
{
    static std::default_random_engine generator({(long unsigned int) shard});
    static std::normal_distribution<double> n01(0.0, 1.0);

    int N = b[0].rows();
    Eigen::VectorXcd z(N);
    for(int i = 0; i < N; ++i)
        z(i) = cd_t(n01(generator), n01(generator));
    z.normalize();

    return compute_correlations(z, b);
}

void write_row(std::ofstream *file_ptr, const std::vector<double> &x, int b_size, int ix, bool positive)
{
    for(int i = 0; i < b_size; ++i)
        if(i != ix) *file_ptr << x[i] << ",";
    *file_ptr << (positive ? "1" : "0") << "\n";
}

void generate_dataset(int qubits, int dataset_size, int shard)
{
    std::vector<Eigen::MatrixXcd> basis = construct_basis(qubits);
    int b_size = basis.size();
    std::vector<std::ofstream *> file_ptrs;
    std::vector<int> positive_counts;
    std::vector<int> negative_counts;
    int_map_t vecix_to_bitstr;
    int_map_t bitstr_to_vecix;

    file_ptrs.reserve(b_size);
    positive_counts.reserve(b_size);
    negative_counts.reserve(b_size);
    for(int ix = 0; ix < b_size; ++ix)
    {
        std::stringstream out_filename;
        out_filename << "/drv1/training_data/" << qubits << "_" << ix << "_" << shard << ".csv";
        std::ofstream *file_ptr = new std::ofstream(out_filename.str());
        *file_ptr << std::setprecision(2) << std::fixed; 
        file_ptrs.push_back(file_ptr);
        
        positive_counts.push_back(0);
        negative_counts.push_back(0);
    }
    std::tie(vecix_to_bitstr, bitstr_to_vecix) = generate_op_maps(qubits);

    int completed = 0;
    while(completed < b_size)
    {
        std::vector<double> x = sample(basis, shard);
        int max_ix = std::distance(x.begin(), std::max_element(x.begin(), x.end()));
        
        int ix_to_write;
        bool positive;
        if(positive_counts[max_ix] != dataset_size)
        {
            ix_to_write = max_ix;
            positive = true;
            positive_counts[max_ix] = positive_counts[max_ix] + 1;
        } else {
            bool found = false;
            for (int ix = 0; ix < b_size; ++ix)
            {
                if(negative_counts[ix] != dataset_size)
                {
                    found = true;
                    ix_to_write = ix;
                    positive = false;
                    negative_counts[ix] = negative_counts[ix] + 1;
                    break;
                }
            }
            if(!found) {
                // Here we have to apply a rotation to "create" a positive sample
                for(int ix = 0; ix < b_size; ++ix)
                {
                    if(positive_counts[ix] != dataset_size)
                    {
                        ix_to_write = ix;
                        positive = true;
                        positive_counts[ix] = positive_counts[ix] + 1;
                        break;
                    }
                }
                rotate(
                    x, 
                    vecix_to_bitstr.find(max_ix)->second, 
                    vecix_to_bitstr.find(ix_to_write)->second, 
                    qubits, 
                    b_size, 
                    vecix_to_bitstr, 
                    bitstr_to_vecix
                );
            }
        }
        if(positive_counts[ix_to_write] == dataset_size && negative_counts[ix_to_write] == dataset_size) 
            completed = completed + 1;
        write_row(file_ptrs[ix_to_write], x, b_size, ix_to_write, positive);
    }
    for(std::ofstream *ptr: file_ptrs)
        delete ptr;
}

int main(int argc, char *argv[])
{
    int NUM_SAMPLES = 33000, N_SHARDS = 6;
    int shard = atoi(argv[1]);
    int qubits_to_generate[] = {2, 3, 4, 5, 6};
    for(int qubits: qubits_to_generate)
        generate_dataset(qubits, NUM_SAMPLES / N_SHARDS, shard);
    return 0;
} 
