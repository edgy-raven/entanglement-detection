#ifndef ENTANGLEMENT_DETECTION_OPERATOR_CPP
#define ENTANGLEMENT_DETECTION_OPERATOR_CPP

#include <complex>
#include <map>
#include <set>
#include <vector>

#include <eigen3/Eigen/Dense>

typedef std::complex<double> cd_t;
typedef std::map<int, int> int_map_t;

cd_t C_0(0, 0);
cd_t C_1(1, 0);
cd_t C_I(0, 1);
Eigen::MatrixXcd PAULI_X = (Eigen::MatrixXcd(2,2) << C_0, C_1, C_1, C_0).finished();
Eigen::MatrixXcd PAULI_Y = (Eigen::MatrixXcd(2,2) << C_0, -C_I, C_I, C_0).finished();
Eigen::MatrixXcd PAULI_Z = (Eigen::MatrixXcd(2,2) << C_1, C_0, C_0, -C_1).finished();
std::vector<Eigen::MatrixXcd> PAULI {PAULI_X, PAULI_Y, PAULI_Z};


std::vector<int> build_op_bitstrs(int qubits)
{
    if(qubits == 1) return std::vector<int> {1, 2, 3};

    std::vector<int> carry = build_op_bitstrs(qubits - 1);
    std::vector<int> to_return;

    for(int bit=1; bit <= 3; ++bit)
        for(int bitstr: carry)
            to_return.push_back(bitstr << 2 | bit);
    return to_return;
}

std::pair<int_map_t, int_map_t> generate_op_maps(int qubits)
{
    std::vector<int> bitstrs = build_op_bitstrs(qubits);
    int b_size = bitstrs.size();

    int_map_t vecix_to_bitstr;
    int_map_t bitstr_to_vecix;
    for(int ix = 0; ix < b_size; ++ix)
    {
        vecix_to_bitstr[ix] = bitstrs[ix];
        bitstr_to_vecix[bitstrs[ix]] = ix;
    }
    return std::make_pair(vecix_to_bitstr, bitstr_to_vecix);
}

Eigen::MatrixXcd kron2(const Eigen::MatrixXcd &p_mtx, const Eigen::MatrixXcd &o)
{
    int r = o.rows(), c = o.cols();
    Eigen::MatrixXcd z(r << 1, c << 1);
    z.block(0, 0, r, c) = p_mtx(0, 0) * o;
    z.block(0, c, r, c) = p_mtx(0, 1) * o;
    z.block(r, 0, r, c) = p_mtx(1, 0) * o;
    z.block(r, c, r, c) = p_mtx(1, 1) * o;
    return z; 
}

std::vector<Eigen::MatrixXcd> construct_basis(int qubits)
{
    if (qubits == 1) return PAULI;
    
    std::vector<Eigen::MatrixXcd> carry = construct_basis(qubits - 1);
    std::vector<Eigen::MatrixXcd> result;
    result.reserve(3 * carry.size());

    for(Eigen::MatrixXcd p : PAULI)
        for(Eigen::MatrixXcd o: carry)
            result.push_back(kron2(p, o));
    return result;
}

std::vector<double> compute_correlations(
    const Eigen::VectorXcd &z,
    const std::vector<Eigen::MatrixXcd> &b)
{
    int N = b[0].rows();
    Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(N, N);
    rho.selfadjointView<Eigen::Upper>().rankUpdate(z);

    std::vector<double> result;
    result.reserve(b.size());
    for (Eigen::MatrixXcd b_element: b)
    {
        cd_t tr = cd_t(0,0);
        for (int i = 1; i < N; ++i)
            tr = tr + rho.cwiseProduct(b_element).col(i).head(i).sum();
        tr = 2.0 * tr + rho.cwiseProduct(b_element).diagonal().sum();
        result.push_back(tr.real() * tr.real());
    }
    return result;
}

void rotate(
    std::vector<double> &x, 
    int src_bitstr, 
    int tgt_bitstr, 
    int qubits, 
    int b_size, 
    const int_map_t &vecix_to_bitstr, 
    const int_map_t &bitstr_to_vecix)
{
    std::set<int> swapped_ixs;
    for(int ix = 0; ix < b_size; ++ix)
    {
        if(swapped_ixs.find(ix) != swapped_ixs.end()) continue;
        int op_bitstr = vecix_to_bitstr.find(ix)->second;
        for(int j = 0; j < qubits; ++j)
        {
            int shift = j << 1;

            int op_j = (op_bitstr >> shift) & 3;
            int src_j = (src_bitstr >> shift) & 3;
            int tgt_j = (tgt_bitstr >> shift) & 3;
            
            if(src_j != tgt_j) 
            {
                if(op_j == src_j) op_bitstr = (op_bitstr & (~(3 << shift))) | (tgt_j << shift);
                if(op_j == tgt_j) op_bitstr = (op_bitstr & (~(3 << shift))) | (src_j << shift);
            }
        }
        int vecix_to_swap = bitstr_to_vecix.find(op_bitstr)->second;
        swapped_ixs.insert(vecix_to_swap);
        double tmp = x[vecix_to_swap];
        x[vecix_to_swap] = x[ix];
        x[ix] = tmp;
    }
}

bool commute(int op_bitstr1, int op_bitstr2, int qubits)
{
    int diffs = 0;
    for (int i = 0; i < qubits; ++i)
    {
        if ((op_bitstr1 & 3) != (op_bitstr2 & 3)) diffs++;
        op_bitstr1 = op_bitstr1 >> 2;
        op_bitstr2 = op_bitstr2 >> 2;
    }
    return (diffs & 1) == 0;
}

struct tracker_t {
    int n_measurements = 0;
    double total = 0.0;

    double do_measurement(const std::vector<double> &x, int op)
    {
        double measurement_value = x[op];
        n_measurements++;
        total += measurement_value;
        return measurement_value;
    }
};

#endif
