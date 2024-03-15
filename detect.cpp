#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <stack>

#include <iostream>
#include <fstream>
#include <sstream>

#include <eigen3/Eigen/Sparse>

#include <./operators.cpp>


template<class T> std::vector<T> read_vec_from_line(std::string line, int sz=0)
{
    std::vector<T> result;
    if(sz) result.reserve(sz);

    std::stringstream ss(line);
    for(T i; ss >> i; )
    {
        result.push_back(i);
        if(ss.peek() == ',')
            ss.ignore();
    }
    return result;
}

std::vector<std::vector<int>> read_cliques(int qubits)
{
    std::stringstream clique_filename;
    clique_filename << "cliques/cliques_" << qubits << ".csv";
    std::ifstream clique_file(clique_filename.str());

    std::vector<std::vector<int>> all_cliques;
    std::string line;
    while(std::getline(clique_file, line))
    {
        std::vector<int> clique;
        clique.reserve((2 << (qubits -1)) + !(qubits & 1));
        all_cliques.push_back(read_vec_from_line<int>(line));
    }
    return all_cliques;
}

std::vector<std::vector<int>> generate_anticommuting_map(int qubits, const int_map_t &vecix_to_bitstr)
{
    std::vector<std::vector<int>> to_return;

    int b_size = vecix_to_bitstr.size();
    for(int i = 0; i < b_size; ++i)
    {
        std::vector<int> anticomm;
        int bitstr = vecix_to_bitstr.find(i)->second;
        for(int j = i + 1; j < b_size; ++j)
            if (!commute(bitstr, vecix_to_bitstr.find(j)->second, qubits)) anticomm.push_back(j);
        to_return.push_back(anticomm);
    }
    for(int i = 0; i < b_size; ++i)
        for(int j : to_return[i])
            if(j > i) to_return[j].push_back(i);
    return to_return;
}

void update_priorities(
    std::vector<double> &prio,
    int op,
    double measurement_value,
    const std::vector<std::vector<int>> &anticommuting_map)
{
    for(int vec_ix: anticommuting_map[op]) prio[vec_ix] += measurement_value;
    prio[op] = 99999.0;    // arbitrary large value to ensure measurement isn't used again
}

int lask_detect(
    const std::vector<double> &x,
    const std::vector<std::vector<int>> &all_cliques, 
    const std::vector<std::vector<int>> &anticommuting_map)
{
    tracker_t tracker;
    int n_cliques = all_cliques.size(), clique_sz = all_cliques[0].size();
    // i tracks which clique, j tracks location within clique i
    int i = 0, j = 0, op = 0;
   
    std::vector<double> prio(x.size(), 0.0);
    while(i < n_cliques && j < clique_sz)
    {
        op = all_cliques[i][j];
        double measurement_value = tracker.do_measurement(x, op);
        update_priorities(prio, op, measurement_value, anticommuting_map);
        if(tracker.total > 1.0) break;
        if (measurement_value > 0.16) ++j;
        else if(++i < n_cliques)
            for(j = 0; j < clique_sz; ++j) if(all_cliques[i][j] != all_cliques[i-1][j]) break;
    }
    while(tracker.total <= 1.0)
    {
        op = std::distance(std::begin(prio), std::min_element(std::begin(prio), std::end(prio)));
        update_priorities(prio, op, tracker.do_measurement(x, op), anticommuting_map);
    }
    return tracker.n_measurements;
}


int do_seq_measurment(const std::vector<double> &x)
{
    tracker_t t;
    for(int i = 0; t.total <= 1.0 && i < x.size(); ++i) t.do_measurement(x, i);
    return t.n_measurements;
}

int random_detect(const std::vector<double> &x)
{
    std::vector<double> x_copy(x);
    // start at xx...x
    std::random_shuffle(x_copy.begin() + 1, x_copy.end());
    return do_seq_measurment(x_copy);
}

int optimal_detect(const std::vector<double> &x)
{
    std::vector<double> x_copy(x);
    std::sort(x_copy.begin(), x_copy.end(), std::greater<double>());
    return do_seq_measurment(x_copy);
}


struct tree_t {
    std::vector<int> feature;
    std::vector<double> threshold;
    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> p;
    std::vector<int> n;
};
typedef std::vector<tree_t> forest_t;

std::vector<forest_t> load_all_forests(int qubits, int b_size)
{
    std::vector<forest_t> all_forests;
    all_forests.reserve(b_size);

    for(int op = 0; op < b_size; ++op)
    {
        std::stringstream forest_filename;
        forest_filename << "/drv1/models/" << qubits << "_"  << op << ".txt";
        std::ifstream forest_file(forest_filename.str());

        forest_t forest;
        std::string line;
        while(std::getline(forest_file, line))
        {
            int sz = std::stoi(line);
            tree_t t;
            std::getline(forest_file, line);
            t.feature = read_vec_from_line<int>(line, sz);
            std::getline(forest_file, line);
            t.threshold = read_vec_from_line<double>(line, sz);
            std::getline(forest_file, line);
            t.left = read_vec_from_line<int>(line, sz);
            std::getline(forest_file, line);
            t.right = read_vec_from_line<int>(line, sz);            
            std::getline(forest_file, line);
            t.p = read_vec_from_line<int>(line, sz);
            std::getline(forest_file, line);
            t.n = read_vec_from_line<int>(line, sz);
            forest.push_back(t);
        }
        for(tree_t tree : forest)
            for(int i = 0; i < tree.feature.size(); ++i)
                if(tree.feature[i] >= op) ++tree.feature[i];
        all_forests.push_back(forest);
    }
    return all_forests;
}

struct tree_mask_t {
    std::vector<bool> mask;
    int p;
    int n;
};
typedef std::vector<tree_mask_t> forest_mask_t;

double avg_priority(const forest_mask_t &forest_mask)
{
    std::vector<double> scores;
    scores.reserve(forest_mask.size());

    for (tree_mask_t mask: forest_mask) 
        if(mask.p > 0 && mask.n > 0) 
            scores.push_back(((double) mask.p) / (mask.p + mask.n));
    std::sort(scores.begin(), scores.end(), std::greater<double>());
    
    // Try olympic average to smooth out sample noise    
    int sz = scores.size(), mid_q = sz >> 3;
    int count = 0;
    double total = 0.0;
    for(int i = mid_q; i < sz - mid_q; ++i)
    {
        ++count;
        total += scores[i];
    }
    return total / count;
}

void update_mask(const tree_t &tree, tree_mask_t &mask, int op, double measurement_value)
{
    std::stack<int> node_stack;
    node_stack.push(0);
    while(!node_stack.empty())
    {
        int node_value = node_stack.top();
        node_stack.pop();
        bool reachable = node_value >= 0;
        int node_ix = reachable ? node_value : -node_value;
        
        if(!mask.mask[node_ix]) continue;
        mask.mask[node_ix] = reachable;
        if(tree.left[node_ix] == -1)   // leaf node
        {
            if(!reachable)
            {
                mask.p -= tree.p[node_ix];
                mask.n -= tree.n[node_ix];
            }
        } else {
            // branch
            int mult_left = 1, mult_right = 1;
            if(reachable)
            {
                if(tree.feature[node_ix] == op)
                    (measurement_value > tree.threshold[node_ix] ? mult_left : mult_right) = -1;
            } else {
                mult_left = -1;
                mult_right = -1;
            }
            node_stack.push(mult_left*tree.left[node_ix]);
            node_stack.push(mult_right*tree.right[node_ix]);
        }
    }
}

int forest_detect(const std::vector<double> &x, const std::vector<forest_t> &all_forests, bool debug=false)
{
    std::vector<forest_mask_t> all_masks;
    all_masks.reserve(x.size());

    for(forest_t forest : all_forests)
    {
        forest_mask_t forest_mask;
        forest_mask.reserve(forest.size());
        for(tree_t tree : forest) {
            int p = 0, n = 0;
            for(int v : tree.p) p += v;
            for(int v : tree.n) n += v;
            tree_mask_t mask;
            mask.mask = std::vector<bool>(tree.feature.size(), true);
            mask.p = p;
            mask.n = n;
            forest_mask.push_back(mask);
        }
        all_masks.push_back(forest_mask);
    }
    std::vector<double> prio;
    prio.reserve(all_masks.size());
    if(!debug)
        prio.push_back(999.0);
    for(int i = (debug ? 0 : 1); i < all_masks.size(); ++i) prio.push_back(avg_priority(all_masks[i]));

    tracker_t tracker;
    while(tracker.total <= 1.0)
    {
        int op = std::distance(std::begin(prio), std::max_element(std::begin(prio), std::end(prio)));
        double measurement_value = tracker.do_measurement(x, op);
        if(debug)
            std::cout << "measurement:" << op << " value:" << measurement_value << std::endl;
        for(int i = 0; i < prio.size(); ++i)
        {
            if(prio[i] < 0) continue;
            if(i == op) prio[i] = -1.0;
            else {
                for(int j = 0; j < all_forests[i].size(); ++j)
                    update_mask(all_forests[i][j], all_masks[i][j], op, measurement_value);
                prio[i] = avg_priority(all_masks[i]);
            }
        }
    }
    return tracker.n_measurements;
}


typedef Eigen::Triplet<cd_t> spt_t;
double INVSQRT_2 = 1.0 / sqrt(2.0);
std::vector<spt_t> sp_H = {
    spt_t(0, 0, cd_t(INVSQRT_2, 0)), spt_t(0, 1, cd_t(INVSQRT_2, 0)),
    spt_t(1, 0, cd_t(INVSQRT_2, 0)), spt_t(1, 1, cd_t(-INVSQRT_2, 0))
};
std::vector<spt_t> sp_T = {spt_t(0, 0, C_1), spt_t(1, 1, cd_t(INVSQRT_2, INVSQRT_2))};
std::vector<spt_t> sp_I = {spt_t(0, 0, C_1), spt_t(1, 1, C_1)};
std::vector<spt_t> sp_NOT = {spt_t(0, 1, C_1), spt_t(1, 0, C_1)};
std::vector<spt_t> sp_ZERO = {spt_t(0, 0, C_1)};
std::vector<spt_t> sp_ONE = {spt_t(1, 1, C_1)};

std::vector<spt_t> sp_kron2(
    const std::vector<spt_t> &p_mtx, const std::vector<Eigen::Triplet<cd_t>> &o, int sz)
{
    std::vector<spt_t> new_triplets;
    new_triplets.reserve(p_mtx.size() * o.size());
    for(spt_t s: p_mtx)
        for(spt_t t: o) 
            new_triplets.push_back(spt_t(t.row() + sz*s.row(), t.col() + sz*s.col(), t.value()*s.value()));
    return new_triplets;
}

Eigen::SparseMatrix<cd_t> gen_sys_gate(int sys, const std::vector<spt_t> &U, int qubits)
{
    std::vector<spt_t> entries = (sys == 0) ? U : sp_I;
    int sz = 2;
    for(int i = 1; i < qubits; ++i)
    {
        entries = sp_kron2((sys == i) ? U : sp_I, entries, sz);
        sz = sz << 1;
    }
    Eigen::SparseMatrix<cd_t> spm(sz, sz);
    spm.setFromTriplets(entries.begin(), entries.end());
    return spm;
}

Eigen::SparseMatrix<cd_t> gen_cnot(int ctl_sys, int op_sys, int qubits)
{
    Eigen::SparseMatrix<cd_t> ctl = gen_sys_gate(ctl_sys, sp_ZERO, qubits);

    std::vector<spt_t> entries = (ctl_sys == 0) ? sp_ONE : (op_sys == 0 ? sp_NOT : sp_I);
    int sz = 2;
    for(int i = 1; i < qubits; ++i)
    {
        entries = sp_kron2((ctl_sys == i) ? sp_ONE : (op_sys == i ? sp_NOT : sp_I), entries, sz);
        sz = sz << 1;
    }
    Eigen::SparseMatrix<cd_t> op(sz, sz);
    op.setFromTriplets(entries.begin(), entries.end());
    return ctl + op;
}

std::vector<double> sample_computable(const std::vector<Eigen::MatrixXcd> &b, int qubits, int shard)
{
    int N = b[0].rows(), n_gates = 12 * qubits * qubits, lb_gates = n_gates >> 1, count = 0;

    static std::default_random_engine generator({(long unsigned int) shard});
    static std::uniform_int_distribution<int> gate_chooser(0, 2);
    std::uniform_int_distribution<int> system_chooser(0, qubits - 1);
    std::uniform_int_distribution<int> continue_chooser(0, lb_gates - 1);

    Eigen::SparseMatrix<cd_t> G(N, N);
    G.setIdentity();
    while((count++ < lb_gates) || (continue_chooser(generator) > 0))
    {
        int gate_type = gate_chooser(generator);
        int sys = system_chooser(generator);

        Eigen::SparseMatrix<cd_t> G_n;
        switch(gate_type){
            case 0:
                G_n = gen_sys_gate(sys, sp_H, qubits);
                break;
            case 1:
                G_n = gen_sys_gate(sys, sp_T, qubits);
                break;
            case 2:
                int ctl_sys;
                do { ctl_sys = system_chooser(generator); } while(ctl_sys == sys);
                G_n = gen_cnot(ctl_sys, sys, qubits);
                break;
        }
        G = G_n * G; 
    }
    Eigen::VectorXcd z(N);
    for(int i = 0; i < N; ++i) z(i) = G.coeff(i, 0);
    z.normalize();
    return compute_correlations(z, b);
}

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        std::vector<Eigen::MatrixXcd> b = construct_basis(3);
        std::vector<forest_t> all_forests = load_all_forests(3, b.size());
        
        std::vector<double> x;
        double alpha = 5.0 * M_PI / 71.0;
        Eigen::VectorXcd d3_1(8);
        d3_1 << C_0, C_1, C_1, C_0, C_1, C_0, C_0, C_0;
        Eigen::VectorXcd d3_2(8);
        d3_2 << C_0, C_0, C_0, C_1, C_0, C_1, C_1, C_0;
        Eigen::VectorXcd dicke = cos(alpha)*d3_2 + sin(alpha)*d3_1;
        dicke.normalize();

        x = compute_correlations(dicke, b);
        forest_detect(x, all_forests, true); 
        return 0;
    }
    int shard = atoi(argv[1]);
    int qubits = (argc == 3) ? atoi(argv[2]) : 3;
    for(; qubits < 7; ++qubits){
        int_map_t vecix_to_bitstr;
        int_map_t bitstr_to_vecix;
        std::tie(vecix_to_bitstr, bitstr_to_vecix) = generate_op_maps(qubits);
        std::vector<Eigen::MatrixXcd> b = construct_basis(qubits);

        std::vector<std::vector<int>> all_cliques = read_cliques(qubits);
        std::vector<std::vector<int>> anticommuting_map = generate_anticommuting_map(qubits, vecix_to_bitstr);

        std::vector<forest_t> all_forests = load_all_forests(qubits, b.size());
        
        std::vector<int> random_res(b.size(), 0);
        std::vector<int> optimal_res(b.size(), 0);
        std::vector<int> lask_res(b.size(), 0);
        std::vector<int> forest_res(b.size(), 0);
        for(int i = 0; i < 350; ++i)
        {
            std::vector<double> x = sample_computable(b, qubits, shard);
            int max_ix = std::distance(x.begin(), std::max_element(x.begin(), x.end()));
            rotate(    
                x,
                vecix_to_bitstr.find(max_ix)->second, 
                vecix_to_bitstr.find(0)->second, 
                qubits,
                b.size(), 
                vecix_to_bitstr, 
                bitstr_to_vecix
            );
            random_res[random_detect(x)]++;
            optimal_res[optimal_detect(x)]++;
            lask_res[lask_detect(x, all_cliques, anticommuting_map)]++; 
            forest_res[forest_detect(x, all_forests)]++;
        }
        std::cout << "qubits = " << qubits << std::endl;
        for(int i = 1; i < b.size(); ++i)
        {
            random_res[i] += random_res[i-1];
            optimal_res[i] += optimal_res[i-1];
            lask_res[i] += lask_res[i-1];
            forest_res[i] += forest_res[i-1];
        }
        for(int i = 0; i < b.size(); ++i)
            std::cout << random_res[i] << "," << optimal_res[i] << "," 
                      << lask_res[i] << "," << forest_res[i] << std::endl;
        std::cout << "------" << std::endl;
    }
    return 0;
}
